from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler, default_collate


CLASSES = ("background", "foreground")
PALETTE = [[0, 0, 0], [255, 255, 255]]

AQUAV3_MEAN = np.array([0.4251161849, 0.4954026022, 0.4909278856, 0.2641421088], dtype=np.float32)
AQUAV3_STD = np.array([0.2261392075, 0.2244978271, 0.229417928, 0.2071894144], dtype=np.float32)
AQUAV3_RAW_CLIP_MIN = 100.0
AQUAV3_RAW_CLIP_MAX = 60000.0
AQUAV3_RAW_OFFSET = 100.0
AQUAV3_RAW_SCALE = 59900.0


def _read_raster(path: Path, indexes: list[int]) -> np.ndarray:
    import rasterio

    with rasterio.open(str(path)) as ds:
        return ds.read(indexes=indexes).astype(np.float32)


def _read_mask(path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        import rasterio

        with rasterio.open(str(path)) as ds:
            arr = ds.read(1)
    else:
        image = Image.open(path).convert("L")
        if image.size != (shape_hw[1], shape_hw[0]):
            image = image.resize((shape_hw[1], shape_hw[0]), Image.Resampling.NEAREST)
        arr = np.asarray(image)
    if arr.shape != shape_hw:
        raise ValueError(f"Mask shape {arr.shape} does not match image shape {shape_hw} for {path}")
    return (arr.astype(np.float32) > 0).astype(np.float32)


def normalized_aquav3_to_rgb(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().float().numpy()
    arr = arr * AQUAV3_STD[: arr.shape[0], None, None] + AQUAV3_MEAN[: arr.shape[0], None, None]
    if arr.shape[0] >= 3:
        rgb = arr[[2, 1, 0]]
    elif arr.shape[0] == 2:
        rgb = np.concatenate([arr, arr[:1]], axis=0)
    else:
        rgb = np.repeat(arr[:1], 3, axis=0)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8).transpose(1, 2, 0)


def save_aquav3_prediction_preview(image: torch.Tensor, mask: torch.Tensor, pred: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = normalized_aquav3_to_rgb(image)
    gt = (mask.detach().cpu().squeeze().numpy() > 0.5).astype(np.uint8) * 255
    pred_arr = (pred.detach().cpu().squeeze().numpy() > 0.5).astype(np.uint8) * 255
    overlay = rgb.copy()
    overlay[pred_arr > 0] = (0.45 * overlay[pred_arr > 0] + 0.55 * np.array([255, 64, 32])).astype(np.uint8)
    overlay[gt > 0] = (0.55 * overlay[gt > 0] + 0.45 * np.array([32, 220, 120])).astype(np.uint8)
    canvas = np.concatenate(
        [
            rgb,
            np.repeat(gt[..., None], 3, axis=2),
            np.repeat(pred_arr[..., None], 3, axis=2),
            overlay,
        ],
        axis=1,
    )
    Image.fromarray(canvas).save(path)


class AquaV3ManifestDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 512,
        band_indices: list[int] | None = None,
        augment: bool = False,
        raw_clip_min: float = AQUAV3_RAW_CLIP_MIN,
        raw_clip_max: float = AQUAV3_RAW_CLIP_MAX,
        raw_offset: float = AQUAV3_RAW_OFFSET,
        raw_scale: float = AQUAV3_RAW_SCALE,
        normalize_mean: list[float] | np.ndarray | None = None,
        normalize_std: list[float] | np.ndarray | None = None,
        instance_target_root: str | Path | None = None,
        resize_mode: str = "resize",
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        self.root = Path(root)
        self.split = split
        self.image_size = int(image_size)
        self.band_indices = list(band_indices if band_indices is not None else [0, 1, 2, 3])
        self.rasterio_indexes = [index + 1 for index in self.band_indices]
        self.augment = bool(augment)
        self.resize_mode = str(resize_mode).strip().lower()
        if self.resize_mode not in {"resize", "pad"}:
            raise ValueError(f"Unsupported resize_mode: {resize_mode}")
        self.raw_clip_min = float(raw_clip_min)
        self.raw_clip_max = float(raw_clip_max)
        self.raw_offset = float(raw_offset)
        self.raw_scale = float(raw_scale)
        self.normalize_mean = np.asarray(normalize_mean if normalize_mean is not None else AQUAV3_MEAN, dtype=np.float32)
        self.normalize_std = np.asarray(normalize_std if normalize_std is not None else AQUAV3_STD, dtype=np.float32)
        if instance_target_root is None:
            self.instance_target_root = None
        else:
            instance_root = Path(instance_target_root)
            self.instance_target_root = instance_root if instance_root.is_absolute() else self.root / instance_root
        if self.raw_scale <= 0:
            raise ValueError("raw_scale must be positive")
        if len(self.normalize_mean) < len(self.band_indices) or len(self.normalize_std) < len(self.band_indices):
            raise ValueError("normalize_mean/std must cover selected bands")
        if np.any(self.normalize_std[: len(self.band_indices)] <= 0):
            raise ValueError("normalize_std values must be positive")
        self.entries = self._load_entries()
        if not self.entries:
            raise RuntimeError(f"No AquaV3 entries for split={split} under {self.root}")

    def _load_entries(self) -> list[dict[str, Any]]:
        split_path = self.root / "splits" / f"{self.split}.jsonl"
        manifest_path = split_path if split_path.exists() else self.root / "manifest.jsonl"
        entries: list[dict[str, Any]] = []
        with open(manifest_path, encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                if manifest_path.name == "manifest.jsonl" and record.get("split") != self.split:
                    continue
                entries.append(record)
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def is_negative(self) -> list[bool]:
        return [bool(entry.get("is_background", False)) for entry in self.entries]

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        image_path = self.root / entry["image"]
        raw = _read_raster(image_path, self.rasterio_indexes)
        arr = np.clip(raw, self.raw_clip_min, self.raw_clip_max)
        arr = (arr - self.raw_offset) / self.raw_scale
        mean = self.normalize_mean[: len(self.band_indices), None, None]
        std = self.normalize_std[: len(self.band_indices), None, None]
        image = torch.from_numpy(((arr - mean) / std).astype(np.float32, copy=False)).contiguous()

        instance_map, instance_pairs = self._load_instance_target(entry, tuple(image.shape[-2:]))
        mask_path = entry.get("semantic")
        if mask_path:
            mask_arr = _read_mask(self.root / mask_path, tuple(image.shape[-2:]))
        elif instance_map is not None:
            mask_arr = (instance_map.squeeze(0).numpy() > 0).astype(np.float32)
        else:
            mask_arr = np.zeros(tuple(image.shape[-2:]), dtype=np.float32)
        mask = torch.from_numpy(mask_arr).unsqueeze(0).contiguous()

        supervision = mask
        if instance_map is not None:
            supervision = torch.cat((mask, instance_map.float()), dim=0)
        if self.augment:
            image, supervision = self._augment(image, supervision)
        if self.resize_mode == "pad":
            valid_mask = torch.ones(
                (1, *image.shape[-2:]), dtype=torch.bool, device=image.device
            )
            image, supervision = self._pad(image, supervision)
            valid_mask, _ = self._pad(valid_mask, valid_mask)
        else:
            image, supervision = self._resize(image, supervision)
            valid_mask = None
        mask = supervision[:1]
        if instance_map is not None:
            instance_map = supervision[1:2].round().to(dtype=torch.long).contiguous()

        item = {
            "img": image,
            "image": image,
            "gt_semantic_seg": mask,
            "mask": mask,
            "img_id": entry["tile_id"],
            "tile_id": entry["tile_id"],
            "path": str(image_path),
            "is_background": bool(entry.get("is_background", False)),
        }
        if instance_map is not None:
            item["instance_map"] = instance_map
            item["instance_pairs"] = instance_pairs
        if valid_mask is not None:
            item["valid_mask"] = valid_mask
        return item

    def _load_instance_target(
        self,
        entry: dict[str, Any],
        shape_hw: tuple[int, int],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.instance_target_root is None:
            return None, None
        tile_id = str(entry["tile_id"])
        if bool(entry.get("is_background", False)):
            label_map = np.zeros(shape_hw, dtype=np.int64)
            pairs = np.empty((0, 3), dtype=np.int64)
        else:
            target_path = self.instance_target_root / f"{tile_id}.npz"
            if not target_path.exists():
                raise FileNotFoundError(f"Missing PBCL instance target for positive tile {tile_id}: {target_path}")
            with np.load(target_path, allow_pickle=False) as target:
                if "instance_map" not in target or "instance_pairs" not in target:
                    raise ValueError(f"PBCL target must contain instance_map and instance_pairs: {target_path}")
                label_map = np.asarray(target["instance_map"], dtype=np.int64)
                pairs = np.asarray(target["instance_pairs"], dtype=np.int64)
            if label_map.shape != shape_hw:
                raise ValueError(
                    f"PBCL instance map shape {label_map.shape} does not match image shape {shape_hw}: {target_path}"
                )
            if np.any(label_map < 0):
                raise ValueError(f"PBCL instance map must contain non-negative ids: {target_path}")
            if pairs.ndim != 2 or pairs.shape[1] != 3:
                raise ValueError(f"PBCL instance pairs must have shape (N,3): {target_path}")
            if pairs.size:
                if np.any(pairs[:, :2] <= 0):
                    raise ValueError(f"PBCL pair instance ids must be positive: {target_path}")
                if np.any((pairs[:, 2] < 0) | (pairs[:, 2] > 2)):
                    raise ValueError(f"PBCL pair bucket codes must be 0, 1, or 2: {target_path}")
                present_ids = set(np.unique(label_map).tolist())
                referenced_ids = set(pairs[:, :2].reshape(-1).tolist())
                missing_ids = sorted(referenced_ids - present_ids)
                if missing_ids:
                    raise ValueError(f"PBCL pairs reference missing instance ids {missing_ids}: {target_path}")
        return (
            torch.from_numpy(label_map).unsqueeze(0).contiguous(),
            torch.from_numpy(pairs).contiguous(),
        )

    def _augment(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            image = torch.flip(image, [2])
            mask = torch.flip(mask, [2])
        if random.random() < 0.25:
            image = torch.flip(image, [1])
            mask = torch.flip(mask, [1])
        if random.random() < 0.35:
            turns = random.randint(0, 3)
            if turns:
                image = torch.rot90(image, turns, [1, 2])
                mask = torch.rot90(mask, turns, [1, 2])
        if random.random() < 0.35:
            brightness = random.uniform(0.85, 1.15)
            contrast = random.uniform(0.85, 1.20)
            image = image * brightness
            mean_val = image.mean()
            image = (image - mean_val) * contrast + mean_val
        return image.contiguous(), mask.contiguous()

    def _resize(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.shape[-2:] == (self.image_size, self.image_size):
            return image, mask
        image = F.interpolate(
            image.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=(self.image_size, self.image_size), mode="nearest").squeeze(0)
        return image.contiguous(), mask.contiguous()

    def _pad(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = image.shape[-2:]
        if height > self.image_size or width > self.image_size:
            raise ValueError(
                f"Cannot pad shape {(height, width)} to {(self.image_size, self.image_size)}"
            )
        padding = (0, self.image_size - width, 0, self.image_size - height)
        return (
            F.pad(image, padding, mode="constant", value=0).contiguous(),
            F.pad(mask, padding, mode="constant", value=0).contiguous(),
        )


def make_balanced_sampler(
    dataset: AquaV3ManifestDataset,
    *,
    generator: torch.Generator | None = None,
) -> WeightedRandomSampler:
    negatives = dataset.is_negative
    neg_count = sum(1 for value in negatives if value)
    pos_count = len(negatives) - neg_count
    weights = [0.5 / max(1, neg_count) if is_neg else 0.5 / max(1, pos_count) for is_neg in negatives]
    return WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def aquav3_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Default-collate fixed fields while preserving variable-length PBCL pairs."""
    if not batch:
        raise ValueError("aquav3_collate requires a non-empty batch")
    has_instance = ["instance_map" in item or "instance_pairs" in item for item in batch]
    if any(has_instance) and not all(
        "instance_map" in item and "instance_pairs" in item for item in batch
    ):
        raise ValueError("instance_map and instance_pairs must be present for every item in a batch")
    if not any(has_instance):
        return default_collate(batch)
    fixed_items = [{key: value for key, value in item.items() if key != "instance_pairs"} for item in batch]
    collated = default_collate(fixed_items)
    collated["instance_pairs"] = [item["instance_pairs"] for item in batch]
    return collated
