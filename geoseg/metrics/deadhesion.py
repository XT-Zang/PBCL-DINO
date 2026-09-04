from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None


PAIR_BUCKET_NAME = {0: "touching", 1: "1_2", 2: "3_5"}


@dataclass(frozen=True)
class CachedTile:
    tile_id: str
    scene_id: str
    species_domain: str
    gt_mask: np.ndarray
    instance_map: np.ndarray
    gt_areas: np.ndarray
    pairs: tuple[tuple[int, int, str], ...]


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask
    if ndi is not None:
        return ndi.binary_dilation(mask, iterations=radius)
    result = mask.copy()
    for _ in range(radius):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        expanded = np.zeros_like(result)
        for y in range(3):
            for x in range(3):
                expanded |= padded[y : y + result.shape[0], x : x + result.shape[1]]
        result = expanded
    return result


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask
    if ndi is not None:
        return ndi.binary_erosion(mask, iterations=radius, border_value=0)
    return ~_dilate(~mask, radius)


def _boundary(mask: np.ndarray, width: int) -> np.ndarray:
    return _dilate(mask, width) & ~_erode(mask, width)


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    mask = np.asarray(mask, dtype=bool)
    if ndi is not None:
        labels, count = ndi.label(mask)
        return labels.astype(np.int32, copy=False), int(count)
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    count = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or labels[y, x]:
                continue
            count += 1
            stack = [(y, x)]
            labels[y, x] = count
            while stack:
                cy, cx = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = count
                        stack.append((ny, nx))
    return labels, count


def _binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "fg_iou": _safe_rate(tp, tp + fp + fn),
    }


def _boundary_metrics(pred: np.ndarray, gt: np.ndarray, width: int, tolerance: int) -> dict[str, float | int]:
    pred_b = _boundary(pred, width)
    gt_b = _boundary(gt, width)
    pred_match = int((pred_b & _dilate(gt_b, tolerance)).sum())
    gt_match = int((gt_b & _dilate(pred_b, tolerance)).sum())
    pred_pixels = int(pred_b.sum())
    gt_pixels = int(gt_b.sum())
    precision = pred_match / pred_pixels if pred_pixels else (1.0 if not gt_pixels else 0.0)
    recall = gt_match / gt_pixels if gt_pixels else (1.0 if not pred_pixels else 0.0)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "boundary_pred_match": pred_match,
        "boundary_gt_match": gt_match,
        "boundary_pred_pixels": pred_pixels,
        "boundary_gt_pixels": gt_pixels,
        "boundary_f1": float(f1),
    }


def _instance_metrics(
    pred: np.ndarray,
    instance_map: np.ndarray,
    gt_areas: np.ndarray,
    pairs: tuple[tuple[int, int, str], ...],
    *,
    min_overlap_pixels: int,
    min_overlap_ratio: float,
) -> dict[str, float | int]:
    pred_labels, pred_count = _label_components(pred)
    pred_areas = np.bincount(pred_labels.ravel(), minlength=pred_count + 1)[1:]
    gt_count = int(len(gt_areas))
    intersections = np.zeros((gt_count, pred_count), dtype=np.int64)
    if gt_count and pred_count:
        valid = (instance_map > 0) & (pred_labels > 0)
        if np.any(valid):
            encoded = (instance_map[valid].astype(np.int64) - 1) * pred_count + (pred_labels[valid].astype(np.int64) - 1)
            intersections = np.bincount(encoded, minlength=gt_count * pred_count).reshape(gt_count, pred_count)

    linked = np.zeros((gt_count, pred_count), dtype=bool)
    dominant = np.zeros(gt_count, dtype=np.int32)
    for gt_idx, area in enumerate(gt_areas):
        if not pred_count:
            break
        required = min(int(area), max(int(min_overlap_pixels), int(math.ceil(int(area) * min_overlap_ratio))))
        linked[gt_idx] = intersections[gt_idx] >= required
        best = int(np.argmax(intersections[gt_idx]))
        if intersections[gt_idx, best] >= required:
            dominant[gt_idx] = best + 1

    gt_per_pred = linked.sum(axis=0) if pred_count else np.zeros(0, dtype=int)
    pred_per_gt = linked.sum(axis=1) if gt_count else np.zeros(0, dtype=int)
    merge_extra = int(np.maximum(0, gt_per_pred - 1).sum())
    split_extra = int(np.maximum(0, pred_per_gt - 1).sum())

    pair_totals = {name: 0 for name in PAIR_BUCKET_NAME.values()}
    pair_separated = {name: 0 for name in PAIR_BUCKET_NAME.values()}
    for first, second, bucket in pairs:
        pair_totals[bucket] += 1
        first_pred = int(dominant[first - 1]) if 0 < first <= gt_count else 0
        second_pred = int(dominant[second - 1]) if 0 < second <= gt_count else 0
        if first_pred > 0 and second_pred > 0 and first_pred != second_pred:
            pair_separated[bucket] += 1

    pq_candidates: list[tuple[float, int, int]] = []
    for gt_idx, gt_area in enumerate(gt_areas):
        for pred_idx in range(pred_count):
            inter = int(intersections[gt_idx, pred_idx])
            if not inter:
                continue
            union = int(gt_area) + int(pred_areas[pred_idx]) - inter
            iou = inter / max(1, union)
            if iou > 0.5:
                pq_candidates.append((iou, gt_idx, pred_idx))
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    pq_iou_sum = 0.0
    for iou, gt_idx, pred_idx in sorted(pq_candidates, reverse=True):
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        pq_iou_sum += iou
    pq_tp = len(matched_gt)
    pq_fp = pred_count - pq_tp
    pq_fn = gt_count - pq_tp

    dcp_total = pair_totals["1_2"] + pair_totals["3_5"]
    dcp_separated = pair_separated["1_2"] + pair_separated["3_5"]
    return {
        "gt_instances": gt_count,
        "pred_components": pred_count,
        "count_abs_error": abs(pred_count - gt_count),
        "count_denominator": gt_count,
        "merge_extra": merge_extra,
        "merge_denominator": gt_count,
        "split_extra": split_extra,
        "split_denominator": gt_count,
        "pq_iou_sum": float(pq_iou_sum),
        "pq_tp": pq_tp,
        "pq_fp": pq_fp,
        "pq_fn": pq_fn,
        "dcp_total": dcp_total,
        "dcp_separated": dcp_separated,
    }


def _read_manifest(root: Path, split: str) -> list[dict[str, Any]]:
    path = root / "splits" / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class CachedDeadhesionEvaluator:
    """Validation structure metrics backed by the same PBCL target files used for training."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        split: str = "val",
        instances_coco: str | Path | None = None,
        boundary_width: int = 2,
        boundary_tolerance: int = 2,
        min_overlap_pixels: int = 8,
        min_overlap_ratio: float = 0.05,
        tile_ids: set[str] | None = None,
    ) -> None:
        del instances_coco
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.boundary_width = int(boundary_width)
        self.boundary_tolerance = int(boundary_tolerance)
        self.min_overlap_pixels = int(min_overlap_pixels)
        self.min_overlap_ratio = float(min_overlap_ratio)
        target_root = self.dataset_root / "instance_targets" / "pbcl_gap5"
        self._tiles: dict[str, CachedTile] = {}
        for item in _read_manifest(self.dataset_root, split):
            tile_id = str(item["tile_id"])
            if tile_ids is not None and tile_id not in tile_ids:
                continue
            target_path = target_root / f"{tile_id}.npz"
            if target_path.is_file():
                with np.load(target_path, allow_pickle=False) as payload:
                    instance_map = np.asarray(payload["instance_map"], dtype=np.int32)
                    raw_pairs = np.asarray(payload["instance_pairs"], dtype=np.int64)
                gt_count = int(instance_map.max())
                gt_areas = np.bincount(instance_map.ravel(), minlength=gt_count + 1)[1:].astype(np.int64, copy=False)
                pairs = tuple(
                    (int(first), int(second), PAIR_BUCKET_NAME[int(bucket)])
                    for first, second, bucket in raw_pairs
                    if int(bucket) in PAIR_BUCKET_NAME
                )
            else:
                if not bool(item.get("is_background", False)):
                    raise FileNotFoundError(f"Missing PBCL target for positive validation tile: {target_path}")
                height = int(item.get("valid_height") or item.get("tile_size") or 512)
                width = int(item.get("valid_width") or item.get("tile_size") or 512)
                instance_map = np.zeros((height, width), dtype=np.int32)
                gt_areas = np.zeros(0, dtype=np.int64)
                pairs = ()
            self._tiles[tile_id] = CachedTile(
                tile_id=tile_id,
                scene_id=str(item.get("scene_id", "")),
                species_domain=str(item.get("species_domain", "")),
                gt_mask=instance_map > 0,
                instance_map=instance_map,
                gt_areas=gt_areas,
                pairs=pairs,
            )
        self.reset()

    @property
    def cached_tile_count(self) -> int:
        return len(self._tiles)

    def reset(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def update_batch(self, pred_masks: Any, gt_masks: Any, tile_ids: list[str] | tuple[str, ...]) -> None:
        del gt_masks
        arr = pred_masks.detach().cpu().numpy() if isinstance(pred_masks, torch.Tensor) else np.asarray(pred_masks)
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.ndim != 3 or len(tile_ids) != arr.shape[0]:
            raise ValueError("pred_masks must be a binary batch aligned with tile_ids")
        for index, tile_id in enumerate(tile_ids):
            tile = self._tiles[str(tile_id)]
            height, width = tile.gt_mask.shape
            pred = np.asarray(arr[index, :height, :width], dtype=bool)
            row = {
                "scene_id": tile.scene_id,
                "species_domain": tile.species_domain,
            }
            row.update(_binary_metrics(pred, tile.gt_mask))
            row.update(_boundary_metrics(pred, tile.gt_mask, self.boundary_width, self.boundary_tolerance))
            row.update(
                _instance_metrics(
                    pred,
                    tile.instance_map,
                    tile.gt_areas,
                    tile.pairs,
                    min_overlap_pixels=self.min_overlap_pixels,
                    min_overlap_ratio=self.min_overlap_ratio,
                )
            )
            self.rows.append(row)

    @staticmethod
    def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
        def total(key: str) -> float:
            return float(sum(float(row.get(key, 0.0)) for row in rows))
        tp, fp, fn, tn = total("tp"), total("fp"), total("fn"), total("tn")
        pred_match, gt_match = total("boundary_pred_match"), total("boundary_gt_match")
        pred_pixels, gt_pixels = total("boundary_pred_pixels"), total("boundary_gt_pixels")
        bp = pred_match / pred_pixels if pred_pixels else (1.0 if not gt_pixels else 0.0)
        br = gt_match / gt_pixels if gt_pixels else (1.0 if not pred_pixels else 0.0)
        bf1 = 2.0 * bp * br / (bp + br) if bp + br else 0.0
        pq_iou, pq_tp, pq_fp, pq_fn = total("pq_iou_sum"), total("pq_tp"), total("pq_fp"), total("pq_fn")
        pq_den = pq_tp + 0.5 * pq_fp + 0.5 * pq_fn
        return {
            "fg_iou": _safe_rate(tp, tp + fp + fn),
            "bg_iou": _safe_rate(tn, tn + fp + fn),
            "boundary_f1": float(bf1),
            "merge_rate": _safe_rate(total("merge_extra"), total("merge_denominator")),
            "split_rate": _safe_rate(total("split_extra"), total("split_denominator")),
            "normalized_object_count_abs_error": _safe_rate(total("count_abs_error"), total("count_denominator")),
            "pq": float(pq_iou / pq_den) if pq_den else 1.0,
            "close_pair_separation_recall": _safe_rate(total("dcp_separated"), total("dcp_total")),
        }

    def summary(self) -> dict[str, Any]:
        by_scene: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        by_species: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.rows:
            by_scene[str(row["scene_id"])].append(row)
            by_species[str(row["species_domain"])].append(row)
        return {
            "sample_count": len(self.rows),
            "cached_tile_count": self.cached_tile_count,
            "overall": self._aggregate(self.rows),
            "by_scene": {key: self._aggregate(value) for key, value in sorted(by_scene.items())},
            "by_species_domain": {key: self._aggregate(value) for key, value in sorted(by_species.items())},
        }

    @staticmethod
    def flat_metrics(summary: Mapping[str, Any]) -> dict[str, float]:
        overall = summary["overall"]
        return {
            "boundary_f1": float(overall["boundary_f1"]),
            "merge_rate": float(overall["merge_rate"]),
            "split_rate": float(overall["split_rate"]),
            "close_pair_recall": float(overall["close_pair_separation_recall"]),
            "count_nmae": float(overall["normalized_object_count_abs_error"]),
            "pq": float(overall["pq"]),
        }


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
