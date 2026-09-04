from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from geoseg.experiments.reproducibility import make_generator
from geoseg.runtime.specs import ExperimentSpec

from .aquav3_dataset import AquaV3ManifestDataset, aquav3_collate, make_balanced_sampler


class AquaV3DataModule:
    """Own AquaV3 datasets and deterministic loaders without building them at import time."""

    def __init__(self, spec: ExperimentSpec) -> None:
        if spec.data.get("dataset") not in {
            "aquav3",
            "aquadataset",
            "vegasdenseclosepbcl_v1",
        }:
            raise ValueError(
                "AquaV3DataModule requires an Aqua manifest dataset, "
                f"got {spec.data.get('dataset')!r}"
            )
        self.spec = spec
        self.train_dataset: AquaV3ManifestDataset | None = None
        self.val_dataset: AquaV3ManifestDataset | None = None
        self.test_dataset: AquaV3ManifestDataset | None = None
        self._train_loader: DataLoader | None = None
        self._val_loader: DataLoader | None = None
        self._test_loader: DataLoader | None = None

    @classmethod
    def from_legacy_config(cls, config: Any) -> "AquaV3DataModule":
        instance = cls.__new__(cls)
        instance.spec = None
        instance._train_loader = getattr(config, "train_loader", None)
        instance._val_loader = getattr(config, "val_loader", None)
        instance._test_loader = getattr(config, "test_loader", None)
        instance.train_dataset = getattr(config, "train_dataset", getattr(instance._train_loader, "dataset", None))
        instance.val_dataset = getattr(config, "val_dataset", getattr(instance._val_loader, "dataset", None))
        instance.test_dataset = getattr(config, "test_dataset", getattr(instance._test_loader, "dataset", None))
        return instance

    def _dataset(self, split: str, *, augment: bool) -> AquaV3ManifestDataset:
        return AquaV3ManifestDataset(
            root=self.spec.data["root"],
            split=split,
            image_size=int(self.spec.data["image_size"]),
            band_indices=list(self.spec.data.get("band_indices", [0, 1, 2, 3])),
            augment=augment,
            raw_clip_min=float(self.spec.data.get("raw_clip_min", 100.0)),
            raw_clip_max=float(self.spec.data.get("raw_clip_max", 60000.0)),
            raw_offset=float(self.spec.data.get("raw_offset", 100.0)),
            raw_scale=float(self.spec.data.get("raw_scale", 59900.0)),
            normalize_mean=self.spec.data.get("normalize_mean"),
            normalize_std=self.spec.data.get("normalize_std"),
            instance_target_root=self.spec.data.get("instance_target_root"),
            resize_mode=str(self.spec.data.get("resize_mode", "resize")),
        )

    def setup(self, stage: str | None = None) -> None:
        normalized = stage.lower() if isinstance(stage, str) else None
        if normalized in {None, "fit", "train"}:
            if self.train_dataset is None:
                self.train_dataset = self._dataset("train", augment=True)
            if self.val_dataset is None:
                self.val_dataset = self._dataset("val", augment=False)
        elif normalized in {"validate", "validation", "val"}:
            if self.val_dataset is None:
                self.val_dataset = self._dataset("val", augment=False)
        if normalized in {None, "test", "predict"} and self.test_dataset is None:
            self.test_dataset = self._dataset("test", augment=False)

    @staticmethod
    def _worker_options(loader_spec: dict[str, Any], workers: int) -> dict[str, Any]:
        persistent = bool(loader_spec.get("persistent_workers", False)) and workers > 0
        options: dict[str, Any] = {"persistent_workers": persistent}
        prefetch = loader_spec.get("prefetch_factor")
        if workers > 0 and prefetch is not None:
            options["prefetch_factor"] = int(prefetch)
        return options

    def train_dataloader(self) -> DataLoader:
        if self._train_loader is not None:
            return self._train_loader
        self.setup("fit")
        assert self.train_dataset is not None
        seed = int(self.spec.runtime["seed"])
        loader_spec = dict(self.spec.loader)
        workers = int(loader_spec["train_workers"])
        sampler_stream = loader_spec.get("sampler_generator_stream")
        sampler_generator = make_generator(seed, stream=int(sampler_stream)) if sampler_stream is not None else None
        sampler = make_balanced_sampler(self.train_dataset, generator=sampler_generator) if bool(
            loader_spec.get("balanced_sampler", True)
        ) else None
        loader_streams = dict(loader_spec.get("loader_generator_streams", {}))
        loader_options: dict[str, Any] = {}
        if "train" in loader_streams:
            loader_options["generator"] = make_generator(seed, stream=int(loader_streams["train"]))
        if loader_spec.get("collate") == "aquav3":
            loader_options["collate_fn"] = aquav3_collate
        self._train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=int(loader_spec["train_batch_size"]),
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=workers,
            pin_memory=bool(loader_spec.get("pin_memory", True)),
            drop_last=bool(loader_spec.get("drop_last", True)),
            **self._worker_options(loader_spec, workers),
            **loader_options,
        )
        return self._train_loader

    def _evaluation_loader(self, split: str) -> DataLoader:
        self.setup(split)
        dataset = self.val_dataset if split == "val" else self.test_dataset
        assert dataset is not None
        loader_spec = dict(self.spec.loader)
        workers = int(loader_spec["eval_workers"])
        batch_key = "val_batch_size" if split == "val" else "test_batch_size"
        loader_options: dict[str, Any] = {}
        loader_streams = dict(loader_spec.get("loader_generator_streams", {}))
        if split in loader_streams:
            loader_options["generator"] = make_generator(
                int(self.spec.runtime["seed"]),
                stream=int(loader_streams[split]),
            )
        if loader_spec.get("collate") == "aquav3":
            loader_options["collate_fn"] = aquav3_collate
        return DataLoader(
            dataset=dataset,
            batch_size=int(loader_spec[batch_key]),
            shuffle=False,
            num_workers=workers,
            pin_memory=bool(loader_spec.get("pin_memory", True)),
            drop_last=False,
            **self._worker_options(loader_spec, workers),
            **loader_options,
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_loader is None:
            self._val_loader = self._evaluation_loader("val")
        return self._val_loader

    def test_dataloader(self) -> DataLoader:
        if self._test_loader is None:
            self._test_loader = self._evaluation_loader("test")
        return self._test_loader
