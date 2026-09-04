from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytorch_lightning as pl
import torch

from geoseg.losses.binary_seg import loss_batch_kwargs
from geoseg.metrics.aquadataset import AQUADATASET_OVERALL_SCORE_WEIGHTS, aquadataset_overall_score
from geoseg.metrics.binary import resize_logits_to_targets
from geoseg.metrics.deadhesion import CachedDeadhesionEvaluator, json_ready as deadhesion_json_ready
from geoseg.models.aquav3_outputs import get_main_logits
from .artifacts import ValidationArtifactWriter, build_binary_validation_cpu_buffer
from .metrics import BinaryMetricAccumulator


class SegmentationTask(pl.LightningModule):
    """PBCL-DINO binary segmentation training task."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.net = config.net
        self.loss = config.loss
        self.metric_threshold = float(config.metric_threshold)
        self.validation_artifacts = ValidationArtifactWriter(config)
        self.metrics_train_binary = BinaryMetricAccumulator(self.metric_threshold)
        self.metrics_val_binary = BinaryMetricAccumulator(self.metric_threshold)
        self.deadhesion_val_evaluator = CachedDeadhesionEvaluator(
            config.data_root,
            split="val",
            instances_coco=None,
            boundary_width=int(config.deadhesion_boundary_width),
            boundary_tolerance=int(config.deadhesion_boundary_tolerance),
            min_overlap_pixels=int(config.deadhesion_min_overlap_pixels),
            min_overlap_ratio=float(config.deadhesion_min_overlap_ratio),
        )

    def forward(self, inputs: torch.Tensor):
        return self.net(inputs)

    def transfer_batch_to_device(
        self,
        batch: Any,
        device: torch.device,
        dataloader_idx: int,
    ) -> Any:
        if not isinstance(batch, Mapping):
            return super().transfer_batch_to_device(batch, device, dataloader_idx)
        canonical = dict(batch)
        had_image_alias = "image" in canonical and "img" in canonical
        had_mask_alias = "mask" in canonical and "gt_semantic_seg" in canonical
        if had_image_alias:
            canonical.pop("image")
        if had_mask_alias:
            canonical.pop("mask")
        moved = super().transfer_batch_to_device(canonical, device, dataloader_idx)
        if bool(getattr(self.config, "channels_last", False)):
            moved["img"] = moved["img"].contiguous(memory_format=torch.channels_last)
        if had_image_alias:
            moved["image"] = moved["img"]
        if had_mask_alias:
            moved["mask"] = moved["gt_semantic_seg"]
        return moved

    @staticmethod
    def _batch_tile_ids(batch: Mapping[str, Any], batch_size: int) -> list[str]:
        tile_ids = batch.get("tile_id", batch.get("img_id"))
        if isinstance(tile_ids, (list, tuple)) and len(tile_ids) == batch_size:
            return [str(tile_id) for tile_id in tile_ids]
        paths = batch.get("path")
        if isinstance(paths, (list, tuple)) and len(paths) == batch_size:
            return [Path(str(path)).stem for path in paths]
        raise ValueError("validation requires batch tile_id, img_id, or path metadata")

    def _write_deadhesion_val_summary(self, summary: dict[str, Any]) -> None:
        detail_dir = Path(self.config.weights_path) / "deadhesion_val"
        detail_dir.mkdir(parents=True, exist_ok=True)
        epoch = int(self.current_epoch) + 1
        (detail_dir / f"epoch_{epoch:03d}.json").write_text(
            json.dumps(deadhesion_json_ready(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def _loss_kwargs(self, batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
        return loss_batch_kwargs(batch, device)

    def _binary_epoch_end(self, split: str) -> None:
        accumulator = self.metrics_train_binary if split == "train" else self.metrics_val_binary
        stats = accumulator.compute()
        values = {
            "fg_iou": stats["fg_iou"],
            "bg_iou": stats["bg_iou"],
            "miou": stats["miou"],
            "fg_dice": stats["fg_dice"],
            "fg_precision": stats["fg_precision"],
            "fg_recall": stats["fg_recall"],
            "bg_fp_rate": stats["bg_fp_rate"],
        }
        if split == "val":
            deadhesion = self.deadhesion_val_evaluator.summary()
            values.update(self.deadhesion_val_evaluator.flat_metrics(deadhesion))
            self._write_deadhesion_val_summary(deadhesion)
            self.deadhesion_val_evaluator.reset()
            if dict(self.config.overall_score_weights) != AQUADATASET_OVERALL_SCORE_WEIGHTS:
                raise ValueError("overall_score_weights must match the locked Aquadataset formula")
            values["overall_score"] = aquadataset_overall_score(values)
        print(f"{split}:", values)
        self.log_dict({f"{split}_{key}": value for key, value in values.items()}, prog_bar=True)
        accumulator.reset()

    def training_step(self, batch: Mapping[str, Any], batch_idx: int):
        del batch_idx
        image, mask = batch["img"], batch["gt_semantic_seg"]
        loss_kwargs = self._loss_kwargs(batch, image.device)
        prediction = self.net(image)
        loss = self.loss(prediction, mask, **loss_kwargs)
        logits = resize_logits_to_targets(get_main_logits(prediction).detach(), mask.detach())
        self.metrics_train_binary.update(
            logits,
            mask.detach(),
            probabilities=torch.sigmoid(logits),
            valid_mask=batch.get("valid_mask"),
        )
        return {"loss": loss}

    def on_train_epoch_end(self) -> None:
        self._binary_epoch_end("train")

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int):
        del batch_idx
        image, mask = batch["img"], batch["gt_semantic_seg"]
        loss_kwargs = self._loss_kwargs(batch, image.device)
        prediction = self.net(image)
        logits = resize_logits_to_targets(get_main_logits(prediction).detach(), mask.detach())
        probabilities = torch.sigmoid(logits)
        self.metrics_val_binary.update(
            logits,
            mask.detach(),
            probabilities=probabilities,
            valid_mask=batch.get("valid_mask"),
        )
        cpu_buffer = build_binary_validation_cpu_buffer(
            probabilities,
            mask,
            threshold=self.metric_threshold,
        )
        tile_ids = self._batch_tile_ids(batch, batch_size=int(logits.shape[0]))
        self.deadhesion_val_evaluator.update_batch(
            cpu_buffer.predictions.numpy(),
            cpu_buffer.targets.numpy(),
            tile_ids,
        )
        if self.validation_artifacts.enabled and not self.trainer.sanity_checking:
            self.validation_artifacts.save_binary_batch(
                batch,
                probabilities=probabilities,
                mask=mask,
                epoch=int(self.current_epoch) + 1,
                global_step=int(self.global_step),
                cpu_buffer=cpu_buffer,
            )
        return {"loss_val": self.loss(prediction, mask, **loss_kwargs)}

    def on_validation_epoch_end(self) -> None:
        self.validation_artifacts.flush()
        self._binary_epoch_end("val")

    def on_fit_end(self) -> None:
        self.validation_artifacts.close()

    def configure_optimizers(self):
        return [self.config.optimizer], [self.config.lr_scheduler]

    def train_dataloader(self):
        return self.config.train_loader

    def val_dataloader(self):
        return self.config.val_loader
