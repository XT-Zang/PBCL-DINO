from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import math
from typing import Any

import torch

from geoseg.datasets.aquav3_datamodule import AquaV3DataModule
from geoseg.experiments.reproducibility import seed_experiment
from geoseg.losses.ssepnet import SSepNetLoss
from geoseg.models.ssepnet import build_ssepnet_variant

from .experiment import ExperimentComponents
from .optimization import build_uniform_adamw
from .specs import ExperimentSpec


class LegacyConfigView:
    _LAZY_DATA = {
        "train_loader": "train_dataloader",
        "val_loader": "val_dataloader",
        "test_loader": "test_dataloader",
        "train_dataset": "train_dataset",
        "val_dataset": "val_dataset",
        "test_dataset": "test_dataset",
    }

    def __init__(self, values: dict[str, Any], datamodule: AquaV3DataModule) -> None:
        object.__setattr__(self, "_values", values)
        object.__setattr__(self, "_datamodule", datamodule)

    def __getattr__(self, name: str) -> Any:
        if name in self._values:
            return self._values[name]
        if name in {"train_loader", "val_loader", "test_loader"}:
            return getattr(self._datamodule, self._LAZY_DATA[name])()
        if name in {"train_dataset", "val_dataset", "test_dataset"}:
            stage = {"train_dataset": "fit", "val_dataset": "validate", "test_dataset": "test"}[name]
            self._datamodule.setup(stage)
            return getattr(self._datamodule, name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self._values[name] = value


def _warmup_cosine_lambda(epoch: int, max_epoch: int, warmup_epochs: int, min_lr_ratio: float) -> float:
    current = epoch + 1
    if warmup_epochs > 0 and current <= warmup_epochs:
        return current / warmup_epochs
    progress = (current - warmup_epochs) / max(1, max_epoch - warmup_epochs)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def build_aquav3_components(spec: ExperimentSpec) -> tuple[ExperimentSpec, ExperimentComponents]:
    if spec.provenance.get("family") != "PBCL-DINO":
        raise ValueError("This public runtime supports PBCL-DINO only")
    seed_experiment(int(spec.runtime["seed"]))

    model = build_ssepnet_variant(
        str(spec.model["key"]),
        in_chans=int(spec.model["in_channels"]),
        num_classes=int(spec.model["num_classes"]),
        pretrained=bool(spec.model["pretrained"]),
        freeze_backbones=bool(spec.model["freeze_backbones"]),
        activation_checkpoint=bool(spec.model["activation_checkpoint"]),
        dinov3_adaptation=str(spec.model["dinov3_adaptation"]),
        dinov3_lora_rank=int(spec.model["dinov3_lora_rank"]),
        dinov3_lora_alpha=spec.model["dinov3_lora_alpha"],
    )
    loss_spec = dict(spec.optimization["loss"])
    loss = SSepNetLoss(
        boundary_weight=float(loss_spec["boundary_weight"]),
        boundary_kernel=int(loss_spec["boundary_kernel"]),
        pbcl_weight=float(loss_spec["pbcl_weight"]),
        pbcl_band_radius=int(loss_spec["pbcl_band_radius"]),
        pbcl_iterations=int(loss_spec["pbcl_iterations"]),
        pbcl_threshold=float(loss_spec["pbcl_threshold"]),
        pbcl_margin=float(loss_spec["pbcl_margin"]),
        pbcl_neighbors=int(loss_spec["pbcl_neighbors"]),
    )

    optimizer_build = build_uniform_adamw(model, spec.optimization["optimizer"])
    optimizer = optimizer_build.optimizer
    scheduler_spec = dict(spec.optimization["scheduler"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: _warmup_cosine_lambda(
            epoch,
            max_epoch=int(scheduler_spec.get("schedule_epochs", spec.runtime["max_epochs"])),
            warmup_epochs=int(scheduler_spec["warmup_epochs"]),
            min_lr_ratio=float(scheduler_spec["min_lr_ratio"]),
        ),
    )
    datamodule = AquaV3DataModule(spec)

    training_protocol = {
        "schema_version": 1,
        "campaign": "PBCL-DINO",
        "experiment_id": str(spec.provenance["experiment_id"]),
        "seed": int(spec.runtime["seed"]),
        "epochs": int(spec.runtime["max_epochs"]),
        "data": dict(spec.data),
        "model": dict(spec.model),
        "loss": loss_spec,
        "optimizer": dict(spec.optimization["optimizer"]),
        "scheduler": scheduler_spec,
        "checkpoint_policy": dict(spec.checkpoint),
        "optimizer_parameter_names": dict(optimizer_build.parameter_names),
        "test_threshold": 0.5,
    }
    values: dict[str, Any] = {
        "max_epoch": int(spec.runtime["max_epochs"]),
        "train_batch_size": int(spec.loader["train_batch_size"]),
        "val_batch_size": int(spec.loader["val_batch_size"]),
        "test_batch_size": int(spec.loader["test_batch_size"]),
        "accumulate_grad_batches": int(spec.optimization["accumulate_grad_batches"]),
        "lr": float(spec.optimization["optimizer"]["base_lr"]),
        "weight_decay": float(spec.optimization["optimizer"]["weight_decay"]),
        "warmup_epochs": int(scheduler_spec["warmup_epochs"]),
        "num_classes": 1,
        "classes": ("background", "foreground"),
        "weights_name": str(spec.artifacts["run_name"]),
        "weights_path": str(spec.artifacts["weights_path"]),
        "log_name": str(spec.artifacts["log_name"]),
        "monitor": str(spec.checkpoint["monitor"]),
        "monitor_mode": str(spec.checkpoint["mode"]),
        "save_top_k": int(spec.checkpoint["save_top_k"]),
        "save_last": False,
        "check_val_every_n_epoch": int(spec.runtime["check_val_every_n_epoch"]),
        "checkpoint_every_n_epochs": int(spec.checkpoint["every_n_epochs"]),
        "checkpoint_monitors": dict(spec.checkpoint["monitors"]),
        "checkpoint_primary_role": str(spec.checkpoint["primary_role"]),
        "checkpoint_save_weights_only": bool(spec.checkpoint["save_weights_only"]),
        "overall_score_weights": dict(spec.checkpoint["overall_score"]["weights"]),
        "pretrained_ckpt_path": None,
        "resume_ckpt_path": None,
        "gpus": spec.runtime["devices"],
        "precision": str(spec.runtime["precision"]),
        "gradient_clip_val": float(spec.optimization["gradient_clip_val"]),
        "num_sanity_val_steps": int(spec.runtime["num_sanity_val_steps"]),
        "binary_segmentation": True,
        "instance_segmentation": False,
        "metric_threshold": float(spec.metrics["threshold"]),
        "prediction_kind": "connected_components",
        "save_val_predictions": bool(spec.artifacts["save_val_predictions"]),
        "val_prediction_dir": str(spec.artifacts["val_prediction_dir"]),
        "async_writes": bool(spec.artifacts["async_writes"]),
        "deadhesion_validation": True,
        "deadhesion_split": "val",
        "deadhesion_instances_coco": None,
        "deadhesion_boundary_width": int(spec.metrics["boundary_width"]),
        "deadhesion_boundary_tolerance": int(spec.metrics["boundary_tolerance"]),
        "deadhesion_min_overlap_pixels": int(spec.metrics["min_overlap_pixels"]),
        "deadhesion_min_overlap_ratio": float(spec.metrics["min_overlap_ratio"]),
        "data_root": str(spec.data["root"]),
        "seed": int(spec.runtime["seed"]),
        "model_name": str(spec.model["key"]),
        "net": model,
        "loss": loss,
        "use_aux_loss": False,
        "optimizer": optimizer,
        "lr_scheduler": scheduler,
        "channels_last": bool(spec.runtime.get("channels_last", False)),
        "training_protocol": training_protocol,
        "optimizer_parameter_names": dict(optimizer_build.parameter_names),
    }
    config = LegacyConfigView(values, datamodule)
    resolved_spec = replace(
        spec,
        model={
            **spec.model,
            "class_name": type(model).__name__,
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        },
    )
    return resolved_spec, ExperimentComponents(
        config=config,
        model=model,
        loss=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        datamodule=datamodule,
    )
