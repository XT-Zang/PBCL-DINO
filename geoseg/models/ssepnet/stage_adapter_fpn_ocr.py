from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import torch
import torch.nn as nn

from .multiscale import ConvGNReLU, PyramidBinaryHead, StandardPyramidDecoder, _resize_outputs
from .multiscale_dinov3 import StageAdapterDINOv3Backbone


TARGET_VARIANT = "ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6_binary_head"


@dataclass(frozen=True)
class StageAdapterFPNOCRSpec:
    name: str
    stage_depths: tuple[int, int, int, int] = (6, 6, 6, 6)
    stage_dims: tuple[int, int, int, int] = (128, 256, 512, 512)
    head_kind: str = "binary"
    boundary_enabled: bool = True
    pbcl_weight: float = 0.1
    boundary_source: str = "raw_p2"
    decoder_channels: int = 128
    ocr_channels: int = 256
    dropout: float = 0.05

    @property
    def output_layers(self) -> tuple[int, int, int, int]:
        total = 0
        values: list[int] = []
        for depth in self.stage_depths:
            total += int(depth)
            values.append(total - 1)
        return tuple(values)  # type: ignore[return-value]

    @property
    def active_backbones(self) -> tuple[str, ...]:
        return ("dinov3",)


_TARGET_SPEC = StageAdapterFPNOCRSpec(name=TARGET_VARIANT)
SSEPNET_STAGE_ADAPTER_FPN_OCR: Mapping[str, StageAdapterFPNOCRSpec] = MappingProxyType(
    {TARGET_VARIANT: _TARGET_SPEC}
)


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def is_ssepnet_stage_adapter_fpn_ocr(name: str) -> bool:
    return _normalize_name(name) == TARGET_VARIANT


def get_ssepnet_stage_adapter_fpn_ocr(name: str) -> StageAdapterFPNOCRSpec:
    key = _normalize_name(name)
    if key != TARGET_VARIANT:
        raise ValueError(f"PBCL-DINO publishes only {TARGET_VARIANT!r}; got {name!r}")
    return _TARGET_SPEC


class StageAdapterStandardFPNBinaryHead(nn.Module):
    """Standard FPN binary head with Boundary supervision from raw lateral P2."""

    def __init__(
        self,
        input_channels: Sequence[int],
        *,
        decoder_channels: int = 128,
        dropout: float = 0.05,
        boundary_enabled: bool = True,
        boundary_source: str = "raw_p2",
    ) -> None:
        super().__init__()
        if len(input_channels) != 4:
            raise ValueError("Stage-Adapter FPN/Binary head requires P2/P3/P4/P5")
        if boundary_source not in {"raw_p2", "decoded_p2"}:
            raise ValueError(f"unsupported Boundary source: {boundary_source}")
        decoder_channels = int(decoder_channels)
        self.output_strides = (4, 8, 16, 32)
        self.auxiliary_stride = 16
        self.boundary_stride = 4
        self.boundary_enabled = bool(boundary_enabled)
        self.boundary_source = str(boundary_source)
        self.lateral_projections = nn.ModuleList(
            ConvGNReLU(int(channels), decoder_channels, 1) for channels in input_channels
        )
        self.decoder = StandardPyramidDecoder(decoder_channels, dropout=float(dropout))
        self.binary_head = PyramidBinaryHead(decoder_channels, float(dropout))
        if not self.boundary_enabled:
            self.binary_head.boundary = nn.Identity()

    def _project_and_decode(
        self, features: Sequence[torch.Tensor]
    ) -> tuple[
        list[torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        if len(features) != 4:
            raise ValueError(f"Stage-Adapter FPN/Binary requires four features, got {len(features)}")
        projected = [
            projection(feature)
            for projection, feature in zip(self.lateral_projections, features, strict=True)
        ]
        return projected, self.decoder(projected)

    def forward_main(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        _projected, pyramid = self._project_and_decode(features)
        return self.binary_head.main(pyramid[4])

    def forward(self, features: Sequence[torch.Tensor]) -> dict[str, torch.Tensor]:
        projected, pyramid = self._project_and_decode(features)
        outputs = {
            "out": self.binary_head.main(pyramid[4]),
            "aux": self.binary_head.auxiliary(pyramid[2]),
        }
        if self.boundary_enabled:
            boundary_features = projected[0] if self.boundary_source == "raw_p2" else pyramid[0]
            outputs["edge"] = self.binary_head.boundary(boundary_features)
        return outputs


class StageAdapterFPNSSepNet(nn.Module):
    """DINOv3 6/6/6/6 Stage-Adapter with Standard FPN binary head."""

    active_backbones = ("dinov3",)

    def __init__(
        self,
        *,
        multiscale_backbone: nn.Module,
        decoder_channels: int = 128,
        boundary_enabled: bool = True,
        boundary_source: str = "raw_p2",
        num_classes: int = 1,
        freeze_backbone: bool = False,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if int(num_classes) != 1:
            raise ValueError("PBCL-DINO requires binary segmentation")
        channels = tuple(int(value) for value in multiscale_backbone.out_channels)
        if len(channels) != 4:
            raise ValueError("Stage-Adapter backbone must expose four feature widths")
        self.multiscale_backbone = multiscale_backbone
        self.freeze_backbone = bool(freeze_backbone)
        self.head = StageAdapterStandardFPNBinaryHead(
            channels,
            decoder_channels=int(decoder_channels),
            dropout=float(dropout),
            boundary_enabled=bool(boundary_enabled),
            boundary_source=boundary_source,
        )
        self.head_kind = "binary"
        self.boundary_enabled = bool(boundary_enabled)
        self.boundary_source = str(boundary_source)
        if self.freeze_backbone:
            self.multiscale_backbone.requires_grad_(False)

    @property
    def pretrained_initialization_report(self) -> Mapping[str, object]:
        return dict(getattr(self.multiscale_backbone, "initialization_report", {}))

    def interaction_gate_statistics(self) -> tuple[tuple[float, float, float, float], ...]:
        callback = getattr(self.multiscale_backbone, "interaction_gate_statistics", None)
        return () if callback is None else callback()

    def train(self, mode: bool = True) -> "StageAdapterFPNSSepNet":
        super().train(mode)
        if self.freeze_backbone:
            self.multiscale_backbone.eval()
        return self

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        main_only: bool = False,
        disable_hrnet: bool = False,
        disable_dinov3: bool = False,
    ) -> dict[str, torch.Tensor]:
        del disable_hrnet
        features = list(self.multiscale_backbone(inputs))
        if len(features) != 4:
            raise ValueError(f"Stage-Adapter must return four features, got {len(features)}")
        if disable_dinov3:
            features = [torch.zeros_like(feature) for feature in features]
        outputs = {"out": self.head.forward_main(features)} if main_only else self.head(features)
        return _resize_outputs(outputs, tuple(inputs.shape[-2:]))

    def forward_inference(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward(inputs, main_only=True)["out"]


def build_ssepnet_stage_adapter_fpn_ocr(
    name: str,
    *,
    in_chans: int = 4,
    num_classes: int = 1,
    pretrained: bool = True,
    dinov3_checkpoint: str | Path | None = None,
    freeze_backbone: bool = False,
    activation_checkpoint: bool = True,
    dinov3_adaptation: str = "full",
    dinov3_lora_rank: int = 0,
    dinov3_lora_alpha: float | None = None,
) -> StageAdapterFPNSSepNet:
    spec = get_ssepnet_stage_adapter_fpn_ocr(name)
    if int(in_chans) not in {3, 4}:
        raise ValueError("PBCL-DINO requires RGB or B/G/R/NIR input")
    from .dinov3 import build_dinov3_vitl16_sat493m
    from .lora import configure_dinov3_adaptation

    flat = build_dinov3_vitl16_sat493m(
        in_chans=int(in_chans),
        pretrained=bool(pretrained),
        checkpoint_path=dinov3_checkpoint,
    )
    adaptation = configure_dinov3_adaptation(
        flat,
        mode=dinov3_adaptation,
        rank=int(dinov3_lora_rank),
        alpha=dinov3_lora_alpha,
    )
    backbone = StageAdapterDINOv3Backbone(
        dinov3_backbone=flat,
        stage_depths=spec.stage_depths,
        spatial_dims=spec.stage_dims,
        in_chans=int(in_chans),
        activation_checkpoint=bool(activation_checkpoint),
    )
    backbone.initialization_report.update(
        {
            "architecture": "stage_adapter_standard_fpn_binary",
            "stage_depths": spec.stage_depths,
            "stage_output_layers": spec.output_layers,
            "output_strides": (4, 8, 16, 32),
            "segmentation_head": "standard_fpn_binary_boundary_v1",
            "boundary_enabled": spec.boundary_enabled,
            "boundary_source": spec.boundary_source,
            "pbcl_weight": spec.pbcl_weight,
            "soft_layer_mixing": False,
            "dinov3_adaptation": getattr(adaptation, "mode", dinov3_adaptation),
        }
    )
    return StageAdapterFPNSSepNet(
        multiscale_backbone=backbone,
        decoder_channels=spec.decoder_channels,
        boundary_enabled=spec.boundary_enabled,
        boundary_source=spec.boundary_source,
        num_classes=int(num_classes),
        freeze_backbone=bool(freeze_backbone),
        dropout=spec.dropout,
    )


def build_ssepnet_stage_adapter_fpn_ocr_loss(name: str) -> nn.Module:
    spec = get_ssepnet_stage_adapter_fpn_ocr(name)
    from geoseg.losses.ssepnet import SSepNetLoss

    return SSepNetLoss(
        boundary_weight=0.6 if spec.boundary_enabled else 0.0,
        pbcl_weight=float(spec.pbcl_weight),
    )


__all__ = [
    "SSEPNET_STAGE_ADAPTER_FPN_OCR",
    "StageAdapterStandardFPNBinaryHead",
    "StageAdapterFPNSSepNet",
    "StageAdapterFPNOCRSpec",
    "build_ssepnet_stage_adapter_fpn_ocr",
    "build_ssepnet_stage_adapter_fpn_ocr_loss",
    "get_ssepnet_stage_adapter_fpn_ocr",
    "is_ssepnet_stage_adapter_fpn_ocr",
]
