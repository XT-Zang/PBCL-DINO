from __future__ import annotations

from .stage_adapter_fpn_ocr import (
    StageAdapterFPNOCRSpec,
    StageAdapterFPNSSepNet,
    StageAdapterStandardFPNBinaryHead,
    build_ssepnet_stage_adapter_fpn_ocr,
    build_ssepnet_stage_adapter_fpn_ocr_loss,
    get_ssepnet_stage_adapter_fpn_ocr,
)

TARGET_VARIANT = "ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6_binary_head"
SSEPNET_VARIANTS = {TARGET_VARIANT: get_ssepnet_stage_adapter_fpn_ocr(TARGET_VARIANT)}


def _require_target(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    if key != TARGET_VARIANT:
        raise ValueError(
            f"This minimal release contains only {TARGET_VARIANT!r}; got {name!r}"
        )
    return key


def build_ssepnet_variant(
    name: str,
    *,
    in_chans: int = 4,
    num_classes: int = 1,
    pretrained: bool = True,
    freeze_backbones: bool = False,
    activation_checkpoint: bool = True,
    dinov3_adaptation: str = "full",
    dinov3_lora_rank: int = 0,
    dinov3_lora_alpha: float | None = None,
    physical_pruning: bool = False,
    **_: object,
):
    key = _require_target(name)
    if physical_pruning:
        raise ValueError("physical_pruning is not part of the published target recipe")
    return build_ssepnet_stage_adapter_fpn_ocr(
        key,
        in_chans=in_chans,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbones,
        activation_checkpoint=activation_checkpoint,
        dinov3_adaptation=dinov3_adaptation,
        dinov3_lora_rank=dinov3_lora_rank,
        dinov3_lora_alpha=dinov3_lora_alpha,
    )


def build_ssepnet_loss(name: str):
    return build_ssepnet_stage_adapter_fpn_ocr_loss(_require_target(name))


def get_ssepnet_variant(name: str) -> StageAdapterFPNOCRSpec:
    return get_ssepnet_stage_adapter_fpn_ocr(_require_target(name))


__all__ = [
    "TARGET_VARIANT",
    "SSEPNET_VARIANTS",
    "StageAdapterFPNOCRSpec",
    "StageAdapterFPNSSepNet",
    "StageAdapterStandardFPNBinaryHead",
    "build_ssepnet_variant",
    "build_ssepnet_loss",
    "get_ssepnet_variant",
]
