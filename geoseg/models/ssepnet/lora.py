from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn


DINOV3_ADAPTATION_MODES = frozenset({"full", "frozen", "lora"})


@dataclass(frozen=True)
class DinoV3AdaptationReport:
    mode: str
    rank: int
    alpha: float | None
    target_blocks: tuple[int, ...]
    trainable_parameters: int


class QVLoRAQKV(nn.Module):
    """Add trainable low-rank updates to the query and value thirds of QKV."""

    def __init__(self, base_qkv: nn.Module, *, rank: int, alpha: float | None = None) -> None:
        super().__init__()
        if int(rank) <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if not hasattr(base_qkv, "in_features") or not hasattr(base_qkv, "out_features"):
            raise TypeError("DINOv3 QKV projection must expose in_features and out_features")
        dim = int(base_qkv.in_features)
        if int(base_qkv.out_features) != dim * 3:
            raise ValueError("DINOv3 QKV LoRA requires a projection with out_features == 3 * in_features")
        self.base_qkv = base_qkv
        self.rank = int(rank)
        self.alpha = float(self.rank if alpha is None else alpha)
        self.scaling = self.alpha / float(self.rank)
        self.in_features = dim
        self.out_features = dim * 3
        weight = next(base_qkv.parameters(), None)
        if weight is None:
            raise ValueError("DINOv3 QKV projection has no parameters")
        factory_kwargs = {"device": weight.device, "dtype": weight.dtype}
        self.lora_q_a = nn.Linear(dim, self.rank, bias=False, **factory_kwargs)
        self.lora_q_b = nn.Linear(self.rank, dim, bias=False, **factory_kwargs)
        self.lora_v_a = nn.Linear(dim, self.rank, bias=False, **factory_kwargs)
        self.lora_v_b = nn.Linear(self.rank, dim, bias=False, **factory_kwargs)
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_q_a.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_v_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_q_b.weight)
        nn.init.zeros_(self.lora_v_b.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base_qkv(inputs)
        query = self.lora_q_b(self.lora_q_a(inputs)) * self.scaling
        value = self.lora_v_b(self.lora_v_a(inputs)) * self.scaling
        key = torch.zeros_like(query)
        return base + torch.cat((query, key, value), dim=-1)

    def merge(self) -> nn.Module:
        q_delta = self.lora_q_b.weight @ self.lora_q_a.weight
        v_delta = self.lora_v_b.weight @ self.lora_v_a.weight
        with torch.no_grad():
            self.base_qkv.weight[: self.in_features].add_(q_delta.to(self.base_qkv.weight.dtype), alpha=self.scaling)
            self.base_qkv.weight[-self.in_features :].add_(v_delta.to(self.base_qkv.weight.dtype), alpha=self.scaling)
        return self.base_qkv


def _dinov3_blocks(backbone: nn.Module) -> nn.ModuleList:
    model = getattr(backbone, "model", backbone)
    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError("DINOv3 backbone must expose model.blocks as nn.ModuleList")
    return blocks


def configure_dinov3_adaptation(backbone: nn.Module, *, mode: str, rank: int = 3, alpha: float | None = None) -> DinoV3AdaptationReport:
    normalized = str(mode).strip().lower()
    if normalized not in DINOV3_ADAPTATION_MODES:
        raise ValueError(f"Unsupported DINOv3 adaptation mode {mode!r}; choose from {sorted(DINOV3_ADAPTATION_MODES)}")
    if normalized == "full":
        backbone.requires_grad_(True)
        return DinoV3AdaptationReport("full", 0, None, (), 0)
    if normalized == "frozen":
        backbone.requires_grad_(False)
        return DinoV3AdaptationReport("frozen", 0, None, (), 0)
    if int(rank) <= 0:
        raise ValueError(f"LoRA rank must be positive, got {rank}")
    backbone.requires_grad_(False)
    target_blocks: list[int] = []
    for index, block in enumerate(_dinov3_blocks(backbone)):
        attention = getattr(block, "attn", None)
        qkv = getattr(attention, "qkv", None)
        if qkv is None:
            raise TypeError(f"DINOv3 block {index} does not expose attn.qkv")
        if isinstance(qkv, QVLoRAQKV):
            raise ValueError(f"DINOv3 block {index} already contains Q/V LoRA")
        attention.qkv = QVLoRAQKV(qkv, rank=int(rank), alpha=alpha)
        target_blocks.append(index)
    trainable = sum(parameter.numel() for parameter in backbone.parameters() if parameter.requires_grad)
    effective_alpha = float(rank if alpha is None else alpha)
    return DinoV3AdaptationReport("lora", int(rank), effective_alpha, tuple(target_blocks), int(trainable))


def merge_dinov3_qv_lora(backbone: nn.Module) -> tuple[int, ...]:
    merged: list[int] = []
    for index, block in enumerate(_dinov3_blocks(backbone)):
        qkv = getattr(getattr(block, "attn", None), "qkv", None)
        if not isinstance(qkv, QVLoRAQKV):
            continue
        block.attn.qkv = qkv.merge()
        merged.append(index)
    return tuple(merged)


__all__ = ["DINOV3_ADAPTATION_MODES", "DinoV3AdaptationReport", "QVLoRAQKV", "configure_dinov3_adaptation", "merge_dinov3_qv_lora"]
