from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

_POSITION_TOKEN_MARKERS = (
    "pos_embed",
    "position",
    "relative_position",
    "absolute_pos",
    "cls_token",
    "mask_token",
    "storage_token",
)


@dataclass(frozen=True)
class UniformAdamWBuild:
    optimizer: torch.optim.AdamW
    parameter_names: Mapping[str, tuple[str, ...]]


def _uses_weight_decay(name: str, parameter: torch.nn.Parameter) -> bool:
    if parameter.ndim <= 1 or name.endswith(".bias"):
        return False
    normalized = name.lower()
    return not any(marker in normalized for marker in _POSITION_TOKEN_MARKERS)


def build_uniform_adamw(
    model: torch.nn.Module,
    optimizer_spec: Mapping[str, object],
) -> UniformAdamWBuild:
    if str(optimizer_spec.get("name", "")) != "AdamW":
        raise ValueError("PBCL-DINO optimization requires AdamW")
    base_lr = float(optimizer_spec["base_lr"])
    encoder_lr = float(optimizer_spec["encoder_lr"])
    weight_decay = float(optimizer_spec["weight_decay"])
    encoder_prefixes = tuple(str(value) for value in optimizer_spec.get("encoder_prefixes", ()))
    buckets: dict[str, list[tuple[str, torch.nn.Parameter]]] = {
        "encoder_decay": [],
        "encoder_no_decay": [],
        "head_decay": [],
        "head_no_decay": [],
    }
    prefix_matches = {prefix: 0 for prefix in encoder_prefixes}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        matching_prefix = next((prefix for prefix in encoder_prefixes if name.startswith(prefix)), None)
        if matching_prefix is not None:
            prefix_matches[matching_prefix] += 1
        role = "encoder" if matching_prefix is not None else "head"
        decay = "decay" if _uses_weight_decay(name, parameter) else "no_decay"
        buckets[f"{role}_{decay}"].append((name, parameter))
    missing = [prefix for prefix, count in prefix_matches.items() if count == 0]
    if missing:
        raise ValueError(f"Declared encoder prefixes matched no trainable parameters: {missing}")
    parameter_groups: list[dict[str, object]] = []
    parameter_names: dict[str, tuple[str, ...]] = {}
    for group_name in ("encoder_decay", "encoder_no_decay", "head_decay", "head_no_decay"):
        entries = buckets[group_name]
        if not entries:
            continue
        is_encoder = group_name.startswith("encoder_")
        uses_decay = group_name.endswith("_decay") and not group_name.endswith("_no_decay")
        parameter_groups.append(
            {
                "name": group_name,
                "params": [parameter for _, parameter in entries],
                "lr": encoder_lr if is_encoder else base_lr,
                "weight_decay": weight_decay if uses_decay else 0.0,
            }
        )
        parameter_names[group_name] = tuple(name for name, _ in entries)
    if not parameter_groups:
        raise ValueError("Model has no trainable parameters")
    optimizer = torch.optim.AdamW(parameter_groups, lr=base_lr, weight_decay=weight_decay)
    return UniformAdamWBuild(optimizer=optimizer, parameter_names=parameter_names)
