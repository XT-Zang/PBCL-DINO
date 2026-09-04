from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinarySegmentationLoss(nn.Module):
    """Minimal BCE+Dice base used by the published SSepNet objective."""

    def __init__(self, main_bce: float = 1.0, main_dice: float = 1.0, aux_bce: float = 0.3, aux_dice: float = 0.3):
        super().__init__()
        self.main_bce_weight = float(main_bce)
        self.main_dice_weight = float(main_dice)
        self.aux_bce_weight = float(aux_bce)
        self.aux_dice_weight = float(aux_dice)
        self.requires_instance_supervision = False

    @staticmethod
    def _valid_mask(valid_masks: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = valid_masks.to(device=target.device, dtype=target.dtype)
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)
        if valid.shape[-2:] != target.shape[-2:]:
            valid = F.interpolate(valid, size=target.shape[-2:], mode="nearest")
        if valid.shape[1] == 1 and target.shape[1] != 1:
            valid = valid.expand(-1, target.shape[1], -1, -1)
        if valid.shape != target.shape:
            raise ValueError(f"valid mask shape {tuple(valid.shape)} does not match target {tuple(target.shape)}")
        return valid


def loss_batch_kwargs(batch: Mapping, device: torch.device) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    valid_mask = batch.get("valid_mask")
    if valid_mask is not None:
        kwargs["valid_masks"] = torch.as_tensor(valid_mask).to(device=device, non_blocking=True)
    instance_map = batch.get("instance_map")
    instance_pairs = batch.get("instance_pairs")
    if instance_map is None and instance_pairs is None:
        return kwargs
    if instance_map is None or instance_pairs is None:
        raise KeyError("instance_map and instance_pairs must be present together")
    kwargs.update(
        instance_maps=instance_map.to(device=device, non_blocking=True),
        instance_pairs=[torch.as_tensor(pair).to(device=device, non_blocking=True) for pair in instance_pairs],
    )
    return kwargs


def _soft_boundary(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp(0.0, 1.0)
