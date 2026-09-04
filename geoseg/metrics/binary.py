from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def resize_logits_to_targets(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] == targets.shape[-2:]:
        return logits
    return F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)


@torch.no_grad()
def binary_confusion_tensor(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    *,
    probabilities: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    logits = resize_logits_to_targets(logits, targets)
    probs = probabilities if probabilities is not None else torch.sigmoid(logits)
    if probs.shape != logits.shape:
        raise ValueError(f"probabilities shape {tuple(probs.shape)} does not match logits {tuple(logits.shape)}")
    preds = probs > threshold
    targets_b = targets > 0.5
    if valid_mask is None:
        valid = torch.ones_like(targets_b, dtype=torch.bool)
    else:
        valid = valid_mask.to(device=targets.device, dtype=torch.bool)
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)
        if valid.shape[-2:] != targets.shape[-2:]:
            valid = F.interpolate(valid.float(), size=targets.shape[-2:], mode="nearest").bool()
        if valid.shape[1] == 1 and targets_b.shape[1] != 1:
            valid = valid.expand(-1, targets_b.shape[1], -1, -1)
        if valid.shape != targets_b.shape:
            raise ValueError(f"valid mask shape {tuple(valid.shape)} does not match targets {tuple(targets_b.shape)}")
    return torch.stack(
        (
            (preds & targets_b & valid).sum(dtype=torch.int64),
            (preds & ~targets_b & valid).sum(dtype=torch.int64),
            (~preds & targets_b & valid).sum(dtype=torch.int64),
            (~preds & ~targets_b & valid).sum(dtype=torch.int64),
        )
    )


def binary_global_stats(counts: dict[str, float]) -> dict[str, float]:
    tp = float(counts.get("tp", 0.0))
    fp = float(counts.get("fp", 0.0))
    fn = float(counts.get("fn", 0.0))
    tn = float(counts.get("tn", 0.0))
    eps = 1e-6
    fg_iou = tp / (tp + fp + fn + eps)
    bg_iou = tn / (tn + fp + fn + eps)
    return {
        "fg_iou": fg_iou,
        "bg_iou": bg_iou,
        "miou": 0.5 * (fg_iou + bg_iou),
        "fg_dice": (2.0 * tp) / (2.0 * tp + fp + fn + eps),
        "fg_precision": tp / (tp + fp + eps),
        "fg_recall": tp / (tp + fn + eps),
        "bg_fp_rate": fp / (fp + tn + eps),
    }
