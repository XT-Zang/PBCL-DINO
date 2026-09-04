from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


PAIR_BUCKET_CODE = {"touching": 0, "1_2": 1, "3_5": 2}
PAIR_BUCKET_NAME = {value: key for key, value in PAIR_BUCKET_CODE.items()}


def _four_neighbor_max(values: torch.Tensor) -> torch.Tensor:
    padded = F.pad(values, (1, 1, 1, 1), mode="constant", value=0.0)
    return torch.maximum(
        torch.maximum(padded[..., :-2, 1:-1], padded[..., 2:, 1:-1]),
        torch.maximum(padded[..., 1:-1, :-2], padded[..., 1:-1, 2:]),
    )


def _dilate_eight(mask: torch.Tensor, radius: int) -> torch.Tensor:
    values = mask.to(dtype=torch.float32)
    for _ in range(radius):
        values = F.max_pool2d(values, kernel_size=3, stride=1, padding=1)
    return values > 0.5


def _pair_bottleneck_score(
    probabilities: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    band_radius: int,
    iterations: int,
) -> torch.Tensor | None:
    interaction = _dilate_eight(first, band_radius) | _dilate_eight(second, band_radius)
    source = first & interaction
    target = second & interaction
    if not bool(source.any()) or not bool(target.any()):
        return None
    reach = torch.where(source, probabilities, torch.zeros_like(probabilities))
    for _ in range(iterations):
        propagated = torch.minimum(_four_neighbor_max(reach), probabilities)
        reach = torch.where(interaction, torch.maximum(reach, propagated), torch.zeros_like(reach))
    return reach[target].amax()


def _pair_crop_slices(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    padding: int,
) -> tuple[slice, slice] | None:
    locations = torch.nonzero((first | second)[0, 0], as_tuple=False)
    if not locations.numel():
        return None
    lower = locations.amin(dim=0)
    upper = locations.amax(dim=0)
    y_min, x_min, y_max, x_max = torch.cat((lower, upper)).tolist()
    height, width = first.shape[-2:]
    return (
        slice(max(0, y_min - padding), min(height, y_max + padding + 1)),
        slice(max(0, x_min - padding), min(width, x_max + padding + 1)),
    )


def pair_bottleneck_connectivity_loss(
    logits: torch.Tensor,
    instance_maps: torch.Tensor,
    instance_pairs: Sequence[torch.Tensor],
    *,
    band_radius: int = 3,
    iterations: int = 9,
    threshold: float = 0.5,
    margin: float = 0.1,
    neighbors: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return pair-equal PBCL, detached mean bottleneck score, and pair count."""
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(f"PBCL requires binary logits with shape (B,1,H,W), got {tuple(logits.shape)}")
    if instance_maps.ndim == 3:
        instance_maps = instance_maps.unsqueeze(1)
    if tuple(instance_maps.shape) != tuple(logits.shape):
        raise ValueError(
            f"instance_maps must match logits shape, got {tuple(instance_maps.shape)} and {tuple(logits.shape)}"
        )
    if len(instance_pairs) != logits.shape[0]:
        raise ValueError(
            f"instance_pairs length {len(instance_pairs)} does not match batch size {logits.shape[0]}"
        )
    if band_radius <= 0:
        raise ValueError(f"band_radius must be positive, got {band_radius}")
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")
    if neighbors != 4:
        raise ValueError(f"PBCL currently supports pair_neighbors=4 only, got {neighbors}")
    cutoff = float(threshold) - float(margin)
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError(f"threshold - margin must be in [0,1], got {cutoff}")

    with torch.autocast(device_type=logits.device.type, enabled=False):
        probabilities = torch.sigmoid(logits.float())
        labels = instance_maps.to(device=logits.device, dtype=torch.long)
        scores: list[torch.Tensor] = []
        for batch_idx, raw_pairs in enumerate(instance_pairs):
            pairs = torch.as_tensor(raw_pairs, device=logits.device, dtype=torch.long)
            if pairs.ndim != 2 or pairs.shape[1] != 3:
                raise ValueError(f"each instance_pairs tensor must have shape (N,3), got {tuple(pairs.shape)}")
            if pairs.numel() == 0:
                continue
            for first_id, second_id, bucket_code in pairs.tolist():
                if int(bucket_code) not in PAIR_BUCKET_NAME:
                    continue
                first = labels[batch_idx : batch_idx + 1] == int(first_id)
                second = labels[batch_idx : batch_idx + 1] == int(second_id)
                crop_slices = _pair_crop_slices(first, second, padding=band_radius)
                if crop_slices is None:
                    continue
                y_slice, x_slice = crop_slices
                score = _pair_bottleneck_score(
                    probabilities[batch_idx : batch_idx + 1, :, y_slice, x_slice],
                    first[:, :, y_slice, x_slice],
                    second[:, :, y_slice, x_slice],
                    band_radius=band_radius,
                    iterations=iterations,
                )
                if score is not None:
                    scores.append(score)

        if not scores:
            zero = logits.float().sum() * 0.0
            return zero, zero.detach(), 0
        score_tensor = torch.stack(scores)
        loss = torch.relu(score_tensor - cutoff).square().mean()
        return loss, score_tensor.mean().detach(), len(scores)
