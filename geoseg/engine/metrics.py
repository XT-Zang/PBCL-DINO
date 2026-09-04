from __future__ import annotations

import torch

from geoseg.metrics.binary import binary_confusion_tensor, binary_global_stats


class BinaryMetricAccumulator:
    """Accumulate binary confusion counts on-device and synchronize once per epoch."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = float(threshold)
        self.counts = torch.zeros(4, dtype=torch.int64)

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        probabilities: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        batch_counts = binary_confusion_tensor(
            logits,
            targets,
            threshold=self.threshold,
            probabilities=probabilities,
            valid_mask=valid_mask,
        )
        if self.counts.device != batch_counts.device:
            self.counts = self.counts.to(batch_counts.device)
        self.counts.add_(batch_counts)

    def compute(self) -> dict[str, float]:
        tp, fp, fn, tn = self.counts.detach().cpu().tolist()
        return binary_global_stats({"tp": tp, "fp": fp, "fn": fn, "tn": tn})

    def reset(self) -> None:
        self.counts.zero_()
