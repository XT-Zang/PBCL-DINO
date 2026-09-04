from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from .binary_seg import BinarySegmentationLoss, _soft_boundary
from .pairwise_connectivity import pair_bottleneck_connectivity_loss


class SSepNetLoss(BinarySegmentationLoss):
    """Paper SSepNet objective for main, OCR auxiliary, Boundary, and PBCL outputs."""

    def __init__(
        self,
        *,
        boundary_weight: float,
        pbcl_weight: float,
        boundary_kernel: int = 5,
        pbcl_band_radius: int = 3,
        pbcl_iterations: int = 9,
        pbcl_threshold: float = 0.5,
        pbcl_margin: float = 0.1,
        pbcl_neighbors: int = 4,
    ) -> None:
        super().__init__(main_bce=1.0, main_dice=1.0, aux_bce=0.3, aux_dice=0.3)
        self.boundary_weight = float(boundary_weight)
        self.pbcl_weight = float(pbcl_weight)
        self.boundary_kernel = int(boundary_kernel)
        self.pbcl_band_radius = int(pbcl_band_radius)
        self.pbcl_iterations = int(pbcl_iterations)
        self.pbcl_threshold = float(pbcl_threshold)
        self.pbcl_margin = float(pbcl_margin)
        self.pbcl_neighbors = int(pbcl_neighbors)
        self.requires_instance_supervision = self.pbcl_weight > 0.0
        if self.boundary_weight < 0.0 or self.pbcl_weight < 0.0:
            raise ValueError("boundary_weight and pbcl_weight must be non-negative")
        if self.boundary_kernel < 3 or self.boundary_kernel % 2 == 0:
            raise ValueError("boundary_kernel must be an odd integer >= 3")
        self.last_components: dict[str, float | int] = {}

    @staticmethod
    def _paper_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits).flatten(1)
        flattened_target = target.flatten(1)
        intersection = (probabilities * flattened_target).sum(dim=1)
        denominator = probabilities.sum(dim=1) + flattened_target.sum(dim=1)
        dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        return 1.0 - dice.mean()

    def _single_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        bce_weight: float,
        dice_weight: float,
        valid_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target = target.to(dtype=logits.dtype)
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
        if valid_masks is not None:
            valid = self._valid_mask(valid_masks, target)
            denominator = valid.sum().clamp_min(1.0)
            bce = (
                F.binary_cross_entropy_with_logits(logits, target, reduction="none")
                * valid
            ).sum() / denominator
            probabilities = torch.sigmoid(logits) * valid
            masked_target = target * valid
            intersection = (probabilities * masked_target).flatten(1).sum(dim=1)
            dice_denominator = (
                probabilities.flatten(1).sum(dim=1)
                + masked_target.flatten(1).sum(dim=1)
            )
            dice = 1.0 - (
                (2.0 * intersection + 1e-6) / (dice_denominator + 1e-6)
            ).mean()
            return bce_weight * bce + dice_weight * dice
        return bce_weight * F.binary_cross_entropy_with_logits(
            logits,
            target,
        ) + dice_weight * self._paper_dice_loss(logits, target)

    @staticmethod
    def _require_outputs(outputs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        missing = [name for name in ("out", "aux") if name not in outputs]
        if missing:
            raise KeyError(f"SSepNet loss requires output tensors {missing}")
        return outputs["out"], outputs["aux"]

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        target: torch.Tensor,
        *,
        instance_maps: torch.Tensor | None = None,
        instance_pairs: Sequence[torch.Tensor] | None = None,
        valid_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not isinstance(outputs, Mapping):
            raise TypeError("SSepNet loss requires a mapping of named model outputs")
        main, aux = self._require_outputs(outputs)
        main_loss = self._single_loss(main, target, 1.0, 1.0, valid_masks)
        aux_loss = self._single_loss(aux, target, 0.3, 0.3, valid_masks)
        total = main_loss + aux_loss

        anchor = main.float().sum() * 0.0
        boundary_loss = anchor
        if self.boundary_weight > 0.0:
            if "edge" not in outputs:
                raise KeyError("SSepNet Boundary loss requires an 'edge' output tensor")
            boundary_target = _soft_boundary(target.float(), self.boundary_kernel)
            boundary_valid = valid_masks
            if valid_masks is not None:
                valid = self._valid_mask(valid_masks, target.float())
                padding = self.boundary_kernel // 2
                boundary_valid = 1.0 - F.max_pool2d(
                    1.0 - valid,
                    kernel_size=self.boundary_kernel,
                    stride=1,
                    padding=padding,
                )
            boundary_loss = self._single_loss(
                outputs["edge"],
                boundary_target,
                self.boundary_weight,
                self.boundary_weight,
                boundary_valid,
            )
            total = total + boundary_loss

        pbcl_loss = anchor
        pbcl_score = anchor.detach()
        pbcl_pair_count = 0
        if self.pbcl_weight > 0.0:
            if instance_maps is None or instance_pairs is None:
                raise KeyError("SSepNet PBCL requires instance_maps and instance_pairs")
            pbcl_logits = main
            if pbcl_logits.shape[-2:] != instance_maps.shape[-2:]:
                pbcl_logits = F.interpolate(
                    pbcl_logits,
                    size=instance_maps.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            pbcl_loss, pbcl_score, pbcl_pair_count = pair_bottleneck_connectivity_loss(
                pbcl_logits,
                instance_maps,
                instance_pairs,
                band_radius=self.pbcl_band_radius,
                iterations=self.pbcl_iterations,
                threshold=self.pbcl_threshold,
                margin=self.pbcl_margin,
                neighbors=self.pbcl_neighbors,
            )
            total = total + self.pbcl_weight * pbcl_loss

        self.last_components = {
            "main": float(main_loss.detach().float().cpu()),
            "aux": float(aux_loss.detach().float().cpu()),
            "boundary": float(boundary_loss.detach().float().cpu()),
            "pbcl": float(pbcl_loss.detach().float().cpu()),
            "pbcl_weighted": float((self.pbcl_weight * pbcl_loss).detach().float().cpu()),
            "pbcl_score": float(pbcl_score.detach().float().cpu()),
            "pbcl_pair_count": int(pbcl_pair_count),
        }
        return total
