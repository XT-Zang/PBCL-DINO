from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint_forward


_STAGE_ADAPTER_DIMS = (128, 256, 512, 512)


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if int(channels) % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(int(in_channels), int(out_channels), kernel_size=int(kernel_size), stride=int(stride), padding=int(kernel_size) // 2, bias=False),
            nn.GroupNorm(_group_count(int(out_channels)), int(out_channels)),
            nn.GELU(),
        )


class SpatialPyramidStem(nn.Module):
    def __init__(self, in_chans: int, dims: Sequence[int]) -> None:
        super().__init__()
        if len(dims) != 4:
            raise ValueError("spatial pyramid stem requires four channel widths")
        widths = tuple(int(value) for value in dims)
        hidden = max(32, widths[0] // 2)
        self.stem = nn.Sequential(
            ConvGNAct(in_chans, hidden, stride=2),
            ConvGNAct(hidden, widths[0], stride=2),
        )
        self.downsamples = nn.ModuleList(
            ConvGNAct(widths[index], widths[index + 1], stride=2) for index in range(3)
        )
        self.out_channels = widths

    def first(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.stem(inputs)

    def downsample(self, stage: int, inputs: torch.Tensor) -> torch.Tensor:
        return self.downsamples[int(stage)](inputs)


def _run_token_block(block: nn.Module, tokens: torch.Tensor, rope, *, activation_checkpoint: bool) -> torch.Tensor:
    if activation_checkpoint and tokens.requires_grad:
        def forward(value: torch.Tensor) -> torch.Tensor:
            return block(value, rope)
        return activation_checkpoint_forward(forward, tokens, use_reentrant=False)
    return block(tokens, rope)


class DinoSpatialInteraction(nn.Module):
    def __init__(self, dino_dim: int, spatial_dim: int, *, detail_gate_init_bias: float = -2.0) -> None:
        super().__init__()
        self.dino_to_spatial = ConvGNAct(int(dino_dim), int(spatial_dim), kernel_size=1)
        self.spatial_to_dino = nn.Conv2d(int(spatial_dim), int(dino_dim), 1, bias=False)
        self.detail_gate = nn.Conv2d(int(spatial_dim) * 2, int(spatial_dim), 1)
        nn.init.zeros_(self.detail_gate.weight)
        nn.init.constant_(self.detail_gate.bias, float(detail_gate_init_bias))
        self.dino_gate = nn.Parameter(torch.zeros(()))
        self.last_detail_gate_statistics: torch.Tensor | None = None

    def forward(
        self,
        spatial: torch.Tensor,
        tokens: torch.Tensor,
        normalized_patch: torch.Tensor,
        *,
        patch_size: tuple[int, int],
        prefix_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patch_height, patch_width = patch_size
        batch = int(spatial.shape[0])
        dino_channels = int(normalized_patch.shape[-1])
        dino_map = normalized_patch.transpose(1, 2).reshape(batch, dino_channels, patch_height, patch_width).contiguous()
        semantic_anchor = F.interpolate(
            self.dino_to_spatial(dino_map),
            size=spatial.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        detail_gate = torch.sigmoid(self.detail_gate(torch.cat((semantic_anchor, spatial), dim=1)))
        detached = detail_gate.detach()
        self.last_detail_gate_statistics = torch.stack(
            (detached.mean(), detached.std(unbiased=False), detached.amin(), detached.amax())
        )
        spatial = semantic_anchor + detail_gate * spatial
        injected_dino = F.adaptive_avg_pool2d(
            self.spatial_to_dino(spatial), output_size=(patch_height, patch_width)
        ).flatten(2).transpose(1, 2)
        prefix = tokens[:, :prefix_tokens]
        patches = tokens[:, prefix_tokens:] + torch.tanh(self.dino_gate) * injected_dino
        return spatial, torch.cat((prefix, patches), dim=1)


class StageAdapterDINOv3Backbone(nn.Module):
    """Partition the 24 flat DINOv3 blocks into a 6/6/6/6 spatial hierarchy."""

    def __init__(
        self,
        *,
        dinov3_backbone: nn.Module,
        stage_depths: Sequence[int],
        spatial_dims: Sequence[int] = _STAGE_ADAPTER_DIMS,
        in_chans: int = 4,
        activation_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        depths = tuple(int(value) for value in stage_depths)
        if len(depths) != 4 or sum(depths) != len(dinov3_backbone.model.blocks):
            raise ValueError("stage adapter depths must partition every flat DINOv3 block")
        self.dinov3_backbone = dinov3_backbone
        self.stage_depths = depths
        self.activation_checkpoint = bool(activation_checkpoint)
        self.spatial_stem = SpatialPyramidStem(in_chans, spatial_dims)
        dino_dim = int(dinov3_backbone.model.embed_dim)
        self.interactions = nn.ModuleList(
            DinoSpatialInteraction(dino_dim, int(spatial_dim)) for spatial_dim in spatial_dims
        )
        self.out_channels = tuple(int(value) for value in spatial_dims)
        self.out_strides = (4, 8, 16, 32)
        self.initialization_report = {
            "policy": "strict_flat_dinov3_semantic_anchor_plus_gated_spatial_detail",
            "strict_source_blocks": len(dinov3_backbone.model.blocks),
            "projected_source_blocks": 0,
            "new_stage_modules": 4,
        }

    def interaction_gate_statistics(self) -> tuple[tuple[float, float, float, float], ...]:
        return tuple(
            tuple(float(value) for value in interaction.last_detail_gate_statistics.cpu())
            for interaction in self.interactions
            if interaction.last_detail_gate_statistics is not None
        )

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        model = self.dinov3_backbone.model
        tokens, (patch_height, patch_width) = model._prepare_tokens(inputs)
        prefix_tokens = int(tokens.shape[1] - patch_height * patch_width)
        rope = model.rope_embed(height=patch_height, width=patch_width)
        spatial = self.spatial_stem.first(inputs)
        outputs: list[torch.Tensor] = []
        block_index = 0
        for stage_index, depth in enumerate(self.stage_depths):
            for _ in range(depth):
                tokens = _run_token_block(
                    model.blocks[block_index],
                    tokens,
                    rope,
                    activation_checkpoint=self.activation_checkpoint and self.training,
                )
                block_index += 1
            normalized = model.norm(tokens)[:, prefix_tokens:]
            spatial, tokens = self.interactions[stage_index](
                spatial,
                tokens,
                normalized,
                patch_size=(patch_height, patch_width),
                prefix_tokens=prefix_tokens,
            )
            outputs.append(spatial)
            if stage_index < 3:
                spatial = self.spatial_stem.downsample(stage_index, spatial)
        return outputs
