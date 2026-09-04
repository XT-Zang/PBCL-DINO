from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if int(channels) % groups == 0:
            return groups
    return 1


class ConvGNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__(
            nn.Conv2d(int(in_channels), int(out_channels), kernel_size=int(kernel_size), padding=int(kernel_size) // 2, bias=False),
            nn.GroupNorm(_group_count(int(out_channels)), int(out_channels)),
            nn.ReLU(inplace=True),
        )


def _resize_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.shape[-2:] == target.shape[-2:]:
        return source
    return F.interpolate(source, size=target.shape[-2:], mode="bilinear", align_corners=False)


def _resize_outputs(outputs: dict[str, torch.Tensor], input_size: tuple[int, int]) -> dict[str, torch.Tensor]:
    return {
        key: F.interpolate(value, size=input_size, mode="bilinear", align_corners=False)
        if key in {"out", "aux", "edge"} and value.shape[-2:] != input_size
        else value
        for key, value in outputs.items()
    }


class StandardPyramidDecoder(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.smooth = nn.ModuleList(ConvGNReLU(channels, channels, 3) for _ in range(4))
        self.aggregate = nn.Sequential(
            ConvGNReLU(channels * 4, channels, 3),
            nn.Dropout2d(float(dropout)),
        )

    def forward(self, features: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(features) != 4:
            raise ValueError(f"pyramid decoder requires four features, got {len(features)}")
        p3 = self.smooth[3](features[3])
        p2 = self.smooth[2](features[2] + _resize_like(p3, features[2]))
        p1 = self.smooth[1](features[1] + _resize_like(p2, features[1]))
        p0 = self.smooth[0](features[0] + _resize_like(p1, features[0]))
        fused = self.aggregate(
            torch.cat((p0, _resize_like(p1, p0), _resize_like(p2, p0), _resize_like(p3, p0)), dim=1)
        )
        return p0, p1, p2, p3, fused


class PyramidBinaryHead(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        structure_channels = max(32, int(channels) // 2)
        self.main = nn.Sequential(nn.Dropout2d(float(dropout)), nn.Conv2d(channels, 1, 1))
        self.auxiliary = nn.Conv2d(channels, 1, 1)
        self.boundary = nn.Sequential(
            ConvGNReLU(channels, structure_channels, 3),
            nn.Conv2d(structure_channels, 1, 1),
        )

    def forward(self, pyramid: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
        p0, _p1, p2, _p3, fused = pyramid
        return {
            "out": self.main(fused),
            "aux": self.auxiliary(p2),
            "edge": self.boundary(p0),
        }
