from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .third_party.dinov3 import DinoVisionTransformer


DINOV3_OUT_LAYERS = (5, 11, 17, 23)


def inflate_input_conv(source: nn.Conv2d, *, in_chans: int = 4) -> nn.Conv2d:
    if source.in_channels != 3 or in_chans < 3:
        raise ValueError(
            f"DINOv3 input inflation requires a 3-channel source and at least 3 targets, got "
            f"{source.in_channels} and {in_chans}"
        )
    inflated = nn.Conv2d(
        in_chans,
        source.out_channels,
        kernel_size=source.kernel_size,
        stride=source.stride,
        padding=source.padding,
        dilation=source.dilation,
        groups=source.groups,
        bias=source.bias is not None,
        padding_mode=source.padding_mode,
        device=source.weight.device,
        dtype=source.weight.dtype,
    )
    with torch.no_grad():
        inflated.weight[:, :3].copy_(source.weight)
        visible_mean = source.weight.mean(dim=1, keepdim=True)
        for channel in range(3, in_chans):
            inflated.weight[:, channel : channel + 1].copy_(visible_mean)
        if source.bias is not None:
            inflated.bias.copy_(source.bias)
    return inflated


def _checkpoint_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("DINOv3 checkpoint must contain a state-dict mapping")
    for key in ("model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    return checkpoint


class DinoV3ViTL16(nn.Module):
    def __init__(
        self,
        *,
        in_chans: int = 4,
        checkpoint_path: str | Path | None,
        device=None,
    ) -> None:
        super().__init__()
        if in_chans not in {3, 4}:
            raise ValueError("SSepNet DINOv3 requires 3-channel RGB or 4-channel B/G/R/NIR input")
        self.out_layers = DINOV3_OUT_LAYERS
        self.model = DinoVisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            ffn_ratio=4.0,
            n_storage_tokens=4,
            untie_global_and_local_cls_norm=True,
            device=device,
        )
        self.embed_dim = self.model.embed_dim
        if checkpoint_path is not None:
            self.load_pretrained(checkpoint_path)
        if in_chans == 4:
            self.model.patch_embed.proj = inflate_input_conv(
                self.model.patch_embed.proj,
                in_chans=in_chans,
            )

    @property
    def out_channels(self) -> int:
        return self.embed_dim * len(self.out_layers)

    def load_pretrained(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"DINOv3 ViT-L/16 SAT-493M checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        state = {
            str(key).removeprefix("module.").removeprefix("backbone."): value
            for key, value in _checkpoint_state_dict(checkpoint).items()
            if isinstance(value, torch.Tensor)
        }
        self.model.load_state_dict(state, strict=True)

    def forward_layers(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        return list(
            self.model.get_intermediate_layers(
                inputs,
                n=self.out_layers,
                reshape=True,
                return_class_token=False,
                return_extra_tokens=False,
                norm=True,
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat(self.forward_layers(inputs), dim=1)


def build_dinov3_vitl16_sat493m(
    *,
    in_chans: int = 4,
    pretrained: bool = True,
    checkpoint_path: str | Path | None = None,
) -> DinoV3ViTL16:
    if in_chans not in {3, 4}:
        raise ValueError("SSepNet DINOv3 requires 3-channel RGB or 4-channel B/G/R/NIR input")
    if pretrained:
        if checkpoint_path is None:
            from geoseg.pretrained import resolve_pretrained_model_path

            checkpoint_path = resolve_pretrained_model_path("dinov3-vitl16-sat493m")
    elif checkpoint_path is not None:
        raise ValueError("checkpoint_path requires pretrained=True")
    return DinoV3ViTL16(in_chans=in_chans, checkpoint_path=checkpoint_path)
