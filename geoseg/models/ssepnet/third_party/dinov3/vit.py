# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement included beside this module.
#
# This is a GeoSeg-local, inference-focused extraction of the official DINOv3
# ViT implementation. Unused training/list-processing paths were removed.

from __future__ import annotations

import math
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5, device=None) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(dim, device=device))
        self.init_values = init_values

    def reset_parameters(self) -> None:
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * self.gamma


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int | None = None,
        bias: bool = True,
        device=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, device=device)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, device=device)
        self.drop = nn.Dropout(0.0)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(inputs)))))


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        device=None,
    ) -> None:
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            device=device,
        )
        self.norm = nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.proj(inputs)
        height, width = outputs.shape[-2:]
        return self.norm(outputs.flatten(2).transpose(1, 2)).reshape(
            -1, height, width, self.embed_dim
        )

    def reset_parameters(self) -> None:
        bound = math.sqrt(1 / (self.in_chans * self.patch_size[0] ** 2))
        nn.init.uniform_(self.proj.weight, -bound, bound)
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -bound, bound)


class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float = 100.0,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype = torch.float32,
        device=None,
    ) -> None:
        super().__init__()
        if embed_dim % (4 * num_heads) != 0:
            raise ValueError("RoPE embedding dimension must be divisible by four times num_heads")
        self.base = base
        self.head_dim = embed_dim // num_heads
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype
        self.register_buffer(
            "periods",
            torch.empty(self.head_dim // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        self.periods.data = self.base ** (
            2
            * torch.arange(
                self.head_dim // 4,
                device=self.periods.device,
                dtype=self.dtype,
            )
            / (self.head_dim // 2)
        )

    def forward(self, *, height: int, width: int) -> tuple[Tensor, Tensor]:
        options = {"device": self.periods.device, "dtype": self.dtype}
        if self.normalize_coords == "max":
            scale = max(height, width)
            coords_h = torch.arange(0.5, height, **options) / scale
            coords_w = torch.arange(0.5, width, **options) / scale
        elif self.normalize_coords == "min":
            scale = min(height, width)
            coords_h = torch.arange(0.5, height, **options) / scale
            coords_w = torch.arange(0.5, width, **options) / scale
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, height, **options) / height
            coords_w = torch.arange(0.5, width, **options) / width
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        coordinates = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1
        ).flatten(0, 1)
        coordinates = 2.0 * coordinates - 1.0
        if self.training and self.shift_coords is not None:
            coordinates += torch.empty(2, **options).uniform_(
                -self.shift_coords, self.shift_coords
            )[None, :]
        if self.training and self.jitter_coords is not None:
            limit = np.log(self.jitter_coords)
            coordinates *= torch.empty(2, **options).uniform_(-limit, limit).exp()[None, :]
        if self.training and self.rescale_coords is not None:
            limit = np.log(self.rescale_coords)
            coordinates *= torch.empty(1, **options).uniform_(-limit, limit).exp()
        angles = 2 * math.pi * coordinates[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        return angles.sin(), angles.cos()


def _rope_rotate_half(inputs: Tensor) -> Tensor:
    first, second = inputs.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(inputs: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    return inputs * cos + _rope_rotate_half(inputs) * sin


class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.out_features % 3 != 0:
            raise ValueError("masked QKV projection must have three equal sections")
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.full_like(self.bias, math.nan))

    def forward(self, inputs: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(inputs, self.weight, masked_bias)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        mask_k_bias: bool = True,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        linear = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, inputs: Tensor, rope: tuple[Tensor, Tensor]) -> Tensor:
        batch, tokens, channels = inputs.shape
        qkv = self.qkv(inputs).reshape(
            batch, tokens, 3, self.num_heads, channels // self.num_heads
        )
        query, key, value = [item.transpose(1, 2) for item in torch.unbind(qkv, 2)]
        sin, cos = rope
        prefix = query.shape[-2] - sin.shape[-2]
        query = torch.cat(
            (query[:, :, :prefix], _apply_rope(query[:, :, prefix:].to(sin.dtype), sin, cos)),
            dim=-2,
        ).to(inputs.dtype)
        key = torch.cat(
            (key[:, :, :prefix], _apply_rope(key[:, :, prefix:].to(sin.dtype), sin, cos)),
            dim=-2,
        ).to(inputs.dtype)
        outputs = F.scaled_dot_product_attention(query, key, value)
        outputs = outputs.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(outputs))


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        layerscale_init: float = 1e-5,
        device=None,
    ) -> None:
        super().__init__()
        def norm() -> nn.LayerNorm:
            return nn.LayerNorm(dim, eps=1e-5, device=device)

        self.norm1 = norm()
        self.attn = SelfAttention(dim, num_heads, device=device)
        self.ls1 = LayerScale(dim, init_values=layerscale_init, device=device)
        self.norm2 = norm()
        self.mlp = Mlp(dim, int(dim * ffn_ratio), device=device)
        self.ls2 = LayerScale(dim, init_values=layerscale_init, device=device)

    def forward(self, inputs: Tensor, rope: tuple[Tensor, Tensor]) -> Tensor:
        outputs = inputs + self.ls1(self.attn(self.norm1(inputs), rope=rope))
        return outputs + self.ls2(self.mlp(self.norm2(outputs)))


def _initialize_module(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        bias_mask = getattr(module, "bias_mask", None)
        if bias_mask is not None:
            output_features = module.out_features
            bias_mask.fill_(1)
            bias_mask[output_features // 3 : 2 * output_features // 3].fill_(0)
    elif isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    elif isinstance(module, LayerScale):
        module.reset_parameters()
    elif isinstance(module, PatchEmbed):
        module.reset_parameters()


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        ffn_ratio: float = 4.0,
        n_storage_tokens: int = 4,
        untie_global_and_local_cls_norm: bool = True,
        device=None,
    ) -> None:
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            device=device,
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        self.storage_tokens = nn.Parameter(
            torch.empty(1, n_storage_tokens, embed_dim, device=device)
        )
        self.rope_embed = RopePositionEmbedding(
            embed_dim,
            num_heads=num_heads,
            base=100.0,
            normalize_coords="separate",
            rescale_coords=2.0,
            dtype=torch.float32,
            device=device,
        )
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    embed_dim,
                    num_heads,
                    ffn_ratio=ffn_ratio,
                    layerscale_init=1e-5,
                    device=device,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-5, device=device)
        self.untie_cls_and_patch_norms = False
        self.cls_norm = None
        self.untie_global_and_local_cls_norm = bool(untie_global_and_local_cls_norm)
        self.local_cls_norm = (
            nn.LayerNorm(embed_dim, eps=1e-5, device=device)
            if self.untie_global_and_local_cls_norm
            else None
        )
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))
        self.init_weights()

    def init_weights(self) -> None:
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        self.apply(_initialize_module)

    def _prepare_tokens(self, inputs: Tensor) -> tuple[Tensor, tuple[int, int]]:
        patches = self.patch_embed(inputs)
        batch, height, width, _ = patches.shape
        patches = patches.flatten(1, 2)
        tokens = torch.cat(
            (
                self.cls_token.expand(batch, -1, -1),
                self.storage_tokens.expand(batch, -1, -1),
                patches,
            ),
            dim=1,
        )
        return tokens, (height, width)

    def get_intermediate_layers(
        self,
        inputs: Tensor,
        *,
        n: Sequence[int],
        reshape: bool = True,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> tuple[Tensor, ...]:
        if return_class_token or return_extra_tokens:
            raise ValueError("GeoSeg's embedded DINOv3 path returns patch features only")
        tokens, (height, width) = self._prepare_tokens(inputs)
        wanted = tuple(int(index) for index in n)
        wanted_set = set(wanted)
        collected: dict[int, Tensor] = {}
        for index, block in enumerate(self.blocks):
            tokens = block(tokens, self.rope_embed(height=height, width=width))
            if index in wanted_set:
                collected[index] = tokens
        if set(collected) != wanted_set:
            raise ValueError(f"requested DINOv3 layers are outside 0..{len(self.blocks) - 1}: {wanted}")
        outputs = [collected[index] for index in wanted]
        if norm:
            outputs = [self.norm(output) for output in outputs]
        outputs = [output[:, self.n_storage_tokens + 1 :] for output in outputs]
        if reshape:
            outputs = [
                output.reshape(inputs.shape[0], height, width, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for output in outputs
            ]
        return tuple(outputs)
