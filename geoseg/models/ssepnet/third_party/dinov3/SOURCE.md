# Embedded DINOv3 source notice

The code in this directory is an inference-focused extraction of Meta's official DINOv3 implementation used by PBCL-DINO.

Only the ViT components required to construct DINOv3 ViT-L/16 SAT-493M and expose patch-token features are retained. Training-only, distributed, text, detection, depth, and unrelated backbone code is omitted. Parameter names required by the published pretrained checkpoint remain compatible with the official implementation.

The files in this directory are governed by the adjacent `LICENSE.md`; they are not relicensed under PBCL-DINO's GPL-3.0 license.
