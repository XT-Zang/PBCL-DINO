# PBCL-DINO

Minimal, reproducible implementation of **PBCL-DINO** for dense aquaculture-raft segmentation on **Aquadataset**.

PBCL-DINO was developed from [GeoSeg](https://github.com/WangLibo1995/GeoSeg). This repository distills that codebase into a focused release containing only the data pipeline, model, loss, metrics, and training runtime needed for the PBCL-DINO experiment.

## Overview

PBCL-DINO combines a remote-sensing foundation model with explicit boundary and instance-connectivity supervision:

```text
4-band image → DINOv3 ViT-L/16 → 6/6/6/6 Stage-Adapter → FPN → segmentation
                                                                  ├─ auxiliary supervision
                                                                  └─ boundary supervision + PBCL
```

| Component | Configuration |
| --- | --- |
| Backbone | DINOv3 ViT-L/16, pretrained on SAT-493M |
| Hierarchy | Four Stage-Adapter stages with depths 6/6/6/6 |
| Decoder | Standard FPN binary segmentation head |
| Inputs | Four bands: B, G, R, and NIR |
| Objective | Segmentation + auxiliary + boundary + Pairwise Bottleneck Connectivity Loss |

The release contains one locked experiment configuration:

[`config/aquadataset_stage_adapter_fpn_ocr/ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6_binary_head.py`](config/aquadataset_stage_adapter_fpn_ocr/ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6_binary_head.py)

Other GeoSeg datasets, baselines, archived experiments, campaign scripts, and legacy evaluation wrappers are intentionally excluded.

## Installation

Python 3.11 and PyTorch 2.11.x are used by the reference runtime. Install a [PyTorch build](https://pytorch.org/get-started/locally/) matching your CUDA environment, then install the remaining dependencies:

```bash
git clone https://github.com/XT-Zang/PBCL-DINO.git
cd PBCL-DINO

python3 -m venv .venv
source .venv/bin/activate
```

Install a CUDA-compatible PyTorch 2.11.x build with the official selector, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

The required DINOv3 pretrained weights are downloaded and loaded automatically on first use. Access remains subject to the [DINOv3 license](geoseg/models/ssepnet/third_party/dinov3/LICENSE.md).

## Dataset

Aquadataset is not distributed with this repository. Point `AQUADATASET_ROOT` to a prepared dataset before running an experiment:

```bash
export AQUADATASET_ROOT=/path/to/Aquadataset
```

Expected layout:

```text
Aquadataset/
├── metadata.json
├── splits/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── ... image and semantic-mask files referenced by the manifests ...
└── instance_targets/
    └── pbcl_gap5/
        └── <tile_id>.npz
```

Each positive PBCL target must contain `instance_map` and `instance_pairs`. These targets are shared by PBCL training and the structural validation metrics.

## Quick start

Inspect the resolved experiment:

```bash
python run.py inspect
```

Validate the environment, dataset protocol, PBCL targets, and pretrained weights:

```bash
python run.py preflight
```

Start training:

```bash
python run.py train
```

All commands use seed **42** by default. Seeds 43 and 44 are available for multi-seed reproduction when explicitly requested, for example:

```bash
python run.py train --seed 43
```

## Reproducibility

The locked recipe uses:

| Setting | Value |
| --- | --- |
| Epochs | 40 |
| Optimizer | AdamW |
| Schedule | 3-epoch warm-up + cosine decay |
| Train batch size | 12 |
| Precision | bfloat16 mixed precision |
| Boundary-loss weight | 0.6 |
| PBCL weight | 0.1 |
| Registered seeds | 42, 43, 44 |

Training artifacts are written under `train_out/Aquadataset/PBCL-DINO/seed<seed>/`.

## Acknowledgements

- [GeoSeg](https://github.com/WangLibo1995/GeoSeg) provides the foundation from which PBCL-DINO was developed.
- [DINOv3](https://github.com/facebookresearch/dinov3) provides the pretrained vision backbone. Its vendored source and weights remain under Meta's DINOv3 terms.

## License

PBCL-DINO is distributed under the [GNU GPL v3.0](LICENSE), except for the vendored DINOv3 implementation. See [`geoseg/models/ssepnet/third_party/dinov3/LICENSE.md`](geoseg/models/ssepnet/third_party/dinov3/LICENSE.md) for the applicable DINOv3 terms.
