# PBCL-DINO

Minimal public implementation of **PBCL-DINO** for dense aquaculture-raft segmentation on **Aquadataset**.

PBCL-DINO was developed from [GeoSeg](https://github.com/WangLibo1995/GeoSeg). This repository turns that research codebase into a focused, reproducible release containing only the data pipeline, model, loss, metrics, and training runtime required for the published PBCL-DINO experiment.

## Model

This repository publishes exactly one experiment configuration:

`config/aquadataset_stage_adapter_fpn_ocr/ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6_binary_head.py`

The model contains:

- DINOv3 ViT-L/16 SAT-493M backbone;
- 6/6/6/6 Stage-Adapter hierarchy;
- Standard FPN binary segmentation head;
- boundary supervision;
- Pairwise Bottleneck Connectivity Loss (PBCL).

Other GeoSeg datasets, configs, baselines, archived experiments, campaign scripts, and legacy evaluation wrappers are intentionally excluded.

## Installation

Install a PyTorch build matching your CUDA environment first. The reference runtime uses PyTorch 2.11.x. Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The required DINOv3 pretrained weights are downloaded and loaded automatically on first use.

## Aquadataset

Set the dataset root before running the experiment:

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
├── ... image / semantic-mask files referenced by the manifests ...
└── instance_targets/
    └── pbcl_gap5/
        └── <tile_id>.npz
```

Each positive PBCL target contains `instance_map` and `instance_pairs`. The same targets are used for PBCL training and structural validation metrics. The dataset is not distributed in this repository.

## Reproduce

Inspect the locked experiment specification:

```bash
python run.py inspect
```

Validate dependencies, dataset protocol, PBCL targets, and the DINOv3 checkpoint:

```bash
python run.py preflight
```

Train:

```bash
python run.py train
```

All commands use seed **42** by default. Seeds **43** and **44** remain available through `--seed` for multi-seed reproduction. The locked training recipe uses 40 epochs, AdamW, warm-up cosine scheduling, Boundary weight 0.6, and PBCL weight 0.1.

## License

PBCL-DINO is distributed under **GNU GPL v3.0** except for the vendored DINOv3 implementation. See the top-level `LICENSE` notice. DINOv3 remains subject to Meta's separate terms in `geoseg/models/ssepnet/third_party/dinov3/LICENSE.md`.
