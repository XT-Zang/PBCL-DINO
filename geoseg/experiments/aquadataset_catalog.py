from __future__ import annotations

import os
from pathlib import Path

from geoseg.metrics.aquadataset import AQUADATASET_OVERALL_SCORE_WEIGHTS
from geoseg.pretrained import (
    DINOV3_CHECKPOINT_NAME,
    DINOV3_CHECKPOINT_SHA256,
    pretrained_weight_path,
)
from geoseg.runtime.specs import ExperimentSpec

TARGET_EXPERIMENT = "ssepnet_ms_dinov3_stage_adapter_fpn_ocr_6_6_6_6_binary_head"
AQUADATASET_VERSION = "aquav3-sceneholdout-v1"
AQUADATASET_MEAN = (0.2259130624, 0.2878868692, 0.3214092823, 0.1363257618)
AQUADATASET_STD = (0.2089709094, 0.2542853528, 0.2718809957, 0.1339930134)
AQUADATASET_SPLIT_COUNTS = {"train": 1734, "val": 432, "test": 1382}
AQUADATASET_SPLIT_HASHES = {
    "train": "2a039416e8ee73ae31296d1ac505006239f54b07ff5b947b801950d25d0c823c",
    "val": "b409dc3ca697a998f3a814a1c5525b066cb6103e66d50bb2437951c645377783",
    "test": "72edfa9ec561137b1aed94cc96ea803b537cd418ce7e3929ae803d7e6df2f3a7",
}
AQUADATASET_SEEDS = (42, 43, 44)
DINOV3_SHA256 = DINOV3_CHECKPOINT_SHA256


def build_aquadataset_spec(
    experiment_id: str,
    seed: int,
    *,
    source_config: str | Path | None = None,
) -> ExperimentSpec:
    experiment_id = str(experiment_id).strip().lower().replace("-", "_")
    if experiment_id != TARGET_EXPERIMENT:
        raise ValueError(f"PBCL-DINO publishes only {TARGET_EXPERIMENT!r}; got {experiment_id!r}")
    seed = int(seed)
    if seed not in AQUADATASET_SEEDS:
        raise ValueError(f"seed must be one of {AQUADATASET_SEEDS}, got {seed}")

    data_root = os.environ.get("AQUADATASET_ROOT", "").strip()
    pretrained_path = pretrained_weight_path(DINOV3_CHECKPOINT_NAME)
    run_name = f"{TARGET_EXPERIMENT}_seed{seed}"
    output_dir = str(Path("train_out/Aquadataset/PBCL-DINO") / f"seed{seed}")

    loss = {
        "name": "SSepNetLoss",
        "boundary_weight": 0.6,
        "boundary_kernel": 5,
        "pbcl_weight": 0.1,
        "pbcl_band_radius": 3,
        "pbcl_iterations": 9,
        "pbcl_threshold": 0.5,
        "pbcl_margin": 0.1,
        "pbcl_neighbors": 4,
    }
    return ExperimentSpec(
        model={
            "key": TARGET_EXPERIMENT,
            "in_channels": 4,
            "num_classes": 1,
            "pretrained": True,
            "freeze_backbones": False,
            "activation_checkpoint": True,
            "dinov3_adaptation": "full",
            "dinov3_lora_rank": 0,
            "dinov3_lora_alpha": None,
            "pretrained_keys": ["dinov3-vitl16-sat493m"],
            "pretrained_sha256": {"dinov3-vitl16-sat493m": DINOV3_SHA256},
            "pretrained_artifacts": {
                "dinov3-vitl16-sat493m": {
                    "path": str(pretrained_path),
                    "sha256": DINOV3_SHA256,
                }
            },
            "architecture_revision": "stage_adapter_standard_fpn_binary_pbcl_v1",
            "stage_depths": [6, 6, 6, 6],
            "stage_output_layers": [5, 11, 17, 23],
            "stage_dims": [128, 256, 512, 512],
            "semantic_scales": [4, 8, 16, 32],
            "decoder_channels": 128,
            "segmentation_head": "standard_fpn_binary_boundary_v1",
            "boundary_head_enabled": True,
            "boundary_feature_source": "raw_p2",
            "requires_instance_supervision": True,
            "pbcl_weight": 0.1,
        },
        data={
            "dataset": "aquadataset",
            "root": data_root,
            "dataset_version": AQUADATASET_VERSION,
            "splits": {"train": "train", "val": "val", "test": "test"},
            "split_counts": dict(AQUADATASET_SPLIT_COUNTS),
            "split_hashes": dict(AQUADATASET_SPLIT_HASHES),
            "bands": ["B", "G", "R", "NIR"],
            "band_indices": [0, 1, 2, 3],
            "image_size": 512,
            "raw_clip_min": 100.0,
            "raw_clip_max": 60000.0,
            "raw_offset": 100.0,
            "raw_scale": 59900.0,
            "normalize_mean": list(AQUADATASET_MEAN),
            "normalize_std": list(AQUADATASET_STD),
            "instance_target_root": "instance_targets/pbcl_gap5",
        },
        loader={
            "train_batch_size": 12,
            "val_batch_size": 8,
            "test_batch_size": 8,
            "train_workers": 4,
            "eval_workers": 2,
            "pin_memory": True,
            "drop_last": True,
            "balanced_sampler": True,
            "persistent_workers": False,
            "prefetch_factor": None,
            "collate": "aquav3",
            "sampler_generator_stream": 1,
            "loader_generator_streams": {"train": 2, "val": 3, "test": 4},
        },
        optimization={
            "loss": loss,
            "optimizer": {
                "name": "AdamW",
                "base_lr": 2e-4,
                "encoder_lr": 2e-5,
                "weight_decay": 1e-4,
                "encoder_prefixes": ["multiscale_backbone.dinov3_backbone."],
                "no_decay_policy": "bias_norm_position_token",
            },
            "scheduler": {
                "name": "warmup_cosine",
                "warmup_epochs": 3,
                "min_lr_ratio": 0.02,
                "schedule_epochs": 40,
            },
            "gradient_clip_val": 1.0,
            "accumulate_grad_batches": 1,
        },
        runtime={
            "seed": seed,
            "max_epochs": 40,
            "precision": "bf16-mixed",
            "devices": "auto",
            "check_val_every_n_epoch": 1,
            "num_sanity_val_steps": 0,
            "compile": False,
            "channels_last": False,
            "cudnn_benchmark": False,
        },
        checkpoint={
            "policy": "three_best_validation_v1",
            "monitors": {
                "region": "val_fg_iou",
                "structure": "val_pq",
                "overall": "val_overall_score",
            },
            "primary_role": "overall",
            "monitor": "val_overall_score",
            "mode": "max",
            "save_top_k": 1,
            "every_n_epochs": 0,
            "save_last": False,
            "declared_save_last": False,
            "selection": "strict_improvement_earliest_tie",
            "save_weights_only": True,
            "overall_score": {
                "name": "region_boundary_object_v1",
                "weights": dict(AQUADATASET_OVERALL_SCORE_WEIGHTS),
            },
        },
        metrics={
            "binary": True,
            "threshold": 0.5,
            "deadhesion_validation": True,
            "boundary_width": 2,
            "boundary_tolerance": 2,
            "min_overlap_pixels": 8,
            "min_overlap_ratio": 0.05,
            "instance_segmentation": False,
            "prediction_kind": "connected_components",
        },
        artifacts={
            "run_name": run_name,
            "weights_path": output_dir,
            "log_name": f"aquadataset/PBCL-DINO/seed{seed}",
            "save_val_predictions": False,
            "val_prediction_dir": f"{output_dir}/val_predictions",
            "async_writes": False,
        },
        provenance={
            "schema_version": 1,
            "campaign": "aquadataset",
            "family": "PBCL-DINO",
            "experiment_id": TARGET_EXPERIMENT,
            "model_key": TARGET_EXPERIMENT,
            "treatment": "boundary_pbcl",
            "training_protocol": "uniform_40ep_stage_adapter_fpn_binary_pbcl_v1",
            "source_config": str(source_config) if source_config else "",
            "seed_scope": list(AQUADATASET_SEEDS),
        },
    )
