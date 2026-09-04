from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from geoseg.engine.callbacks import MultiMetricBestCheckpoint, TrainingManifestCallback
from geoseg.engine.task import SegmentationTask
from geoseg.experiments.reproducibility import seed_experiment
from geoseg.runtime import apply_runtime_optimizations, load_experiment, preflight_experiment


Supervision_Train = SegmentationTask


def write_training_protocol_manifest(config, config_path: str | Path) -> Path:
    payload = dict(config.training_protocol)
    payload["config_path"] = str(config_path)
    path = Path(config.weights_path) / "training_protocol.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def build_trainer_kwargs(config, experiment) -> dict:
    checkpoint = MultiMetricBestCheckpoint(
        directory=config.weights_path,
        filename=config.weights_name,
        monitors=dict(config.checkpoint_monitors),
        save_weights_only=bool(config.checkpoint_save_weights_only),
    )
    logger = CSVLogger("lightning_logs", name=config.log_name)
    return {
        "devices": config.gpus,
        "max_epochs": config.max_epoch,
        "accelerator": "auto",
        "check_val_every_n_epoch": config.check_val_every_n_epoch,
        "callbacks": [checkpoint, TrainingManifestCallback(experiment)],
        "strategy": "auto",
        "logger": logger,
        "enable_checkpointing": False,
        "precision": config.precision,
        "gradient_clip_val": config.gradient_clip_val,
        "accumulate_grad_batches": config.accumulate_grad_batches,
        "num_sanity_val_steps": config.num_sanity_val_steps,
    }


def run_training(config_path: str | Path, *, seed: int | None = None) -> None:
    path = Path(config_path).expanduser().resolve()
    unresolved = load_experiment(path, build=False, seed=seed)
    preflight_experiment(unresolved)

    from geoseg.runtime.aquav3_builder import build_aquav3_components

    spec, components = build_aquav3_components(unresolved.spec)
    experiment = replace(unresolved, spec=spec, components=components)
    seed_experiment(int(spec.runtime["seed"]))
    experiment = apply_runtime_optimizations(experiment)
    config = experiment.config
    experiment.write_resolved_config()
    write_training_protocol_manifest(config, path)

    model = Supervision_Train(config)
    trainer = pl.Trainer(**build_trainer_kwargs(config, experiment))
    trainer.fit(model=model)
