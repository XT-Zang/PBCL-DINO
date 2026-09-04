from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from typing import Mapping

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from .experiment import ResolvedExperiment

DEPENDENCY_POLICY: Mapping[str, str] = {
    "torch": ">=2.11,<2.12",
    "pytorch-lightning": ">=2.6,<2.7",
    "rasterio": ">=1.4,<1.5",
    "tifffile": ">=2026.3,<2027",
    "Pillow": ">=12,<13",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_dependency_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in DEPENDENCY_POLICY:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "missing"
    return result


def validate_dependency_versions(versions: Mapping[str, str] | None = None) -> None:
    actual = dict(versions or installed_dependency_versions())
    errors: list[str] = []
    for name, requirement in DEPENDENCY_POLICY.items():
        raw = actual.get(name, "missing")
        if raw == "missing":
            errors.append(f"{name} is not installed; required {requirement}")
        elif Version(raw) not in SpecifierSet(requirement):
            errors.append(f"{name}=={raw} is incompatible with {requirement}")
    if errors:
        raise RuntimeError("Dependency preflight failed:\n- " + "\n- ".join(errors))


def validate_aquadataset_data_contract(experiment: ResolvedExperiment) -> dict[str, object]:
    import numpy as np
    import tifffile

    data = experiment.spec.data
    root_value = str(data.get("root", "")).strip()
    if not root_value:
        raise RuntimeError("Set AQUADATASET_ROOT before running PBCL-DINO")
    root = Path(root_value).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"Aquadataset root is missing: {root}")

    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"Aquadataset metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_version = str(data["dataset_version"])
    if metadata.get("version") != expected_version:
        raise RuntimeError(
            f"Aquadataset version mismatch: expected {expected_version}, got {metadata.get('version')}"
        )

    split_counts: dict[str, int] = {}
    all_records: list[dict[str, object]] = []
    for split, expected_count in dict(data["split_counts"]).items():
        path = root / "splits" / f"{split}.jsonl"
        if not path.is_file():
            raise RuntimeError(f"Aquadataset split is missing: {path}")
        actual_hash = _sha256_file(path)
        expected_hash = str(data["split_hashes"][split])
        if actual_hash != expected_hash:
            raise RuntimeError(f"Aquadataset {split} hash mismatch")
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if len(records) != int(expected_count):
            raise RuntimeError(
                f"Aquadataset {split} count mismatch: expected {expected_count}, got {len(records)}"
            )
        split_counts[str(split)] = len(records)
        all_records.extend(records)

    sample_path = root / str(all_records[0]["image"])
    with tifffile.TiffFile(sample_path) as dataset:
        series = dataset.series[0]
        sample = {
            "shape": tuple(int(v) for v in series.shape),
            "dtype": str(series.dtype),
        }
    if sample != {"shape": (512, 512, 4), "dtype": "uint16"}:
        raise RuntimeError(f"Aquadataset raster contract mismatch: {sample}")

    target_root = root / str(data["instance_target_root"])
    positive_records = [record for record in all_records if not bool(record.get("is_background", False))]
    missing = 0
    pair_count = 0
    for record in positive_records:
        path = target_root / f"{record['tile_id']}.npz"
        if not path.is_file():
            missing += 1
            continue
        with np.load(path, allow_pickle=False) as payload:
            if "instance_map" not in payload or "instance_pairs" not in payload:
                raise RuntimeError(f"Invalid PBCL target: {path}")
            pair_count += int(len(payload["instance_pairs"]))
    if missing:
        raise RuntimeError(f"PBCL targets are missing for {missing} positive tiles")
    return {
        "dataset_version": expected_version,
        "split_counts": split_counts,
        "sample": sample,
        "pbcl": {"positive_tiles": len(positive_records), "pairs": pair_count, "missing_targets": 0},
    }


def validate_pretrained_contract(experiment: ResolvedExperiment) -> dict[str, object]:
    artifact = dict(experiment.spec.model["pretrained_artifacts"])["dinov3-vitl16-sat493m"]
    path = Path(str(artifact["path"])).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(
            "DINOv3 checkpoint is missing. Expected the standard PyTorch cache path: " + str(path)
        )
    expected = str(artifact["sha256"])
    actual = _sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"DINOv3 checkpoint SHA256 mismatch: expected {expected}, got {actual}")
    return {"path": str(path), "sha256": actual, "backend": "torch"}


def collect_environment_report() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": installed_dependency_versions(),
        "AQUADATASET_ROOT": os.environ.get("AQUADATASET_ROOT", ""),
        "TORCH_HOME": os.environ.get("TORCH_HOME", ""),
    }


def write_environment_report(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(collect_environment_report(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def preflight_experiment(
    experiment: ResolvedExperiment,
    *,
    report_path: str | Path | None = None,
) -> dict[str, object]:
    validate_dependency_versions()
    report = {
        "status": "ok",
        "experiment_id": str(experiment.spec.provenance["experiment_id"]),
        "seed": int(experiment.spec.runtime["seed"]),
        "dataset": validate_aquadataset_data_contract(experiment),
        "pretrained": validate_pretrained_contract(experiment),
        "environment": collect_environment_report(),
    }
    if report_path is not None:
        target = Path(report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def apply_runtime_optimizations(experiment: ResolvedExperiment) -> ResolvedExperiment:
    if experiment.components is None:
        return experiment
    import torch

    spec = experiment.spec
    torch.backends.cudnn.benchmark = bool(spec.runtime.get("cudnn_benchmark", False))
    model = experiment.components.model
    if bool(spec.runtime.get("channels_last", False)):
        model.to(memory_format=torch.channels_last)
    if bool(spec.runtime.get("compile", False)):
        compiled = torch.compile(model)
        experiment.components.config.net = compiled
        components = replace(experiment.components, model=compiled)
        return replace(experiment, components=components)
    return experiment
