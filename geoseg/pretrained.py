from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PretrainedModelSpec:
    key: str
    framework: str
    relative_path: str
    source: str
    consumers: tuple[str, ...]
    revision: str = ""
    url: str = ""
    sha256: str = ""


DINOV3_CHECKPOINT_NAME = "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
DINOV3_RELATIVE_PATH = f"torch/hub/checkpoints/{DINOV3_CHECKPOINT_NAME}"
DINOV3_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16/"
    f"{DINOV3_CHECKPOINT_NAME}"
)
DINOV3_CHECKPOINT_SHA256 = "eadcf0ffc02418b6c22a885ea1a7aaeeef84fbf0f5bb4d0b7d1d36e68a964f48"

DEFAULT_PRETRAINED_MODELS = (
    PretrainedModelSpec(
        "dinov3-vitl16-sat493m",
        "torch",
        DINOV3_RELATIVE_PATH,
        "DINOv3 ViT-L/16 SAT-493M",
        ("PBCL-DINO",),
        url=DINOV3_CHECKPOINT_URL,
        sha256=DINOV3_CHECKPOINT_SHA256,
    ),
)


def default_pretrained_root() -> Path:
    """Return the optional compatibility cache root."""
    configured = os.environ.get("GEOSEG_PRETRAINED_ROOT")
    root = Path(configured).expanduser() if configured else Path.home() / ".cache"
    return root.resolve()


def configure_pretrained_environment(root: str | Path | None = None) -> dict[str, str]:
    if root is not None:
        resolved_root = Path(root).expanduser().resolve()
        os.environ["GEOSEG_PRETRAINED_ROOT"] = str(resolved_root)
        os.environ["TORCH_HOME"] = str((resolved_root / "torch").resolve())
    elif os.environ.get("GEOSEG_PRETRAINED_ROOT") and not os.environ.get("TORCH_HOME"):
        resolved_root = default_pretrained_root()
        os.environ["TORCH_HOME"] = str((resolved_root / "torch").resolve())
    checkpoint = pretrained_weight_path(DINOV3_CHECKPOINT_NAME)
    return {
        "root": str(default_pretrained_root()),
        "torch_home": str(checkpoint.parents[2]),
        "dinov3_checkpoint": str(checkpoint),
    }


def pretrained_weight_path(filename: str) -> Path:
    """Return a checkpoint path using PyTorch's normal Hub cache rules."""
    configured = os.environ.get("GEOSEG_PRETRAINED_ROOT")
    if configured:
        checkpoint_dir = Path(configured).expanduser().resolve() / "torch" / "hub" / "checkpoints"
    else:
        import torch

        checkpoint_dir = Path(torch.hub.get_dir()).expanduser().resolve() / "checkpoints"
    return (checkpoint_dir / filename).resolve()


def resolve_legacy_pretrained_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if len(candidate.parts) == 2 and candidate.parts[0] == "pretrain_weights":
        return pretrained_weight_path(candidate.name)
    return candidate


def _has_payload(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def resolve_pretrained_model_path(key: str, *, root: str | Path | None = None) -> Path:
    """Resolve a pretrained model, downloading it through PyTorch Hub if needed."""
    normalized = str(key).strip().lower()
    try:
        spec = next(model for model in DEFAULT_PRETRAINED_MODELS if model.key == normalized)
    except StopIteration as exc:
        raise KeyError(f"Unknown pretrained model key: {key!r}") from exc

    path = (
        (Path(root).expanduser().resolve() / spec.relative_path).resolve()
        if root is not None
        else pretrained_weight_path(DINOV3_CHECKPOINT_NAME)
    )
    if not _has_payload(path):
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        if not spec.url or not spec.sha256:
            raise RuntimeError(f"No verified download is configured for {spec.key!r}")
        try:
            torch.hub.download_url_to_file(
                spec.url,
                str(path),
                hash_prefix=spec.sha256,
                progress=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Unable to download the DINOv3 ViT-L/16 SAT-493M pretrained weights. "
                "Confirm that your network can reach the official DINOv3 download and "
                "that you have accepted Meta's DINOv3 license."
            ) from exc
    return path


def _path_size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def index_pretrained_models(
    *,
    root: str | Path | None = None,
    workspace_root: str | Path | None = None,
    models: Iterable[PretrainedModelSpec] = DEFAULT_PRETRAINED_MODELS,
) -> dict[str, object]:
    del workspace_root
    resolved_root = Path(root).expanduser().resolve() if root is not None else default_pretrained_root()
    entries: list[dict[str, object]] = []
    for spec in models:
        path = resolved_root / spec.relative_path
        available = _has_payload(path)
        entry = asdict(spec)
        entry.update(
            {
                "path": str(path),
                "status": "available" if available else "missing",
                "size_bytes": _path_size(path),
            }
        )
        entries.append(entry)
    available_count = sum(entry["status"] == "available" for entry in entries)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(resolved_root),
        "environment": {"TORCH_HOME": str(resolved_root / "torch")},
        "summary": {
            "available": available_count,
            "missing": len(entries) - available_count,
            "total": len(entries),
        },
        "models": entries,
    }
