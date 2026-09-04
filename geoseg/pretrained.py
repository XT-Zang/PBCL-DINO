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


DINOV3_CHECKPOINT_NAME = "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
DINOV3_RELATIVE_PATH = f"torch/hub/checkpoints/{DINOV3_CHECKPOINT_NAME}"

DEFAULT_PRETRAINED_MODELS = (
    PretrainedModelSpec(
        "dinov3-vitl16-sat493m",
        "torch",
        DINOV3_RELATIVE_PATH,
        "DINOv3 ViT-L/16 SAT-493M",
        ("PBCL-DINO",),
    ),
)


def default_pretrained_root() -> Path:
    """Return the public-release pretrained root.

    By default this is ``~/.cache`` so DINOv3 resolves from the standard
    PyTorch cache at ``~/.cache/torch/hub/checkpoints``. Set
    ``GEOSEG_PRETRAINED_ROOT`` to override the cache root.
    """
    configured = os.environ.get("GEOSEG_PRETRAINED_ROOT")
    root = Path(configured).expanduser() if configured else Path.home() / ".cache"
    return root.resolve()


def configure_pretrained_environment(root: str | Path | None = None) -> dict[str, str]:
    resolved_root = Path(root).expanduser().resolve() if root is not None else default_pretrained_root()
    torch_home = (resolved_root / "torch").resolve()
    os.environ.setdefault("GEOSEG_PRETRAINED_ROOT", str(resolved_root))
    os.environ.setdefault("TORCH_HOME", str(torch_home))
    return {
        "root": str(resolved_root),
        "torch_home": os.environ["TORCH_HOME"],
        "dinov3_checkpoint": str((torch_home / "hub" / "checkpoints" / DINOV3_CHECKPOINT_NAME).resolve()),
    }


def pretrained_weight_path(filename: str) -> Path:
    """Compatibility helper for callers that pass a checkpoint filename."""
    return (default_pretrained_root() / "torch" / "hub" / "checkpoints" / filename).resolve()


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
    """Resolve DINOv3 from the PyTorch checkpoint cache without downloading it."""
    normalized = str(key).strip().lower()
    try:
        spec = next(model for model in DEFAULT_PRETRAINED_MODELS if model.key == normalized)
    except StopIteration as exc:
        raise KeyError(f"Unknown pretrained model key: {key!r}") from exc

    resolved_root = Path(root).expanduser().resolve() if root is not None else default_pretrained_root()
    path = (resolved_root / spec.relative_path).resolve()
    if not _has_payload(path):
        raise FileNotFoundError(
            f"DINOv3 pretrained checkpoint is missing at {path}. "
            f"Place {DINOV3_CHECKPOINT_NAME} in the PyTorch checkpoint cache, "
            "or set GEOSEG_PRETRAINED_ROOT to a cache root containing torch/hub/checkpoints/."
        )
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
