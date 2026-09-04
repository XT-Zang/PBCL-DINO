from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"ExperimentSpec contains non-serializable value {type(value)!r}")


@dataclass(frozen=True)
class ExperimentSpec:
    model: Mapping[str, Any]
    data: Mapping[str, Any]
    loader: Mapping[str, Any]
    optimization: Mapping[str, Any]
    runtime: Mapping[str, Any]
    checkpoint: Mapping[str, Any]
    metrics: Mapping[str, Any]
    artifacts: Mapping[str, Any]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not str(self.model.get("key", "")).strip():
            raise ValueError("ExperimentSpec.model.key is required")
        if not str(self.data.get("dataset", "")).strip():
            raise ValueError("ExperimentSpec.data.dataset is required")
        if int(self.runtime.get("seed", -1)) < 0:
            raise ValueError("ExperimentSpec.runtime.seed must be non-negative")
        if int(self.runtime.get("max_epochs", 0)) <= 0:
            raise ValueError("ExperimentSpec.runtime.max_epochs must be positive")
        _plain(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
