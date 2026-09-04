from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .specs import ExperimentSpec

PUBLIC_CONFIG_FILENAME = "PBCL-DINO.py"
PUBLIC_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / PUBLIC_CONFIG_FILENAME


@dataclass(frozen=True)
class ExperimentComponents:
    config: Any
    model: Any
    loss: Any
    optimizer: Any
    scheduler: Any
    datamodule: Any


@dataclass(frozen=True)
class ResolvedExperiment:
    spec: ExperimentSpec
    config_path: Path
    components: ExperimentComponents | None = None

    @property
    def config(self) -> Any | None:
        return self.components.config if self.components is not None else None

    def write_resolved_config(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else Path(self.spec.artifacts["weights_path"]) / "resolved_config.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.spec.to_json(), encoding="utf-8")
        return target


def load_experiment(
    config_path: str | Path,
    *,
    build: bool = True,
    seed: int | None = None,
) -> ResolvedExperiment:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path != PUBLIC_CONFIG_PATH:
        raise ValueError(f"PBCL-DINO accepts only config/{PUBLIC_CONFIG_FILENAME}")

    from geoseg.experiments.aquadataset_catalog import build_aquadataset_spec

    resolved_seed = 42 if seed is None else int(seed)
    spec = build_aquadataset_spec(path.stem, resolved_seed, source_config=path)
    unresolved = ResolvedExperiment(spec=spec, config_path=path)
    if not build:
        return unresolved

    from .aquav3_builder import build_aquav3_components

    resolved_spec, components = build_aquav3_components(spec)
    return ResolvedExperiment(spec=resolved_spec, config_path=path, components=components)
