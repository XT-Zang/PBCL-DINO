from __future__ import annotations

import math
from types import MappingProxyType
from typing import Mapping


AQUADATASET_OVERALL_SCORE_WEIGHTS = MappingProxyType(
    {
        "fg_iou": 0.30,
        "boundary_f1": 0.20,
        "pq": 0.25,
        "close_pair_recall": 0.10,
        "merge_rate": 0.05,
        "split_rate": 0.05,
        "count_nmae": 0.05,
    }
)


def aquadataset_overall_score(metrics: Mapping[str, float]) -> float:
    values: dict[str, float] = {}
    for name in AQUADATASET_OVERALL_SCORE_WEIGHTS:
        value = float(metrics[name])
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
        values[name] = value
    return (
        0.30 * values["fg_iou"]
        + 0.20 * values["boundary_f1"]
        + 0.25 * values["pq"]
        + 0.10 * values["close_pair_recall"]
        + 0.05 * (1.0 - min(max(values["merge_rate"], 0.0), 1.0))
        + 0.05 * (1.0 - min(max(values["split_rate"], 0.0), 1.0))
        + 0.05 * (1.0 - min(max(values["count_nmae"], 0.0), 1.0))
    )
