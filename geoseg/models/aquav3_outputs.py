from __future__ import annotations

from collections.abc import Mapping

import torch


def get_main_logits(outputs) -> torch.Tensor:
    if isinstance(outputs, Mapping):
        if "out" not in outputs:
            raise KeyError("model output mapping does not contain 'out'")
        return outputs["out"]
    if isinstance(outputs, (tuple, list)):
        if not outputs:
            raise ValueError("model output sequence is empty")
        return outputs[0]
    if not isinstance(outputs, torch.Tensor):
        raise TypeError(f"unsupported model output type: {type(outputs)!r}")
    return outputs
