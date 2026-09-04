from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_experiment(seed: int) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_generator(seed: int, *, stream: int = 0) -> torch.Generator:
    if stream < 0:
        raise ValueError("stream must be non-negative")
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(stream))
    return generator
