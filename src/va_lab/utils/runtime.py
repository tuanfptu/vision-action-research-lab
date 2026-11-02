"""Seeds, configuration and structured experiment records."""

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as handle:
        cfg = yaml.safe_load(handle)
    if (
        cfg["chunk"] < 1
        or cfg["batch_size"] < 1
        or cfg["epochs"] < 1
        or cfg.get("gradient_accumulation", 1) < 1
    ):
        raise ValueError("chunk, batch_size and epochs must be positive")
    return cfg


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
