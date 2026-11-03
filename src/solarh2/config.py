"""YAML konfigürasyonunu yükleyen yardımcılar."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Tekrarlanabilirlik için tüm rastgelelik kaynaklarını sabitler."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
