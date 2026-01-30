from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_checkpoint_step(path: Path) -> int | None:
    match = re.search(r"adapter-step(\d+)\.pt$", path.name)
    if match:
        return int(match.group(1))
    return None


def find_latest_checkpoint(outputs_dir: Path) -> tuple[int, Path] | None:
    if not outputs_dir.exists():
        return None
    candidates = []
    for path in outputs_dir.glob("adapter-step*.pt"):
        step = parse_checkpoint_step(path)
        if step is not None:
            candidates.append((step, path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])
