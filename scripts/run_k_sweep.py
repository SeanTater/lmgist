from __future__ import annotations

import argparse
import re
from pathlib import Path

from etd.config import load_config
from etd.eval import evaluate_model
from etd.train import train


def _parse_ks(raw: str) -> list[int]:
    return [int(item) for item in raw.split(",") if item.strip()]


def _with_k(run_name: str, k: int) -> str:
    updated = re.sub(r"-k\d+$", f"-k{k}", run_name)
    if updated == run_name:
        return f"{run_name}-k{k}"
    return updated


def _run_for_k(config_path: Path, k: int) -> None:
    cfg = load_config(config_path)
    cfg.model.prefix_tokens = k
    cfg.project.run_name = _with_k(cfg.project.run_name, k)
    base_outputs = cfg.paths.outputs_dir
    cfg.paths.outputs_dir = base_outputs / f"k{k}"
    cfg.evaluation.adapter_checkpoint = str(cfg.paths.outputs_dir / "adapter-final.pt")
    print(f"Running K={k} -> {cfg.paths.outputs_dir}")
    train(cfg)
    evaluate_model(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--ks", default="1,2,4,8,32")
    args = parser.parse_args()

    ks = _parse_ks(args.ks)
    for k in ks:
        _run_for_k(args.config, k)


if __name__ == "__main__":
    main()
