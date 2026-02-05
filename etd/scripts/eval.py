from __future__ import annotations

import argparse
from pathlib import Path

from etd.config import load_config
from etd.eval import evaluate_model
from etd.selection import evaluate_selection, load_selection_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--task",
        choices=("reconstruct", "select"),
        default="reconstruct",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.task == "select":
        selection_cfg = load_selection_config(args.config)
        metrics = evaluate_selection(cfg, selection_cfg)
    else:
        metrics = evaluate_model(cfg)
    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
