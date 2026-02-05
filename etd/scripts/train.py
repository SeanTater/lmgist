from __future__ import annotations

import argparse
from pathlib import Path

from etd.config import load_config
from etd.selection import load_selection_config, train_selection
from etd.train import train


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
        train_selection(cfg, selection_cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
