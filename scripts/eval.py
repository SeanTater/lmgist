from __future__ import annotations

import argparse
from pathlib import Path

from etd.config import load_config
from etd.eval import evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    metrics = evaluate_model(cfg)
    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
