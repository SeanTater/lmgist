from __future__ import annotations

import argparse
from pathlib import Path

from etd.config import load_config
from etd.datasets import load_splits, prepare_text
from etd.embedding import estimate_embedding_storage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    datasets = load_splits(cfg.dataset)
    train_ds = prepare_text(datasets.train, cfg.dataset)

    estimate = estimate_embedding_storage(len(train_ds))
    print(
        "Estimated embedding storage: "
        f"{estimate.total_gb:.2f} GB for {estimate.num_rows} rows "
        f"({estimate.dims} dims, {estimate.bytes_per_row} bytes/row)"
    )


if __name__ == "__main__":
    main()
