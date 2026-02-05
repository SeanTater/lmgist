from __future__ import annotations

import argparse
from pathlib import Path

from etd.config import load_config
from etd.datasets import load_splits, prepare_text
from etd.embedding import load_embedding_model, precompute_embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="train",
        help="dataset split to precompute embeddings for",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="override device for embedding model (e.g., cpu, cuda)",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="resume from an existing embeddings file if progress exists",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="disable resume and overwrite existing embeddings",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    datasets = load_splits(cfg.dataset)
    if args.split == "train":
        dataset = datasets.train
    elif args.split == "validation":
        dataset = datasets.validation
    else:
        dataset = datasets.test
    dataset = prepare_text(dataset, cfg.dataset)

    device = args.device or cfg.hardware.device
    model = load_embedding_model(cfg.embeddings, device)
    model.compile()
    output_path = cfg.paths.embeddings_dir / f"{args.split}.npy"
    precompute_embeddings(dataset, model, cfg.embeddings, output_path, resume=args.resume)
    print(f"Saved embeddings to {output_path}")


if __name__ == "__main__":
    main()
