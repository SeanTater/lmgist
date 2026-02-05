"""Train LoRA SFT model."""
from __future__ import annotations

import argparse

from agent.train import train


def main():
    parser = argparse.ArgumentParser(description="Train LoRA SFT")
    parser.add_argument("--config", default="configs/train.yaml", help="Training config")
    parser.add_argument("--data", required=True, help="Training data JSONL")
    args = parser.parse_args()
    train(args.config, args.data)


if __name__ == "__main__":
    main()
