"""Generate SFT training data from cloned repositories."""
from __future__ import annotations

import argparse

from agent.data import generate_from_repos


def main():
    parser = argparse.ArgumentParser(description="Generate SFT data from repos")
    parser.add_argument("--repos", required=True, help="Directory containing cloned repos")
    parser.add_argument("--output", default="data/sft.jsonl", help="Output JSONL file")
    parser.add_argument("--max-per-repo", type=int, default=30, help="Max examples per repo")
    args = parser.parse_args()
    generate_from_repos(args.repos, args.output, args.max_per_repo)


if __name__ == "__main__":
    main()
