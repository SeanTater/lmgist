#!/usr/bin/env python3
"""Aggregate length-bucketed eval results across experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def load_eval_report(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def collect_experiments(outputs_root: Path) -> list[dict]:
    """Collect all experiments with eval-report.json."""
    experiments = []

    # Check direct outputs directories
    for outputs_dir in outputs_root.parent.glob("outputs*"):
        report_path = outputs_dir / "eval-report.json"
        if report_path.exists():
            report = load_eval_report(report_path)
            if report and "length_buckets" in report:
                experiments.append({
                    "name": outputs_dir.name,
                    "path": outputs_dir,
                    "report": report,
                })

    # Check subdirectories in outputs/
    for subdir in outputs_root.iterdir():
        if subdir.is_dir():
            report_path = subdir / "eval-report.json"
            if report_path.exists():
                report = load_eval_report(report_path)
                if report and "length_buckets" in report:
                    experiments.append({
                        "name": subdir.name,
                        "path": subdir,
                        "report": report,
                    })

    return experiments


def build_comparison_table(experiments: list[dict]) -> pl.DataFrame:
    """Build a table comparing length buckets across experiments."""
    rows = []
    for exp in experiments:
        report = exp["report"]
        for i, bucket in enumerate(report.get("length_buckets", [])):
            if bucket.get("bleu") is None:
                continue
            rows.append({
                "experiment": exp["name"],
                "bucket": i,
                "min_tokens": bucket["min_tokens"],
                "max_tokens": bucket["max_tokens"],
                "avg_tokens": bucket["avg_ref_tokens"],
                "bleu": bucket["bleu"],
                "token_f1": bucket["token_f1"],
                "samples": bucket["samples"],
            })

    return pl.DataFrame(rows)


def summarize_by_length(df: pl.DataFrame) -> pl.DataFrame:
    """Summarize metrics by avg_tokens across experiments."""
    return df.group_by("experiment").agg([
        pl.col("avg_tokens").mean().alias("mean_ref_tokens"),
        pl.col("bleu").mean().alias("mean_bleu"),
        pl.col("token_f1").mean().alias("mean_token_f1"),
        pl.len().alias("n_buckets"),
    ]).sort("experiment")


def export_for_typst(df: pl.DataFrame, output_path: Path) -> None:
    """Export data in a format suitable for Typst charts."""
    # Group by dataset type (short/long) based on avg_tokens
    short_text = df.filter(pl.col("avg_tokens") < 50)
    long_text = df.filter(pl.col("avg_tokens") >= 50)

    output = {"short_text": {}, "long_text": {}}

    for exp_name in short_text["experiment"].unique().to_list():
        exp_df = short_text.filter(pl.col("experiment") == exp_name).sort("bucket")
        output["short_text"][exp_name] = [
            {"avg_tokens": row["avg_tokens"], "bleu": row["bleu"], "token_f1": row["token_f1"]}
            for row in exp_df.iter_rows(named=True)
        ]

    for exp_name in long_text["experiment"].unique().to_list():
        exp_df = long_text.filter(pl.col("experiment") == exp_name).sort("bucket")
        output["long_text"][exp_name] = [
            {"avg_tokens": row["avg_tokens"], "bleu": row["bleu"], "token_f1": row["token_f1"]}
            for row in exp_df.iter_rows(named=True)
        ]

    with output_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"Exported chart data to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze length buckets across experiments")
    parser.add_argument("--outputs", type=Path, default=Path("outputs"), help="Outputs directory")
    parser.add_argument("--format", choices=["table", "csv", "json"], default="table")
    parser.add_argument("--detail", action="store_true", help="Show per-bucket detail")
    parser.add_argument("--export", type=Path, help="Export data for Typst charts")
    args = parser.parse_args()

    experiments = collect_experiments(args.outputs)
    if not experiments:
        print("No experiments found with length bucket data.")
        return

    print(f"Found {len(experiments)} experiments with length bucket data:\n")
    for exp in sorted(experiments, key=lambda x: x["name"]):
        report = exp["report"]
        buckets = report.get("length_buckets", [])
        print(f"  {exp['name']}: {len(buckets)} buckets, overall BLEU={report.get('bleu', 0):.2f}")

    df = build_comparison_table(experiments)

    if args.export:
        export_for_typst(df, args.export)

    if args.detail:
        print("\n--- Per-Bucket Detail ---\n")
        if args.format == "csv":
            print(df.write_csv())
        elif args.format == "json":
            print(df.write_json())
        else:
            print(df.sort(["experiment", "bucket"]))

    print("\n--- Summary by Experiment ---\n")
    summary = summarize_by_length(df)
    print(summary)

    # Print length vs BLEU trend for each experiment
    print("\n--- BLEU by Length Bucket ---\n")
    for exp_name in sorted(df["experiment"].unique().to_list()):
        exp_df = df.filter(pl.col("experiment") == exp_name).sort("bucket")
        bleu_trend = [f"{row['bleu']:.1f}" for row in exp_df.iter_rows(named=True)]
        tokens_range = f"{exp_df['min_tokens'].min()}-{exp_df['max_tokens'].max()}"
        print(f"{exp_name} ({tokens_range} tokens): {' -> '.join(bleu_trend)}")


if __name__ == "__main__":
    main()
