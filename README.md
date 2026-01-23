# Embedding-to-Text Decoding

This project explores how well modern LLMs can decode text from e5-base-v2 embeddings. The current pipeline implements Stage A (adapter-only training) with an option to precompute embeddings based on storage estimates.

## Highlights
- uv-based, reproducible environment (`pyproject.toml`).
- Adapter-only prefix tuning for embeddings -> text reconstruction.
- Training with embedding precompute gate and progress bars.
- Evaluation with BLEU + token-F1 and full per-example outputs to Parquet.

## Quick start
```
uv venv
uv lock
uv sync
uv run scripts/estimate_embeddings.py --config configs/base.yaml
uv run scripts/train.py --config configs/base.yaml
uv run scripts/eval.py --config configs/base.yaml
```

## Outputs
- Adapter checkpoint: `outputs/adapter-final.pt`
- Metrics report: `outputs/eval-report.json`
- Full eval results: `outputs/eval-results.parquet`

## Notes
- The current default model is `ministral/Ministral-3B-Instruct`.
- Switch back to Llama-3.2-3B after access approval.
