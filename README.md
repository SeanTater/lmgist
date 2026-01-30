# Embedding-to-Text Decoding

This project explores how well modern LLMs can decode text from sentence embeddings using lightweight prefix adapters. We train small adapters that map embeddings into synthetic prefix tokens, then use a frozen decoder LLM to reconstruct the original text.

## Key Results

On MS-MARCO v2.1 queries with E5-base-v2 embeddings and Llama-3.2-1B-Instruct:

| Adapter | Best K | BLEU | Token F1 |
|---------|--------|------|----------|
| MLP (2-layer) | 1 | 44.6 | 0.75 |
| Linear | 32 | 38.7 | 0.72 |

- **MLP adapters** excel at maximal compression (K=1), with performance degrading as K increases
- **Linear adapters** improve with more prefix tokens, peaking at K=32
- **LoRA fine-tuning** requires freezing the adapter; joint training destroys performance
- **Cross-domain transfer** fails completely (MS-MARCO ↔ OpenWebText)
- **Embedding capacity matters**: MiniLM (384 dim) underperforms E5 (768 dim) by 14 BLEU points

## Report

A full NeurIPS-formatted report is available at `report/main.typ`. Compile with:
```
typst compile report/main.typ
```

## Quick Start

```bash
uv sync
uv run scripts/train.py --config configs/k1.yaml
uv run scripts/eval.py --config configs/k1.yaml
```

## Project Structure

```
configs/           # Experiment configurations
  base.yaml        # Primary MS-MARCO defaults
  k{1,2,4,8,16,32}.yaml  # MLP adapter K sweep
  linear/          # Linear adapter configs
  lora-*.yaml      # LoRA experiment configs
  embed-*.yaml     # Embedding model comparison
  variable-length.yaml   # Variable-length training
scripts/           # Entry points
  train.py         # Training script
  eval.py          # Evaluation script
  run_k_sweep.py   # Batch K sweep runner
src/etd/           # Core library
  config.py        # Configuration dataclasses
  train.py         # Training loop
  eval.py          # Evaluation with BLEU, F1, bootstrap CI
  models.py        # Adapter and decoder setup
  embedding.py     # Embedding computation
  datasets.py      # Dataset loading
report/            # Typst report with cetz charts
```

## Outputs

- Adapter checkpoint: `outputs/*/adapter-final.pt`
- Metrics report: `outputs/*/eval-report.json` (includes 95% bootstrap CI)
- Full eval results: `outputs/*/eval-results.parquet`

## Configuration Options

Key dataset options:
- `max_tokens`: Maximum sequence length (default: 512)
- `min_tokens`: If set, randomly truncates to [min_tokens, max_tokens] per batch for variable-length training

Key model options:
- `prefix_tokens`: Number of synthetic prefix tokens (K)
- `adapter_layers`: 2 for MLP, 0 for linear
- `adapter_hidden_dim`: MLP hidden dimension (default: 2048)

Key training options:
- `freeze_adapter`: Freeze adapter during LoRA training
- `adapter_checkpoint`: Resume from checkpoint or use as LoRA base

## Entry Points

- `scripts/train.py --config <config>` — Train adapter
- `scripts/eval.py --config <config>` — Evaluate with BLEU, F1, and bootstrap CI
- `scripts/run_k_sweep.py --config <config> --ks 1,2,4,8,16,32` — Run K sweep
- `scripts/precompute_embeddings.py --config <config> --split train` — Precompute embeddings
