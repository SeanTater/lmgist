# Runbook

Operational guide for running experiments.

## Setup

```bash
# Install dependencies
uv sync

# Verify installation
uv run scripts/train.py --help
```

## Quick Start

```bash
# Train MLP adapter with K=1 prefix tokens
uv run scripts/train.py --config configs/k1.yaml

# Evaluate
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
  precompute_embeddings.py  # Precompute embeddings
src/etd/           # Core library
report/            # Typst report
```

## Entry Points

| Script | Purpose |
|--------|---------|
| `scripts/train.py --config <config>` | Train adapter |
| `scripts/eval.py --config <config>` | Evaluate with BLEU, F1, bootstrap CI |
| `scripts/run_k_sweep.py --config <config> --ks 1,2,4,8,16,32` | Run K sweep |
| `scripts/precompute_embeddings.py --config <config> --split train` | Precompute embeddings |

## Outputs

Each experiment writes to its configured `outputs_dir`:

- `adapter-final.pt` — Trained adapter checkpoint
- `adapter-step{N}.pt` — Intermediate checkpoints
- `eval-report.json` — Metrics with 95% bootstrap CI
- `eval-results.parquet` — Per-example predictions

## Configuration Reference

### Dataset Options

```yaml
dataset:
  name: ms_marco              # HuggingFace dataset name
  config: v2.1                # Dataset config/subset
  text_field: query           # Field containing text
  max_tokens: 512             # Maximum sequence length
  min_tokens: null            # If set, random truncation to [min, max]
  train_limit: null           # Limit training examples (null = all)
  validation_limit: 2000      # Limit validation examples
  shuffle_seed: 42            # Deterministic shuffling
```

### Model Options

```yaml
model:
  decoder_model: meta-llama/Llama-3.2-1B-Instruct
  prefix_tokens: 1            # K - number of synthetic prefix tokens
  adapter_hidden_dim: 2048    # MLP hidden dimension
  adapter_layers: 2           # 2 for MLP, 0 for linear
  dropout: 0.1
```

### Training Options

```yaml
training:
  epochs: 8
  batch_size: 8
  grad_accum_steps: 4
  learning_rate: 0.0002
  adapter_checkpoint: null    # Resume from checkpoint
  freeze_adapter: false       # Freeze adapter during LoRA training
```

### LoRA Options

```yaml
lora:
  enabled: false
  r: 8
  alpha: 16
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]
```

### Embedding Options

```yaml
embeddings:
  model: intfloat/e5-base-v2
  batch_size: 64
  precompute: auto            # auto|true|false
  max_precompute_gb: 250      # Storage limit for auto
```

## Common Workflows

### Run a K Sweep

```bash
# MLP adapter across K values
uv run scripts/run_k_sweep.py --config configs/base.yaml --ks 1,2,4,8,16,32

# Linear adapter
uv run scripts/run_k_sweep.py --config configs/linear.yaml --ks 1,2,4,8,16,32

# With pueue for background execution
uv run scripts/run_k_sweep.py --config configs/base.yaml --ks 1,2,4,8,16,32 --pueue
```

### LoRA Fine-tuning

LoRA requires a pre-trained adapter checkpoint and **must freeze the adapter**:

```bash
# First train base adapter
uv run scripts/train.py --config configs/k1.yaml

# Then LoRA with frozen adapter
uv run scripts/train.py --config configs/lora-frozen.yaml
```

### Compare Embedding Models

```bash
uv run scripts/train.py --config configs/embed-minilm.yaml
uv run scripts/eval.py --config configs/embed-minilm.yaml
```

### Precompute Embeddings

For large datasets, precompute embeddings to avoid recomputation:

```bash
uv run scripts/precompute_embeddings.py --config configs/base.yaml --split train
uv run scripts/precompute_embeddings.py --config configs/base.yaml --split validation
```

Supports `--resume` for interrupted runs.

### Variable-Length Training

Train with random truncation for robustness to varying input lengths:

```bash
uv run scripts/train.py --config configs/variable-length.yaml
```

## Evaluation Metrics

- **BLEU**: Corpus-level BLEU score (sacrebleu)
- **Token F1**: Word-level precision/recall F1
- **Bootstrap CI**: 95% confidence intervals via 1000 bootstrap resamples
- **Length buckets**: Metrics broken down by reference length

## Report

Compile the NeurIPS-formatted report:

```bash
typst compile report/main.typ
```

Requires [Typst](https://typst.app/) and downloads `bloated-neurips` and `cetz-plot` packages automatically.
