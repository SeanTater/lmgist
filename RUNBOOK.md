# Stage A Full Runbook (MS-MARCO)

This runbook assumes you will execute commands one by one. It keeps all changes inside the venv and uses `uv` for reproducible installs.

## 0) Optional: start fresh
```
rm -rf .venv
```

## 1) Create venv + install deps
```
uv venv
uv lock
uv sync
```

## 2) Authenticate to Hugging Face (if needed)
```
.venv/bin/huggingface-cli login
```

## 3) Switch to full‑run limits
Edit `configs/base.yaml` and set:
```
dataset:
  train_limit: null
  validation_limit: 2000
  test_limit: 2000
```

## 4) Estimate embedding storage
```
uv run scripts/estimate_embeddings.py --config configs/base.yaml
```

## 5) Train Stage A (adapter‑only)
```
uv run scripts/train.py --config configs/base.yaml
```

Outputs:
- Adapter checkpoint at `outputs/adapter-final.pt`

## 6) Run evaluation
```
uv run scripts/eval.py --config configs/base.yaml
```

Outputs:
- Metrics report: `outputs/eval-report.json`
- Full results: `outputs/eval-results.parquet`
