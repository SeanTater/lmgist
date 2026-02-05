# Agent Instructions

- Keep `README.md` up to date whenever adding or changing any script, entry point, or config.
  Include the new/changed command, a short description, and any new flags or outputs.

## Python & Data Science

- Use `uv` for packaging and dependency management (not pip, poetry, or conda).
- Use `polars` for dataframe operations (not pandas).

## Running Experiments

- Use `pueue` for GPU jobs (the queue runs 1 job at a time in the default group).
- Queue jobs with: `pueue add --group default --label "name" -- uv run scripts/...`
- Check status with: `pueue status` (avoid `pueue log` - causes OOM on large histories).

## Key Scripts

- `scripts/train.py --config <path>`: Train an adapter.
- `scripts/eval.py --config <path>`: Evaluate an adapter.
- `scripts/run_k_sweep.py --config <path> --ks <list>`: Train+eval for multiple K values.

## Config Conventions

- Configs live in `configs/*.yaml`.
- K value is set in `model.prefix_tokens`.
- Outputs go to `paths.outputs_dir` (e.g., `outputs/k1/`, `outputs/linear/k1/`).
- Eval results: `eval-report.json` (metrics) and `eval-results.parquet` (per-example).

## Report

- Progress report is at `report/main.typ` (Typst format).
- Update with new results as experiments complete.
