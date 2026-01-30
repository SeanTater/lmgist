from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

from etd.config import load_config


def _parse_ks(raw: str) -> list[int]:
    return [int(item) for item in raw.split(",") if item.strip()]


def _with_k(run_name: str, k: int) -> str:
    updated = re.sub(r"-k\d+$", f"-k{k}", run_name)
    if updated == run_name:
        return f"{run_name}-k{k}"
    return updated


def _run_for_k(
    config_path: Path,
    k: int,
    *,
    lora_enabled: bool,
    lora_adapter_checkpoint: Path | None,
) -> None:
    from etd.eval import evaluate_model
    from etd.train import train

    cfg = load_config(config_path)
    base_eval_checkpoint = cfg.evaluation.adapter_checkpoint
    cfg.model.prefix_tokens = k
    cfg.project.run_name = _with_k(cfg.project.run_name, k)
    base_outputs = cfg.paths.outputs_dir
    if lora_enabled:
        cfg.lora.enabled = True
        cfg.project.run_name = f"{cfg.project.run_name}-lora"
        cfg.paths.outputs_dir = base_outputs / "lora" / f"k{k}"
        if lora_adapter_checkpoint is None and not base_eval_checkpoint:
            raise ValueError(
                "LoRA enabled but no adapter checkpoint provided and none in config."
            )
        cfg.training.adapter_checkpoint = str(
            lora_adapter_checkpoint or Path(base_eval_checkpoint)
        )
    else:
        cfg.paths.outputs_dir = base_outputs / f"k{k}"
    cfg.evaluation.adapter_checkpoint = str(cfg.paths.outputs_dir / "adapter-final.pt")
    print(f"Running K={k} -> {cfg.paths.outputs_dir}")
    train(cfg)
    evaluate_model(cfg)


def _enqueue_pueue(
    config_path: Path,
    k: int,
    group: str,
    label_prefix: str,
    *,
    lora_enabled: bool,
    lora_adapter_checkpoint: Path | None,
) -> None:
    if not shutil.which("pueue"):
        raise RuntimeError("pueue not found in PATH; install it or run without --pueue")

    label = f"{label_prefix}-k{k}"
    if lora_enabled:
        label = f"{label}-lora"
    extra_args: list[str] = []
    if lora_enabled:
        extra_args.append("--lora")
        if lora_adapter_checkpoint:
            extra_args.extend(["--lora-adapter-checkpoint", str(lora_adapter_checkpoint)])
    cmd = [
        "pueue",
        "add",
        "--group",
        group,
        "--label",
        label,
        "--",
        "uv",
        "run",
        "scripts/run_k_sweep.py",
        "--config",
        str(config_path),
        "--ks",
        str(k),
        *extra_args,
    ]
    print("Enqueue:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--ks", default="1,2,4,8,32")
    parser.add_argument(
        "--pueue",
        action="store_true",
        help="enqueue each k run via pueue instead of running locally",
    )
    parser.add_argument("--pueue-group", default="k-sweep")
    parser.add_argument("--pueue-label-prefix", default="k-sweep")
    parser.add_argument(
        "--lora",
        action="store_true",
        help="enable LoRA for all requested k values",
    )
    parser.add_argument(
        "--lora-adapter-checkpoint",
        type=Path,
        default=None,
        help="adapter checkpoint to load before LoRA training starts",
    )
    args = parser.parse_args()

    ks = _parse_ks(args.ks)
    if args.pueue:
        for k in ks:
            _enqueue_pueue(
                args.config,
                k,
                group=args.pueue_group,
                label_prefix=args.pueue_label_prefix,
                lora_enabled=args.lora,
                lora_adapter_checkpoint=args.lora_adapter_checkpoint,
            )
        return

    for k in ks:
        _run_for_k(
            args.config,
            k,
            lora_enabled=args.lora,
            lora_adapter_checkpoint=args.lora_adapter_checkpoint,
        )


if __name__ == "__main__":
    main()
