"""Agent evaluation and rejection sampling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent.docker import create_container, destroy_container
from agent.eval import compute_metrics
from agent.harness import run_episode, Trajectory
from agent.tasks import load_tasks


def trajectory_to_dict(t: Trajectory) -> dict:
    """Convert trajectory to serializable dict."""
    return {
        "task_id": t.task_id,
        "success": t.success,
        "steps": [
            {
                "model_output": s.model_output,
                "tool_call": {"tool": s.tool_call.tool, "path": s.tool_call.path,
                              "command": s.tool_call.command} if s.tool_call else None,
                "tool_result": {"success": s.tool_result.success, "output": s.tool_result.output}
                    if s.tool_result else None,
            }
            for s in t.steps
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Run agent evaluation")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--adapter", default=None, help="LoRA adapter path")
    parser.add_argument("--tasks", required=True, help="Task directory or file")
    parser.add_argument("--output", default="results.jsonl", help="Output file")
    parser.add_argument("--reject-sample", action="store_true", help="Rejection sampling mode")
    parser.add_argument("--k", type=int, default=8, help="Samples per task for rejection sampling")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    args = parser.parse_args()

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)

    tasks_path = Path(args.tasks)
    tasks_base = tasks_path if tasks_path.is_dir() else tasks_path.parent
    tasks = load_tasks(args.tasks)
    trajectories = []
    k = args.k if args.reject_sample else 1
    temp = args.temperature if args.temperature > 0 else (0.7 if args.reject_sample else 0.0)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        for i in range(k):
            files = str((tasks_base / task.files_dir).resolve()) if task.files_dir else None
            container = create_container(task.docker_image, files)
            try:
                traj = run_episode(model, tokenizer, task, container, temperature=temp)
                trajectories.append(traj)
                with open(out, "a") as f:
                    f.write(json.dumps(trajectory_to_dict(traj)) + "\n")
                status = "✓" if traj.success else "✗"
                print(f"{status} {task.id} (run {i+1}/{k}): {len(traj.steps)} steps")
            finally:
                destroy_container(container)

    metrics = compute_metrics(trajectories)
    print(f"\n{metrics}")

    # In rejection sampling mode, also save successful trajectories as SFT data
    if args.reject_sample:
        sft_path = out.with_suffix(".sft.jsonl")
        from agent.harness import build_messages
        count = 0
        for traj in trajectories:
            if not traj.success:
                continue
            # Find the task to get description
            task = next(t for t in tasks if t.id == traj.task_id)
            messages = build_messages(task.description, traj.steps)
            with open(sft_path, "a") as f:
                f.write(json.dumps({"messages": messages}) + "\n")
            count += 1
        print(f"Saved {count} successful trajectories to {sft_path}")


if __name__ == "__main__":
    main()
