"""Evaluation metrics for agent trajectories."""
from __future__ import annotations

from dataclasses import dataclass

from .harness import Trajectory


@dataclass
class EvalMetrics:
    total: int
    pass_rate: float
    first_try_rate: float
    avg_steps: float
    avg_tool_calls: float
    death_spiral_rate: float  # trajectories that hit max_steps without done

    def __str__(self):
        return (
            f"pass_rate={self.pass_rate:.1%} ({self.total} tasks) | "
            f"first_try={self.first_try_rate:.1%} | "
            f"avg_steps={self.avg_steps:.1f} | "
            f"avg_tools={self.avg_tool_calls:.1f} | "
            f"death_spiral={self.death_spiral_rate:.1%}"
        )


def compute_metrics(trajectories: list[Trajectory], max_steps: int = 20) -> EvalMetrics:
    if not trajectories:
        return EvalMetrics(0, 0, 0, 0, 0, 0)

    n = len(trajectories)
    passed = sum(t.success for t in trajectories)

    # First try = succeeded in ≤2 tool calls (read + edit)
    first_try = sum(
        t.success and sum(1 for s in t.steps if s.tool_call) <= 2
        for t in trajectories
    )

    total_steps = sum(len(t.steps) for t in trajectories)
    total_tool_calls = sum(
        sum(1 for s in t.steps if s.tool_call) for t in trajectories
    )

    # Death spiral: hit max steps and no <done/> signal
    death_spirals = sum(
        len(t.steps) >= max_steps and not any(
            "<done/>" in s.model_output for s in t.steps
        )
        for t in trajectories
    )

    return EvalMetrics(
        total=n,
        pass_rate=passed / n,
        first_try_rate=first_try / n,
        avg_steps=total_steps / n,
        avg_tool_calls=total_tool_calls / n,
        death_spiral_rate=death_spirals / n,
    )


def compare_models(results: dict[str, EvalMetrics]) -> str:
    """Format a comparison table of multiple model results."""
    header = f"{'Model':<30} {'Pass':>8} {'1st Try':>8} {'Steps':>8} {'Tools':>8} {'Spiral':>8}"
    lines = [header, "-" * len(header)]
    for name, m in results.items():
        lines.append(
            f"{name:<30} {m.pass_rate:>7.1%} {m.first_try_rate:>7.1%} "
            f"{m.avg_steps:>7.1f} {m.avg_tool_calls:>7.1f} {m.death_spiral_rate:>7.1%}"
        )
    return "\n".join(lines)
