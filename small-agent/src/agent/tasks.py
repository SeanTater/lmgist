from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Task:
    id: str
    description: str
    docker_image: str = "small-agent-base"
    files_dir: str | None = None
    setup_commands: list[str] = field(default_factory=list)
    test_command: str = ""
    max_steps: int = 20


def load_tasks(path: str) -> list[Task]:
    p = Path(path)

    if p.is_file():
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            return [Task(**data)]
        return [Task(**d) for d in data]

    tasks = []
    for task_file in sorted(p.glob("task_*.json")):
        data = json.loads(task_file.read_text())
        if isinstance(data, dict):
            tasks.append(Task(**data))
        else:
            tasks.extend(Task(**d) for d in data)

    return sorted(tasks, key=lambda t: t.id)
