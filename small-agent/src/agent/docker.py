from __future__ import annotations

import subprocess
from dataclasses import dataclass

from .tools import ToolResult


@dataclass
class Container:
    id: str
    workdir: str = "/workspace"


def create_container(image: str, files_dir: str | None = None) -> Container:
    result = subprocess.run(
        ["docker", "create", "-w", "/workspace", image, "sleep", "infinity"],
        capture_output=True, text=True, check=True,
    )
    container_id = result.stdout.strip()
    subprocess.run(["docker", "start", container_id], check=True, capture_output=True)
    if files_dir:
        subprocess.run(
            ["docker", "cp", f"{files_dir}/.", f"{container_id}:/workspace"],
            check=True, capture_output=True,
        )
    return Container(id=container_id)


def exec_in_container(container: Container, command: str, timeout: int = 30) -> tuple[int, str]:
    try:
        result = subprocess.run(
            ["docker", "exec", container.id, "sh", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return (result.returncode, result.stdout + result.stderr)
    except subprocess.TimeoutExpired:
        return (1, f"timeout after {timeout}s")


def read_file(container: Container, path: str) -> str:
    returncode, output = exec_in_container(container, f"cat {path}")
    if returncode != 0:
        raise RuntimeError(f"failed to read {path}: {output}")
    return output


def write_file(container: Container, path: str, content: str):
    subprocess.run(
        ["docker", "exec", "-i", container.id, "sh", "-c", f"cat > {path}"],
        input=content,
        text=True,
        check=True
    )


def edit_file(container: Container, path: str, old: str, new: str) -> ToolResult:
    try:
        content = read_file(container, path)
    except RuntimeError as e:
        return ToolResult(success=False, output=str(e))

    if old not in content:
        return ToolResult(success=False, output=f"text not found in {path}")

    new_content = content.replace(old, new, 1)

    write_file(container, path, new_content)
    return ToolResult(success=True, output=f"edited {path}")


def destroy_container(container: Container):
    subprocess.run(["docker", "rm", "-f", container.id], capture_output=True)
