from __future__ import annotations

from dataclasses import dataclass

import torch

from .docker import Container, exec_in_container, read_file, edit_file
from .tasks import Task
from .tools import ToolCall, ToolResult, parse_tool_call, format_tool_result, is_done


SYSTEM_PROMPT = """You are a coding agent. You modify files in /workspace to complete tasks.

Tools (use inside <tool_call> tags):
- read <path>: Read file contents
- run <command>: Run shell command
- edit <path>: Replace text in file. Format:
  <<<
  exact text to find
  >>>
  replacement text
  ===

Output <done/> when finished.

Files are in /workspace. If unsure what exists, run `ls` first.

Example - fix typo in hello.py:
<tool_call>
read hello.py
</tool_call>

After seeing `prnt("hi")` in the file:
<tool_call>
edit hello.py
<<<
prnt("hi")
>>>
print("hi")
===
</tool_call>

<done/>"""


@dataclass
class Step:
    model_output: str
    tool_call: ToolCall | None
    tool_result: ToolResult | None


@dataclass
class Trajectory:
    task_id: str
    steps: list[Step]
    success: bool


def build_messages(task_description: str, steps: list[Step]) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_description}
    ]
    for step in steps:
        messages.append({"role": "assistant", "content": step.model_output})
        if step.tool_result is not None:
            messages.append({"role": "user", "content": format_tool_result(step.tool_result)})
        elif step.tool_call is None and not is_done(step.model_output):
            messages.append({"role": "user", "content": "Use a tool or output <done/>."})
    return messages


def generate(model, tokenizer, messages: list[dict], temperature: float = 0.0) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=1024,
            temperature=max(temperature, 0.01),
            do_sample=temperature > 0,
        )

    new_tokens = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def execute_tool_call(tc: ToolCall, container: Container) -> ToolResult:
    try:
        if tc.tool == "read":
            content = read_file(container, tc.path)
            return ToolResult(success=True, output=content)
        if tc.tool == "run":
            exit_code, output = exec_in_container(container, tc.command)
            return ToolResult(success=exit_code == 0, output=output)
        if tc.tool == "edit":
            return edit_file(container, tc.path, tc.old_text, tc.new_text)
        return ToolResult(success=False, output=f"unknown tool: {tc.tool}")
    except Exception as e:
        return ToolResult(success=False, output=str(e))


def run_episode(
    model,
    tokenizer,
    task: Task,
    container: Container,
    max_steps: int | None = None,
    temperature: float = 0.0
) -> Trajectory:
    for cmd in task.setup_commands:
        exec_in_container(container, cmd)

    steps = []
    max_steps = max_steps or task.max_steps

    for _ in range(max_steps):
        messages = build_messages(task.description, steps)
        model_output = generate(model, tokenizer, messages, temperature)

        if is_done(model_output):
            steps.append(Step(model_output=model_output, tool_call=None, tool_result=None))
            break

        tool_call = parse_tool_call(model_output)
        if tool_call is None:
            steps.append(Step(model_output=model_output, tool_call=None, tool_result=None))
            continue

        tool_result = execute_tool_call(tool_call, container)
        steps.append(Step(model_output=model_output, tool_call=tool_call, tool_result=tool_result))

    exit_code, _ = exec_in_container(container, task.test_command)
    success = exit_code == 0

    return Trajectory(task_id=task.id, steps=steps, success=success)
