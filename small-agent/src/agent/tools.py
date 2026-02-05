from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ToolCall:
    tool: str
    path: str | None = None
    command: str | None = None
    old_text: str | None = None
    new_text: str | None = None


@dataclass
class ToolResult:
    success: bool
    output: str


def parse_tool_call(text: str) -> ToolCall | None:
    """Extract tool call from <tool_call>...</tool_call> tags."""
    match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL)
    if not match:
        return None

    content = match.group(1).strip()
    lines = content.split('\n')
    first_line = lines[0].strip()

    # Parse tool name and args from first line
    parts = first_line.split(None, 1)
    tool = parts[0]
    args = parts[1] if len(parts) > 1 else None
    # Strip /workspace/ prefix if model uses absolute paths
    if args and args.startswith('/workspace/'):
        args = args[len('/workspace/'):]

    if tool == 'read':
        return ToolCall(tool='read', path=args)

    if tool == 'run':
        return ToolCall(tool='run', command=args)

    if tool == 'edit':
        # Parse edit block: path on first line, then <<<\n...\n>>>\n...\n===
        edit_match = re.search(r'<<<\n(.*?)\n>>>\n(.*?)\n===', content, re.DOTALL)
        if not edit_match:
            # Fallback: strip whitespace loosely
            edit_match = re.search(r'<<<\s*(.*?)\s*>>>\s*(.*?)\s*===', content, re.DOTALL)
        if not edit_match:
            return None

        old_text = edit_match.group(1)
        new_text = edit_match.group(2)
        return ToolCall(tool='edit', path=args, old_text=old_text, new_text=new_text)

    return None


def format_tool_result(result: ToolResult) -> str:
    """Wrap result in <tool_result> tags with success indicator."""
    prefix = "✓" if result.success else "✗"
    return f"<tool_result>\n{prefix} {result.output}\n</tool_result>"


def is_done(text: str) -> bool:
    """Check if agent signaled completion."""
    return '<done/>' in text
