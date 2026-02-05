"""SFT data generation from real code repositories."""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from .harness import SYSTEM_PROMPT
from .tools import format_tool_result, ToolResult


# --- Mutation types ---

def rename_variable(code: str, lang: str) -> tuple[str, str, str, str] | None:
    """Find a variable assignment and rename it. Returns (original, mutated, old_text, new_text) or None."""
    patterns = {
        "py": r'^(\s*)(\w+)\s*=\s*(.+)$',
        "js": r'^(\s*)(?:let|const|var)\s+(\w+)\s*=\s*(.+)$',
        "rs": r'^(\s*)let\s+(?:mut\s+)?(\w+)\s*=\s*(.+)$',
    }
    pat = patterns.get(lang)
    if not pat:
        return None

    matches = list(re.finditer(pat, code, re.MULTILINE))
    if not matches:
        return None

    m = random.choice(matches)
    var_name = m.group(2)
    # Skip common names and single-char vars
    if var_name in ("self", "cls", "i", "j", "k", "x", "y", "_") or len(var_name) < 2:
        return None

    new_name = var_name + "_renamed"
    old_text = var_name
    new_text = new_name
    mutated = code.replace(var_name, new_name)
    return (code, mutated, old_text, new_text)


def change_string_literal(code: str) -> tuple[str, str, str, str] | None:
    """Find a string literal and change it."""
    matches = list(re.finditer(r'"([^"]{3,30})"', code))
    if not matches:
        matches = list(re.finditer(r"'([^']{3,30})'", code))
    if not matches:
        return None

    m = random.choice(matches)
    old_str = m.group(0)
    inner = m.group(1)
    new_inner = inner + "_modified"
    new_str = old_str[0] + new_inner + old_str[-1]
    mutated = code[:m.start()] + new_str + code[m.end():]
    return (code, mutated, old_str, new_str)


def introduce_typo(code: str) -> tuple[str, str, str, str] | None:
    """Introduce a plausible typo (swap two adjacent chars in a function name)."""
    # Find function calls
    matches = list(re.finditer(r'\b(\w{4,})\s*\(', code))
    if not matches:
        return None

    m = random.choice(matches)
    name = m.group(1)
    if len(name) < 4:
        return None

    # Swap two adjacent chars
    idx = random.randint(1, len(name) - 2)
    typo = name[:idx] + name[idx+1] + name[idx] + name[idx+2:]
    if typo == name:
        return None

    mutated = code[:m.start()] + typo + code[m.start() + len(name):]
    return (mutated, code, typo, name)  # Note: mutated is the "broken" version, code is the fix


MUTATIONS = [rename_variable, change_string_literal]


# --- Conversation builders ---

def make_read_edit_conversation(
    file_path: str,
    description: str,
    original_code: str,
    old_text: str,
    new_text: str,
) -> dict:
    """Build a multi-turn conversation: read file, then edit it."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": description},
        # Step 1: agent reads the file
        {"role": "assistant", "content": f"I'll read the file first.\n\n<tool_call>\nread {file_path}\n</tool_call>"},
        {"role": "user", "content": format_tool_result(ToolResult(success=True, output=original_code))},
        # Step 2: agent edits the file
        {"role": "assistant", "content": f"I'll make the edit now.\n\n<tool_call>\nedit {file_path}\n<<<\n{old_text}\n>>>\n{new_text}\n===\n</tool_call>"},
        {"role": "user", "content": format_tool_result(ToolResult(success=True, output=f"edited {file_path}"))},
        # Step 3: agent signals done
        {"role": "assistant", "content": "The change has been applied.\n\n<done/>"},
    ]
    return {"messages": messages}


def make_direct_edit_conversation(
    file_path: str,
    description: str,
    old_text: str,
    new_text: str,
) -> dict:
    """Build a single-turn edit conversation (agent already knows what to change)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": description},
        {"role": "assistant", "content": f"<tool_call>\nedit {file_path}\n<<<\n{old_text}\n>>>\n{new_text}\n===\n</tool_call>"},
        {"role": "user", "content": format_tool_result(ToolResult(success=True, output=f"edited {file_path}"))},
        {"role": "assistant", "content": "<done/>"},
    ]
    return {"messages": messages}


def make_failure_recovery_conversation(
    file_path: str,
    description: str,
    original_code: str,
    wrong_old: str,
    old_text: str,
    new_text: str,
) -> dict:
    """Build a conversation with a failed edit attempt, then recovery via read + retry."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": description},
        # Failed attempt
        {"role": "assistant", "content": f"<tool_call>\nedit {file_path}\n<<<\n{wrong_old}\n>>>\n{new_text}\n===\n</tool_call>"},
        {"role": "user", "content": format_tool_result(ToolResult(success=False, output=f"text not found in {file_path}"))},
        # Recovery: read first
        {"role": "assistant", "content": f"Let me read the file to see the exact content.\n\n<tool_call>\nread {file_path}\n</tool_call>"},
        {"role": "user", "content": format_tool_result(ToolResult(success=True, output=original_code))},
        # Correct edit
        {"role": "assistant", "content": f"Now I can see the exact text.\n\n<tool_call>\nedit {file_path}\n<<<\n{old_text}\n>>>\n{new_text}\n===\n</tool_call>"},
        {"role": "user", "content": format_tool_result(ToolResult(success=True, output=f"edited {file_path}"))},
        {"role": "assistant", "content": "<done/>"},
    ]
    return {"messages": messages}


# --- Repo walking ---

LANG_EXTENSIONS = {".py": "py", ".js": "js", ".ts": "js", ".rs": "rs", ".go": "go"}


def walk_repo(repo_dir: Path, max_files: int = 200) -> list[tuple[str, str, str]]:
    """Walk repo, return list of (relative_path, content, lang)."""
    files = []
    for ext, lang in LANG_EXTENSIONS.items():
        for f in repo_dir.rglob(f"*{ext}"):
            if any(p in f.parts for p in ("node_modules", "vendor", ".git", "target", "__pycache__")):
                continue
            try:
                content = f.read_text(errors="ignore")
            except Exception:
                continue
            if 20 < len(content) < 5000:  # skip tiny and huge files
                files.append((str(f.relative_to(repo_dir)), content, lang))
    random.shuffle(files)
    return files[:max_files]


def generate_from_repos(repos_dir: str, output: str, max_per_repo: int = 30):
    """Generate SFT data by walking repos and creating mutations."""
    repos = Path(repos_dir)
    examples = []

    for repo in sorted(repos.iterdir()):
        if not repo.is_dir() or repo.name.startswith("."):
            continue

        files = walk_repo(repo)
        count = 0

        for file_path, content, lang in files:
            if count >= max_per_repo:
                break

            # Try rename mutation
            if lang in ("py", "js", "rs"):
                result = rename_variable(content, lang)
                if result:
                    original, mutated, old_text, new_text = result
                    desc = f"Rename the variable `{old_text}` to `{new_text}` in {file_path}"
                    examples.append(make_read_edit_conversation(file_path, desc, original, old_text, new_text))
                    count += 1

                    # 30% chance of failure recovery variant
                    if random.random() < 0.3:
                        wrong = old_text.upper()  # plausible wrong guess
                        examples.append(make_failure_recovery_conversation(
                            file_path, desc, original, wrong, old_text, new_text
                        ))
                        count += 1

            # Try string literal mutation
            result = change_string_literal(content)
            if result and count < max_per_repo:
                original, mutated, old_text, new_text = result
                desc = f"Change the string {old_text} to {new_text} in {file_path}"
                if random.random() < 0.5:
                    examples.append(make_read_edit_conversation(file_path, desc, original, old_text, new_text))
                else:
                    examples.append(make_direct_edit_conversation(file_path, desc, old_text, new_text))
                count += 1

    # Add run-tool examples (broader scenarios)
    run_scenarios = [
        ("Run the test suite and verify all tests pass.", "pytest -x", "5 passed"),
        ("Check what Python version is installed.", "python --version", "Python 3.11.9"),
        ("List all files in the project.", "ls -la", "total 24\ndrwxr-xr-x ..."),
        ("Install the project dependencies.", "pip install -e .", "Successfully installed ..."),
        ("Run the linter on the codebase.", "ruff check .", "All checks passed!"),
    ]
    for desc, cmd, output_text in run_scenarios:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": desc},
            {"role": "assistant", "content": f"<tool_call>\nrun {cmd}\n</tool_call>"},
            {"role": "user", "content": format_tool_result(ToolResult(success=True, output=output_text))},
            {"role": "assistant", "content": "<done/>"},
        ]
        examples.append({"messages": messages})

    random.shuffle(examples)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Generated {len(examples)} examples -> {out}")
