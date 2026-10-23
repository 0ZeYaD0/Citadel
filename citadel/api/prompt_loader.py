"""Load editable study-tool prompts from a plain-text catalog."""

from __future__ import annotations

import re
import threading
from pathlib import Path

from citadel.core.concurrency import DirectFileOperations, FileOperations

REQUIRED_TOOLS = frozenset(
    {"flashcards", "quiz", "summary", "key_terms", "explain_simple"}
)
_SECTION_PATTERN = re.compile(r"^\[([a-z][a-z0-9_]*)\]$")
_prompt_cache: dict[Path, tuple[int, int, dict[str, str]]] = {}
_prompt_cache_lock = threading.Lock()
_direct_file_operations = DirectFileOperations()


def load_study_prompts(
    path: Path,
    operations: FileOperations = _direct_file_operations,
) -> dict[str, str]:
    """Read and validate the editable study-tool prompt catalog."""
    if not path.is_file():
        raise FileNotFoundError(f"Study prompt file does not exist: {path}")

    resolved_path = path.resolve()
    file_stat = resolved_path.stat()
    modified_at = file_stat.st_mtime_ns
    file_size = file_stat.st_size
    with _prompt_cache_lock:
        cached = _prompt_cache.get(resolved_path)
        if cached and cached[0] == modified_at and cached[1] == file_size:
            return cached[2].copy()

    prompts: dict[str, list[str]] = {}
    current_tool: str | None = None
    for line_number, line in enumerate(
        operations.read_text(resolved_path, encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if stripped.startswith("[") or stripped.endswith("]"):
            match = _SECTION_PATTERN.fullmatch(stripped)
            if not match:
                raise ValueError(f"Invalid prompt section at line {line_number}.")
            current_tool = match.group(1)
            if current_tool in prompts:
                raise ValueError(f"Duplicate prompt section: {current_tool}.")
            prompts[current_tool] = []
        elif current_tool is not None:
            prompts[current_tool].append(line)
        elif stripped:
            raise ValueError(
                f"Prompt text must follow a section header at line {line_number}."
            )

    loaded = {
        tool: "\n".join(lines).strip()
        for tool, lines in prompts.items()
        if "\n".join(lines).strip()
    }
    missing = REQUIRED_TOOLS - loaded.keys()
    if missing:
        missing_tools = ", ".join(sorted(missing))
        raise ValueError(f"Missing study prompt sections: {missing_tools}.")
    with _prompt_cache_lock:
        _prompt_cache[resolved_path] = (modified_at, file_size, loaded)
    return loaded.copy()
