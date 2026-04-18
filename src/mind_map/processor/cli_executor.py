"""Generic CLI command executor for memo ingestion.

Runs a shell command with a prompt appended, captures stdout, parses JSON,
and raises on any failure (no fallback chain).
"""
from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
from typing import Any


class CLIExecutionError(RuntimeError):
    """Raised when a CLI command fails, returns non-zero, or produces invalid JSON."""
    pass


_DEFAULT_EXTRACTION_TIMEOUT = 60.0
_FILTER_EXTRACTION_TIMEOUT = 60.0


# ---- Provider command templates ----

_OPENCLAW_TEMPLATE = "openclaw agent --agent {agent} --message"
_OLLAMA_TEMPLATE = "ollama run {model}"


def resolve_cli_template(
    *,
    openclaw: str | None = None,
    ollama: str | None = None,
    llm_cmd: str | None = None,
) -> str | None:
    """Resolve provider-specific convenience flags to a command template string.

    At most one of openclaw / ollama / llm_cmd may be set.
    Returns the resolved template string, or None to use the built-in defaults path.

    Resolution map:
        --openclaw AGENT  -> "openclaw agent --agent AGENT --message"
        --ollama MODEL   -> "ollama run MODEL"
        --llm-cmd "raw"  -> passed through as-is
        (all None)       -> None (use built-in MiniMax with fallback chain)

    Raises ValueError if more than one is set.
    """
    provided = [k for k, v in {"openclaw": openclaw, "ollama": ollama, "llm_cmd": llm_cmd}.items() if v is not None]

    if len(provided) > 1:
        raise ValueError(
            f"Options --openclaw, --ollama, and --llm-cmd are mutually exclusive. "
            f"Got: {', '.join('--' + p for p in provided)}"
        )

    if openclaw is not None:
        return _OPENCLAW_TEMPLATE.format(agent=openclaw)
    if ollama is not None:
        return _OLLAMA_TEMPLATE.format(model=ollama)
    if llm_cmd is not None:
        return llm_cmd

    return None  # built-in MiniMax with fallback chain


def run_cli_json(
    command_template: str,
    prompt: str,
    timeout: float = _DEFAULT_EXTRACTION_TIMEOUT,
) -> dict[str, Any]:
    """Run a CLI command with a prompt appended, return parsed JSON from stdout.

    The prompt is appended as a single argument to the command template.
    E.g. template="openclaw agent --agent minimax --message" + prompt="hello"
         → ["openclaw", "agent", "--agent", "minimax", "--message", "hello"]

    Raises CLIExecutionError on:
        - Command not found (non-zero return)
        - Non-zero return code
        - Empty stdout
        - Stdout that cannot be parsed as JSON
    """
    try:
        command_parts = shlex.split(command_template)
    except ValueError as e:
        raise CLIExecutionError(f"Invalid command template: {command_template}\nerror: {e}")

    if not command_parts:
        raise CLIExecutionError("Command template is empty")

    if shutil.which(command_parts[0]) is None:
        raise CLIExecutionError(f"Command not found: {command_parts[0]}")

    cmd = command_parts + [prompt]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise CLIExecutionError(f"Command timed out after {timeout}s: {command_template}")

    if result.returncode != 0:
        raise CLIExecutionError(
            f"Command failed (rc={result.returncode}): {command_template}\n"
            f"stderr: {result.stderr[:500]}"
        )

    content = (result.stdout or "").strip()

    # Strip Ollama "Thinking..." auxiliary output that precedes JSON.
    # Find the first line that looks like the start of a JSON object or array.
    lines = content.splitlines()
    json_line_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            json_line_idx = i
            break

    if json_line_idx != -1:
        content = "\n".join(lines[json_line_idx:])

    if not content:
        raise CLIExecutionError(
            f"Command returned empty stdout: {command_template}"
        )

    def _find_json_bounds(text: str) -> tuple[int, int] | None:
        """Find start/end offsets of the first balanced JSON object or array in text.

        Returns (start, end+1) offsets, or None if no balanced JSON found.
        """
        if not text:
            return None
        start_char = text[0]
        if start_char not in ("{", "["):
            return None
        start_brace = start_char
        end_brace = "}" if start_brace == "{" else "]"
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == start_brace:
                depth += 1
            elif ch == end_brace:
                depth -= 1
                if depth == 0:
                    return (0, i + 1)
        return None

    # Try direct JSON parse first
    try:
        return json.loads(content, strict=False)
    except json.JSONDecodeError:
        pass

    # Try stripping common markdown fences
    for fence in ("```json\n", "```json", "```\n", "```"):
        if content.startswith(fence):
            trimmed = content[len(fence):].strip()
            try:
                return json.loads(trimmed, strict=False)
            except json.JSONDecodeError:
                pass

    # Strip residual ANSI escape sequences and try again
    content_clean = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", content)
    try:
        return json.loads(content_clean, strict=False)
    except json.JSONDecodeError:
        pass

    # Find balanced JSON using brace matching
    bounds = _find_json_bounds(content)
    if bounds is not None:
        try:
            return json.loads(content[bounds[0]:bounds[1]], strict=False)
        except json.JSONDecodeError:
            pass

    raise CLIExecutionError(
        f"Command output is not valid JSON: {command_template}\n"
        f"(output begins with non-JSON content; first 80 chars: {content[:80]})"
    )


def run_filter_cli(command_template: str, prompt: str) -> dict[str, Any]:
    """Run filter CLI command. Same as run_cli_json but with filter-specific timeout."""
    return run_cli_json(command_template, prompt, timeout=_FILTER_EXTRACTION_TIMEOUT)


def run_extraction_cli(command_template: str, prompt: str) -> dict[str, Any]:
    """Run extraction CLI command. Same as run_cli_json but with extraction-specific timeout."""
    return run_cli_json(command_template, prompt, timeout=_DEFAULT_EXTRACTION_TIMEOUT)
