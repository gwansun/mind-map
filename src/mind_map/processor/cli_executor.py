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
from dataclasses import dataclass
from typing import Any


class CLIExecutionError(RuntimeError):
    """Raised when a CLI command fails, returns non-zero, or produces invalid JSON."""
    pass


_DEFAULT_EXTRACTION_TIMEOUT = 60.0
_FILTER_EXTRACTION_TIMEOUT = 60.0
_DEFAULT_OPENCLAW_MESSAGE = "info"
_DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:11435/v1"
_LOCAL_MAX_COMPLETION_TOKENS = 1200


@dataclass(frozen=True)
class OpenClawTarget:
    """Explicit OpenClaw memo target resolved by the CLI."""

    message: str = _DEFAULT_OPENCLAW_MESSAGE
    agent: str | None = None


@dataclass(frozen=True)
class LocalTarget:
    """Explicit local OpenAI-compatible memo target resolved by the CLI."""

    model: str
    base_url: str = _DEFAULT_LOCAL_BASE_URL


MemoTarget = OpenClawTarget | LocalTarget


# ---- Provider command templates ----

_OPENCLAW_TEMPLATE = "openclaw agent --message"
_OPENCLAW_AGENT_TEMPLATE = "openclaw agent --agent {agent} --message"


def build_openclaw_command(target: OpenClawTarget) -> str:
    """Build the exact OpenClaw CLI command template for a resolved target."""
    if target.agent:
        return _OPENCLAW_AGENT_TEMPLATE.format(agent=target.agent)
    return _OPENCLAW_TEMPLATE


def resolve_local_model(*, model: str | None = None, base_url: str = _DEFAULT_LOCAL_BASE_URL) -> str:
    """Resolve a local OpenAI-compatible model.

    If model is provided, use it directly. Otherwise query /models and choose the
    first returned model id.
    """
    if model:
        return model

    models_url = f"{base_url.rstrip('/')}/models"

    try:
        import requests
    except ImportError as e:
        raise CLIExecutionError(f"requests is required for --local support: {e}") from e

    try:
        response = requests.get(models_url, timeout=_DEFAULT_EXTRACTION_TIMEOUT)
        response.raise_for_status()
    except Exception as e:
        raise CLIExecutionError(f"Failed to query local models at {models_url}: {e}") from e

    try:
        payload = response.json()
    except ValueError as e:
        raise CLIExecutionError(f"Local models endpoint did not return JSON: {models_url}") from e

    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise CLIExecutionError(f"No local models available at {models_url}")

    first = data[0]
    if not isinstance(first, dict) or not isinstance(first.get("id"), str) or not first["id"].strip():
        raise CLIExecutionError(f"Invalid local model entry returned by {models_url}")

    return first["id"].strip()


def build_local_command(target: LocalTarget) -> str:
    """Build a local OpenAI-compatible CLI command template using curl.

    The prompt will be appended as the last argument and injected into the JSON
    payload under messages[0].content.
    """
    return (
        "python3 -c "
        + shlex.quote(
            "import json, sys, urllib.request; "
            f"base={target.base_url.rstrip('/')!r}; "
            f"model={target.model!r}; "
            f"max_tokens={_LOCAL_MAX_COMPLETION_TOKENS!r}; "
            "prompt=sys.argv[1]; "
            "body=json.dumps({'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'response_format': {'type': 'json_object'}, 'max_tokens': max_tokens}).encode(); "
            "req=urllib.request.Request(base + '/chat/completions', data=body, headers={'Content-Type': 'application/json'}); "
            "resp=urllib.request.urlopen(req, timeout=60); "
            "sys.stdout.write(resp.read().decode())"
        )
    )


def build_cli_template(target: MemoTarget) -> str:
    """Build the exact CLI template for a resolved memo target."""
    if isinstance(target, OpenClawTarget):
        return build_openclaw_command(target)
    return build_local_command(target)


def run_cli_json(
    command_template: str,
    prompt: str,
    timeout: float = _DEFAULT_EXTRACTION_TIMEOUT,
) -> dict[str, Any]:
    """Run a CLI command with a prompt appended, return parsed JSON from stdout.

    The prompt is appended as a single argument to the command template.

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

    try:
        return json.loads(content, strict=False)
    except json.JSONDecodeError:
        pass

    for fence in ("```json\n", "```json", "```\n", "```"):
        if content.startswith(fence):
            trimmed = content[len(fence):].strip()
            try:
                return json.loads(trimmed, strict=False)
            except json.JSONDecodeError:
                pass

    content_clean = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", content)
    try:
        return json.loads(content_clean, strict=False)
    except json.JSONDecodeError:
        pass

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


def _extract_json_payload_from_chat_completion(data: dict[str, Any]) -> dict[str, Any] | None:
    """Unwrap an OpenAI-style chat completion envelope into the assistant JSON payload."""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    first = choices[0]
    if not isinstance(first, dict):
        return None

    message = first.get("message")
    if not isinstance(message, dict):
        return None

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return None

    text = content.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None

    try:
        payload = json.loads(text[start:end + 1], strict=False)
    except json.JSONDecodeError:
        return None

    return payload if isinstance(payload, dict) else None


def run_filter_cli(command_template: str, prompt: str) -> dict[str, Any]:
    """Run filter CLI command. Same as run_cli_json but with filter-specific timeout."""
    data = run_cli_json(command_template, prompt, timeout=_FILTER_EXTRACTION_TIMEOUT)
    payload = _extract_json_payload_from_chat_completion(data)
    return payload or data


def run_extraction_cli(command_template: str, prompt: str) -> dict[str, Any]:
    """Run extraction CLI command. Same as run_cli_json but with extraction-specific timeout."""
    data = run_cli_json(command_template, prompt, timeout=_DEFAULT_EXTRACTION_TIMEOUT)
    payload = _extract_json_payload_from_chat_completion(data)
    return payload or data
