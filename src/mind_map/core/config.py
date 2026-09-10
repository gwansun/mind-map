"""Configuration loading for Mind Map."""

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from rich.console import Console

# Resolve project files from the PACKAGE location, never the process cwd.
#
# A cwd-relative lookup silently reads the WRONG config.yaml whenever the
# process runs from another directory: an MCP server spawned by Hermes runs
# with cwd=~/.hermes, so ``Path("config.yaml")`` opened Hermes's own config,
# found no ``reasoning_llm`` key, and every setting fell back to its default
# (which sent ``mind_map_ask`` to MiniMax instead of the configured route).
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)

console = Console()


def load_config() -> dict[str, Any]:
    """Load configuration from the project's config.yaml.

    The path is resolved from ``PROJECT_ROOT``, so the result does not depend
    on the process working directory.

    Returns:
        Configuration dictionary, or empty dict if file not found or invalid
    """
    if not CONFIG_PATH.exists():
        console.print(f"[yellow]{CONFIG_PATH} not found, using defaults[/yellow]")
        return {}

    try:
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        console.print(f"[yellow]Error loading config.yaml: {e}[/yellow]")
        return {}


CANONICAL_DATA_DIR = Path("/Users/gwansun/mind-map/data")
DEFAULT_DATA_DIR = CANONICAL_DATA_DIR


def get_data_dir(explicit: Path | None = None) -> Path:
    """Resolve the active data directory.

    Priority: explicit argument -> MIND_MAP_DATA_DIR env var -> DEFAULT_DATA_DIR
    """
    import os
    if explicit is not None:
        return explicit.expanduser().resolve()
    env_dir = os.getenv("MIND_MAP_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return DEFAULT_DATA_DIR.expanduser().resolve()
