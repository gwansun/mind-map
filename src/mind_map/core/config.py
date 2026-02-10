"""Configuration loading for Mind Map."""

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

console = Console()


def load_config() -> dict[str, Any]:
    """Load configuration from config.yaml.

    Returns:
        Configuration dictionary, or empty dict if file not found or invalid
    """
    config_path = Path("config.yaml")
    if not config_path.exists():
        console.print("[yellow]config.yaml not found, using defaults[/yellow]")
        return {}

    try:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        console.print(f"[yellow]Error loading config.yaml: {e}[/yellow]")
        return {}
