"""OpenClaw plugin registration logic.

Registers mind-map as an external tool with an OpenClaw server instance.
"""

from typing import Any

import requests
from rich.console import Console

from mind_map.app.openclaw.tool_manifest import get_tool_manifest
from mind_map.core.config import load_config

console = Console()


def register_with_openclaw(
    mind_map_url: str = "http://localhost:8000",
) -> bool:
    """Register mind-map as a tool with the configured OpenClaw server.

    Reads OpenClaw connection details from config.yaml and POSTs
    the tool manifest to the OpenClaw tool registry endpoint.

    Args:
        mind_map_url: The URL where the mind-map API is accessible.

    Returns:
        True if registration succeeded, False otherwise.
    """
    config = load_config()
    openclaw_config = config.get("openclaw", {})

    if not openclaw_config.get("enabled", False):
        console.print("[yellow]OpenClaw integration is disabled in config.yaml[/yellow]")
        return False

    base_url = openclaw_config.get("base_url", "http://localhost:3000")
    api_key = openclaw_config.get("api_key", "")

    manifest = get_tool_manifest()
    registration_payload = {
        **manifest,
        "endpoint": mind_map_url,
        "auth": {"type": "none"},
    }

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.post(
            f"{base_url}/api/tools/register",
            json=registration_payload,
            headers=headers,
            timeout=15,
        )
        if resp.status_code in (200, 201):
            console.print("[green]Successfully registered with OpenClaw[/green]")
            return True
        else:
            console.print(
                f"[red]OpenClaw registration failed: {resp.status_code} {resp.text}[/red]"
            )
            return False
    except requests.ConnectionError:
        console.print(
            f"[red]Cannot connect to OpenClaw at {base_url}. Is it running?[/red]"
        )
        return False
    except Exception as e:
        console.print(f"[red]OpenClaw registration error: {e}[/red]")
        return False


def get_openclaw_config() -> dict[str, Any]:
    """Return the current OpenClaw configuration from config.yaml.

    Returns:
        Dictionary with openclaw config (enabled, base_url, api_key).
    """
    config = load_config()
    return config.get("openclaw", {"enabled": False, "base_url": "", "api_key": ""})
