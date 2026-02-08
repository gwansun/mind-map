"""LLM initialization and management for Mind Map.

This module serves as the main entry point for LLM management, providing:
- Configuration loading utilities
- Imports from processing_llm (LLM-B) and reasoning_llm (LLM-A) modules

For backward compatibility, all major functions are re-exported from this module.
"""

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from rich.console import Console

# Import processing LLM functions (LLM-B)
from .processing_llm import (
    DEFAULT_PROCESSING_MODEL,
    RECOMMENDED_MODELS,
    check_ollama_available,
    check_ollama_installed,
    ensure_model_available,
    get_available_models,
    get_available_models_detailed,
    get_ollama_llm,
    get_processing_llm,
    get_selected_model,
    initialize_ollama,
    install_ollama,
    list_models,
    pull_model,
    select_model_interactive,
    set_processing_model,
    start_ollama_service,
)

# Import reasoning LLM functions (LLM-A)
from .reasoning_llm import (
    ClaudeCLILLM,
    check_anthropic_available,
    check_claude_cli_available,
    check_claude_cli_installed,
    check_gemini_available,
    check_openai_available,
    get_anthropic_llm,
    get_claude_cli_llm,
    get_gemini_llm,
    get_openai_llm,
    get_reasoning_llm,
)

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


def get_llm_status() -> dict[str, Any]:
    """Check the health and status of both Reasoning (LLM-A) and Processing (LLM-B) providers.

    Returns:
        Dictionary containing status information for both LLMs.
    """
    config = load_config()
    
    # Check Processing LLM (LLM-B)
    processing_model = config.get("processing_llm", {}).get("model", DEFAULT_PROCESSING_MODEL)
    processing_available = check_ollama_available()
    
    # Check Reasoning LLM (LLM-A)
    reasoning_config = config.get("reasoning_llm", {})
    reasoning_provider = reasoning_config.get("provider", "claude-cli")
    
    reasoning_status = "offline"
    if reasoning_provider == "claude-cli":
        if check_claude_cli_available():
            reasoning_status = "online"
    elif reasoning_provider in ["gemini", "google"]:
        if check_gemini_available():
            reasoning_status = "online"
    elif reasoning_provider == "anthropic":
        if check_anthropic_available():
            reasoning_status = "online"
    elif reasoning_provider == "openai":
        if check_openai_available():
            reasoning_status = "online"

    return {
        "processing_llm": {
            "model": processing_model,
            "status": "online" if processing_available else "offline",
            "provider": "ollama"
        },
        "reasoning_llm": {
            "provider": reasoning_provider,
            "status": reasoning_status
        }
    }


# Re-export all functions for backward compatibility
__all__ = [
    # Status
    "get_llm_status",
    # Configuration
    "load_config",
    # Processing LLM (LLM-B) - Ollama
    "DEFAULT_PROCESSING_MODEL",
    "RECOMMENDED_MODELS",
    "check_ollama_installed",
    "install_ollama",
    "start_ollama_service",
    "check_ollama_available",
    "get_available_models",
    "get_available_models_detailed",
    "pull_model",
    "ensure_model_available",
    "get_ollama_llm",
    "initialize_ollama",
    "get_processing_llm",
    # Model selection
    "get_selected_model",
    "set_processing_model",
    "select_model_interactive",
    "list_models",
    # Reasoning LLM (LLM-A) - Claude CLI, Gemini, Anthropic, OpenAI
    "ClaudeCLILLM",
    "check_claude_cli_installed",
    "check_claude_cli_available",
    "get_claude_cli_llm",
    "get_gemini_llm",
    "check_gemini_available",
    "get_anthropic_llm",
    "check_anthropic_available",
    "get_openai_llm",
    "check_openai_available",
    "get_reasoning_llm",
]
