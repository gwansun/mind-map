"""LLM status checking for Mind Map."""

from typing import Any

from mind_map.core.config import load_config
from mind_map.processor.processing_llm import check_ollama_available, detect_processing_provider
from mind_map.rag.reasoning_llm import (
    check_antigravity_available,
    check_anthropic_available,
    check_claude_cli_available,
    check_gemini_available,
    check_openai_available,
)


def get_llm_status() -> dict[str, Any]:
    """Check the health and status of both Reasoning (LLM-A) and Processing (LLM-B) providers.

    Returns:
        Dictionary containing status information for both LLMs.
    """
    config = load_config()

    # Check Processing LLM (LLM-B) — detect actual provider
    proc_provider, proc_model = detect_processing_provider()
    if proc_provider == "ollama":
        processing_available = check_ollama_available()
    elif proc_provider == "gemini":
        processing_available = check_gemini_available()
    elif proc_provider == "anthropic":
        processing_available = check_anthropic_available()
    elif proc_provider == "openai":
        processing_available = check_openai_available()
    else:
        processing_available = False

    # Check Reasoning LLM (LLM-A)
    reasoning_config = config.get("reasoning_llm", {})
    reasoning_provider = reasoning_config.get("provider", "claude-cli")

    reasoning_status = "offline"
    if reasoning_provider == "claude-cli":
        if check_claude_cli_available():
            reasoning_status = "online"
    elif reasoning_provider == "antigravity":
        if check_antigravity_available():
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
            "model": proc_model,
            "status": "online" if processing_available else "offline",
            "provider": proc_provider,
        },
        "reasoning_llm": {
            "provider": reasoning_provider,
            "status": reasoning_status,
        },
    }
