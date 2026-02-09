"""Processing LLM (LLM-B) management for filtering, extraction, and summarization.

This module handles processing LLM setup with multi-provider support.
Cloud APIs (Gemini, Anthropic, OpenAI) are preferred when available,
with Ollama as the local fallback. Provides flexible model selection
with user configuration options.
"""

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import ollama
from rich.console import Console
from rich.table import Table

console = Console()

# Default model for processing (filtering, extraction)
DEFAULT_PROCESSING_MODEL = "phi3.5"
# Recommended models for processing tasks (lightweight, fast)
RECOMMENDED_MODELS = ["phi3.5", "phi3", "llama3.2", "mistral", "gemma2:2b", "qwen2.5:3b"]

# Runtime model selection (can be changed by user)
_selected_model: str | None = None


def check_ollama_installed() -> bool:
    """Check if Ollama CLI is installed on the system."""
    return shutil.which("ollama") is not None


def install_ollama() -> bool:
    """Install Ollama based on the operating system.

    Returns:
        True if installation was successful or already installed, False otherwise
    """
    if check_ollama_installed():
        console.print("[green]Ollama is already installed[/green]")
        return True

    system = platform.system()
    console.print(f"[yellow]Ollama not found. Installing for {system}...[/yellow]")

    try:
        if system == "Darwin":  # macOS
            # Check if Homebrew is available
            if shutil.which("brew"):
                console.print("[dim]Installing Ollama via Homebrew...[/dim]")
                subprocess.run(["brew", "install", "ollama"], check=True)
                console.print("[green]Ollama installed successfully via Homebrew[/green]")
                return True
            else:
                console.print("[yellow]Homebrew not found. Please install Ollama manually:[/yellow]")
                console.print("[dim]Visit: https://ollama.ai/download[/dim]")
                return False

        elif system == "Linux":
            console.print("[dim]Installing Ollama via install script...[/dim]")
            subprocess.run(
                ["curl", "-fsSL", "https://ollama.ai/install.sh"],
                stdout=subprocess.PIPE,
                check=True
            )
            result = subprocess.run(
                ["sh", "-c", "curl -fsSL https://ollama.ai/install.sh | sh"],
                check=True,
                capture_output=True,
                text=True
            )
            console.print("[green]Ollama installed successfully[/green]")
            return True

        elif system == "Windows":
            console.print("[yellow]Automatic installation not supported on Windows[/yellow]")
            console.print("[dim]Please download and install Ollama from: https://ollama.ai/download[/dim]")
            return False

        else:
            console.print(f"[red]Unsupported operating system: {system}[/red]")
            console.print("[dim]Visit: https://ollama.ai/download[/dim]")
            return False

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Failed to install Ollama: {e}[/red]")
        console.print("[dim]Please install manually from: https://ollama.ai/download[/dim]")
        return False
    except Exception as e:
        console.print(f"[red]Unexpected error during installation: {e}[/red]")
        return False


def start_ollama_service() -> bool:
    """Attempt to start Ollama service.

    Returns:
        True if service started or already running, False otherwise
    """
    if check_ollama_available():
        return True

    console.print("[yellow]Ollama service not running. Attempting to start...[/yellow]")

    try:
        system = platform.system()

        if system == "Darwin":  # macOS
            # Try to start Ollama service via brew services or direct command
            try:
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )
                console.print("[dim]Waiting for Ollama to start...[/dim]")
                import time
                time.sleep(3)  # Give it a few seconds to start

                if check_ollama_available():
                    console.print("[green]Ollama service started successfully[/green]")
                    return True
            except Exception:
                pass

        console.print("[yellow]Could not start Ollama automatically[/yellow]")
        console.print("[dim]Please start Ollama manually with: ollama serve[/dim]")
        return False

    except Exception as e:
        console.print(f"[yellow]Failed to start Ollama service: {e}[/yellow]")
        console.print("[dim]Please start manually with: ollama serve[/dim]")
        return False


def check_ollama_available() -> bool:
    """Check if Ollama server is running."""
    try:
        ollama.list()
        return True
    except Exception:
        return False


def get_available_models() -> list[str]:
    """Get list of models available in Ollama."""
    try:
        response = ollama.list()
        # Handle both dict and object response formats
        models = response.get("models", []) if isinstance(response, dict) else response.models
        return [m.get("model", m.get("name", "")).split(":")[0] for m in models]
    except Exception:
        return []


def get_available_models_detailed() -> list[dict[str, Any]]:
    """Get detailed list of models available in Ollama.

    Returns:
        List of dicts with model name, size, and modified date
    """
    try:
        response = ollama.list()
        # Handle both dict and object response formats
        models = response.get("models", []) if isinstance(response, dict) else response.models
        return [
            {
                "name": m.get("model", m.get("name", "")),
                "size": m.get("size", 0),
                "modified": m.get("modified_at", ""),
            }
            for m in models
        ]
    except Exception:
        return []


def list_models(show_recommended: bool = True) -> None:
    """Display available Ollama models with recommendations.

    Args:
        show_recommended: Whether to highlight recommended models
    """
    if not check_ollama_available():
        console.print("[red]Ollama is not running. Start with: ollama serve[/red]")
        return

    models = get_available_models_detailed()

    table = Table(title="Available Ollama Models")
    table.add_column("Model", style="cyan")
    table.add_column("Size", style="green")
    table.add_column("Status", style="yellow")

    available_names = [m["name"].split(":")[0] for m in models]

    # Show installed models
    for model in models:
        name = model["name"]
        base_name = name.split(":")[0]
        size_gb = model["size"] / (1024 ** 3)
        status = "✓ Installed"
        if show_recommended and base_name in RECOMMENDED_MODELS:
            status += " (Recommended)"
        table.add_row(name, f"{size_gb:.1f} GB", status)

    # Show recommended models not installed
    if show_recommended:
        for rec_model in RECOMMENDED_MODELS:
            if rec_model not in available_names:
                table.add_row(rec_model, "-", "[dim]Not installed (Recommended)[/dim]")

    console.print(table)

    # Show current selection
    current = get_selected_model()
    console.print(f"\n[bold]Current processing model:[/bold] {current}")


def get_selected_model() -> str:
    """Get the currently selected processing model.

    Returns:
        The model name from (in order): runtime selection, config, or default
    """
    global _selected_model

    # 1. Runtime selection takes priority
    if _selected_model:
        return _selected_model

    # 2. Check config.yaml
    from .llm import load_config
    config = load_config()
    processing_config = config.get("processing_llm", {})
    config_model = processing_config.get("model")
    if config_model:
        return config_model

    # 3. Fall back to default
    return DEFAULT_PROCESSING_MODEL


def set_processing_model(model_name: str, persist: bool = False) -> bool:
    """Set the processing model to use.

    Args:
        model_name: Name of the Ollama model to use
        persist: If True, save to config.yaml for future sessions

    Returns:
        True if successful, False otherwise
    """
    global _selected_model

    # Set runtime selection
    _selected_model = model_name
    console.print(f"[green]Processing model set to: {model_name}[/green]")

    # Optionally persist to config
    if persist:
        try:
            import yaml
            config_path = Path("config.yaml")

            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f) or {}
            else:
                config = {}

            if "processing_llm" not in config:
                config["processing_llm"] = {}
            config["processing_llm"]["model"] = model_name

            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)

            console.print(f"[dim]Saved to config.yaml[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not persist to config: {e}[/yellow]")
            return False

    return True


def select_model_interactive() -> str | None:
    """Interactive model selection prompt.

    Returns:
        Selected model name or None if cancelled
    """
    if not check_ollama_available():
        console.print("[red]Ollama is not running. Start with: ollama serve[/red]")
        return None

    available = get_available_models()

    console.print("\n[bold cyan]Select a processing model:[/bold cyan]\n")

    # Build options list
    options = []

    # First, show installed recommended models
    for model in RECOMMENDED_MODELS:
        if model in available:
            options.append((model, "installed", "recommended"))

    # Then other installed models
    for model in available:
        if model not in RECOMMENDED_MODELS:
            options.append((model, "installed", ""))

    # Then not-installed recommended models
    for model in RECOMMENDED_MODELS:
        if model not in available:
            options.append((model, "not installed", "recommended"))

    # Display options
    for i, (model, status, rec) in enumerate(options, 1):
        rec_tag = " [green](Recommended)[/green]" if rec else ""
        status_tag = f" [dim]({status})[/dim]"
        console.print(f"  [{i}] {model}{rec_tag}{status_tag}")

    console.print(f"  [0] Cancel")
    console.print(f"  [c] Enter custom model name\n")

    # Get selection
    while True:
        try:
            choice = console.input("[bold]Enter choice: [/bold]").strip().lower()

            if choice == "0":
                return None
            elif choice == "c":
                custom = console.input("[bold]Enter model name: [/bold]").strip()
                if custom:
                    return custom
                continue
            else:
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return options[idx][0]
                console.print("[red]Invalid choice[/red]")
        except ValueError:
            console.print("[red]Please enter a number or 'c'[/red]")
        except KeyboardInterrupt:
            return None


def pull_model(model_name: str, show_progress: bool = True) -> bool:
    """Pull a model from Ollama registry.

    Args:
        model_name: Name of the model to pull
        show_progress: Whether to show download progress

    Returns:
        True if successful, False otherwise
    """
    try:
        if show_progress:
            console.print(f"[yellow]Pulling {model_name} model...[/yellow]")

        # Stream the pull progress
        for progress in ollama.pull(model_name, stream=True):
            if show_progress and "completed" in progress and "total" in progress:
                pct = (progress["completed"] / progress["total"]) * 100
                console.print(f"\r[dim]Progress: {pct:.1f}%[/dim]", end="")

        if show_progress:
            console.print(f"\n[green]{model_name} model ready[/green]")
        return True
    except Exception as e:
        if show_progress:
            console.print(f"[red]Failed to pull {model_name}: {e}[/red]")
        return False


def ensure_model_available(
    model_name: str | None = None,
    auto_pull: bool | None = None,
    interactive: bool = False,
) -> str | None:
    """Ensure a model is available, optionally pulling if necessary.

    Args:
        model_name: Preferred model name (uses selected model if None)
        auto_pull: Whether to auto-pull if not available (uses config if None)
        interactive: If True and model not available, prompt user for selection

    Returns:
        The available model name, or None if no model available
    """
    # Determine which model to use
    if model_name is None:
        model_name = get_selected_model()

    # Determine auto_pull behavior from config if not specified
    if auto_pull is None:
        from .llm import load_config
        config = load_config()
        processing_config = config.get("processing_llm", {})
        auto_pull = processing_config.get("auto_pull", False)

    available = get_available_models()

    # Check if preferred model is available
    if model_name in available:
        return model_name

    # Model not available - decide what to do
    console.print(f"[yellow]Model '{model_name}' is not installed.[/yellow]")

    if interactive:
        # Interactive mode: let user choose
        console.print("\n[bold]Options:[/bold]")
        console.print(f"  [1] Pull '{model_name}' now")
        console.print("  [2] Select a different model")
        console.print("  [3] Cancel")

        choice = console.input("\n[bold]Enter choice (1-3): [/bold]").strip()

        if choice == "1":
            if pull_model(model_name):
                return model_name
            console.print("[red]Failed to pull model[/red]")
            return None
        elif choice == "2":
            selected = select_model_interactive()
            if selected:
                return ensure_model_available(selected, auto_pull=True, interactive=False)
            return None
        else:
            return None

    elif auto_pull:
        # Auto-pull enabled: try to pull the model
        console.print(f"[dim]Auto-pulling {model_name}...[/dim]")
        if pull_model(model_name):
            return model_name

        # If pull fails, try recommended models that are already installed
        for fallback in RECOMMENDED_MODELS:
            if fallback in available:
                console.print(f"[yellow]Using available model: {fallback}[/yellow]")
                return fallback

        # Try to pull first recommended model
        for fallback in RECOMMENDED_MODELS:
            console.print(f"[dim]Trying to pull {fallback}...[/dim]")
            if pull_model(fallback):
                return fallback

        return None

    else:
        # No auto-pull: suggest available alternatives
        if available:
            console.print("\n[bold]Available models:[/bold]")
            for m in available[:5]:
                rec = " (Recommended)" if m in RECOMMENDED_MODELS else ""
                console.print(f"  • {m}{rec}")
            console.print("\n[dim]Set model with: mind-map model set <model_name>[/dim]")
            console.print("[dim]Or enable auto_pull in config.yaml[/dim]")
        else:
            console.print("[dim]No models installed. Pull one with: ollama pull <model_name>[/dim]")
            console.print(f"[dim]Recommended: ollama pull {RECOMMENDED_MODELS[0]}[/dim]")

        return None


def get_ollama_llm(
    model_name: str | None = None,
    auto_pull: bool | None = None,
    interactive: bool = False,
) -> Any:
    """Get a LangChain-compatible Ollama LLM.

    Args:
        model_name: Name of the model to use (uses selected model if None)
        auto_pull: Whether to auto-pull if model not available
        interactive: If True, prompt user for model selection if needed

    Returns:
        ChatOllama instance or None if not available
    """
    if not check_ollama_available():
        return None

    model = ensure_model_available(model_name, auto_pull=auto_pull, interactive=interactive)
    if model is None:
        return None

    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=0.1)
    except ImportError:
        # Fallback to langchain_community if langchain_ollama not available
        try:
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(model=model, temperature=0.1)
        except ImportError:
            console.print("[red]langchain-ollama not installed[/red]")
            return None


def initialize_ollama(
    model_name: str | None = None,
    auto_pull: bool | None = None,
    interactive: bool = False,
    quiet: bool = False,
) -> bool:
    """Initialize Ollama and ensure processing model is available.

    Args:
        model_name: Model to initialize (uses selected model if None)
        auto_pull: Whether to auto-pull if not available
        interactive: If True, prompt user for model selection
        quiet: Suppress output messages

    Returns:
        True if Ollama is ready, False otherwise
    """
    if not check_ollama_available():
        if not quiet:
            console.print("[yellow]Ollama not running. Using heuristic processing.[/yellow]")
            console.print("[dim]Start Ollama with: ollama serve[/dim]")
        return False

    model = ensure_model_available(model_name, auto_pull=auto_pull, interactive=interactive)
    if model is None:
        if not quiet:
            console.print("[yellow]No LLM model available. Using heuristic processing.[/yellow]")
        return False

    if not quiet:
        console.print(f"[green]Ollama ready with model: {model}[/green]")
    return True


def _try_cloud_processing_llm(
    provider: str = "auto",
    temperature: float = 0.1,
) -> tuple[Any, str] | tuple[None, None]:
    """Try to create a cloud-based processing LLM.

    Args:
        provider: "auto" tries all in order, or specify "gemini"/"anthropic"/"openai"
        temperature: Temperature for generation (low for processing tasks)

    Returns:
        Tuple of (LLM instance, provider name) or (None, None) if unavailable
    """
    providers_to_try: list[str] = []
    if provider == "auto":
        providers_to_try = ["gemini", "anthropic", "openai"]
    elif provider in ("gemini", "anthropic", "openai"):
        providers_to_try = [provider]
    else:
        return None, None

    for p in providers_to_try:
        if p == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                continue
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                gemini_llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    google_api_key=api_key,
                    temperature=temperature,
                )
                return gemini_llm, "gemini"
            except ImportError:
                console.print("[dim]langchain-google-genai not installed, skipping Gemini[/dim]")
                continue
            except Exception as e:
                console.print(f"[dim]Gemini init failed: {e}[/dim]")
                continue

        elif p == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                continue
            try:
                from langchain_anthropic import ChatAnthropic
                anthropic_llm = ChatAnthropic(
                    model="claude-haiku-4-5-20251001",
                    api_key=api_key,
                    temperature=temperature,
                )
                return anthropic_llm, "anthropic"
            except ImportError:
                console.print("[dim]langchain-anthropic not installed, skipping Anthropic[/dim]")
                continue
            except Exception as e:
                console.print(f"[dim]Anthropic init failed: {e}[/dim]")
                continue

        elif p == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                continue
            try:
                from langchain_openai import ChatOpenAI
                openai_llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    api_key=api_key,
                    temperature=temperature,
                )
                return openai_llm, "openai"
            except ImportError:
                console.print("[dim]langchain-openai not installed, skipping OpenAI[/dim]")
                continue
            except Exception as e:
                console.print(f"[dim]OpenAI init failed: {e}[/dim]")
                continue

    return None, None


def _get_ollama_processing_llm(
    model_name: str | None = None,
    auto_pull: bool | None = None,
    interactive: bool = False,
    temperature: float = 0.1,
) -> Any:
    """Get an Ollama-based processing LLM.

    Args:
        model_name: Specific Ollama model to use
        auto_pull: Whether to auto-pull model if not available (uses config if None)
        interactive: If True, prompt user for model selection if needed
        temperature: Temperature for generation

    Returns:
        ChatOllama instance or None if unavailable
    """
    from .llm import load_config

    config = load_config()
    processing_config = config.get("processing_llm", {})

    if auto_pull is None:
        auto_pull = processing_config.get("auto_pull", False)

    if not check_ollama_installed():
        console.print("[yellow]Ollama not installed.[/yellow]")
        console.print("[dim]Install from: https://ollama.ai/download[/dim]")
        return None

    if not check_ollama_available():
        console.print("[yellow]Ollama service not running. Attempting to start...[/yellow]")
        if not start_ollama_service():
            console.print("[red]Ollama service not available.[/red]")
            console.print("[dim]Start Ollama manually with: ollama serve[/dim]")
            return None

    available_model = ensure_model_available(
        model_name,
        auto_pull=auto_pull,
        interactive=interactive,
    )
    if available_model is None:
        console.print("[red]No Ollama model available. Processing LLM unavailable.[/red]")
        console.print("[dim]List models: mind-map model list[/dim]")
        console.print("[dim]Set model: mind-map model set <model_name>[/dim]")
        return None

    return get_ollama_llm(available_model, auto_pull=False)


def detect_processing_provider() -> tuple[str, str]:
    """Detect which processing provider would be used based on config + available keys.

    Returns:
        Tuple of (provider_name, model_name) for the active processing provider.
    """
    from .llm import load_config

    config = load_config()
    processing_config = config.get("processing_llm", {})
    provider = processing_config.get("provider", "auto")

    if provider == "ollama":
        model = processing_config.get("model", DEFAULT_PROCESSING_MODEL)
        return "ollama", model

    # For auto or specific cloud provider, check what's available
    if provider in ("auto", "gemini"):
        if os.getenv("GOOGLE_API_KEY"):
            return "gemini", "gemini-2.0-flash"
        if provider == "gemini":
            return "gemini", "gemini-2.0-flash"

    if provider in ("auto", "anthropic"):
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic", "claude-haiku-4-5-20251001"
        if provider == "anthropic":
            return "anthropic", "claude-haiku-4-5-20251001"

    if provider in ("auto", "openai"):
        if os.getenv("OPENAI_API_KEY"):
            return "openai", "gpt-4o-mini"
        if provider == "openai":
            return "openai", "gpt-4o-mini"

    # Fallback to ollama
    model = processing_config.get("model", DEFAULT_PROCESSING_MODEL)
    return "ollama", model


def get_processing_llm(
    model_name: str | None = None,
    auto_pull: bool | None = None,
    interactive: bool = False,
) -> Any:
    """Get LLM for processing tasks (LLM-B): filtering, extraction, summarization.

    Provider priority is determined by config.yaml processing_llm.provider:
    - "auto" (default): Cloud first (Gemini → Anthropic → OpenAI) → Ollama fallback
    - "gemini"/"anthropic"/"openai": Use specific cloud provider, fall back to Ollama
    - "ollama": Use only Ollama (original behavior)

    Args:
        model_name: Specific Ollama model to use (only applies to Ollama provider)
        auto_pull: Whether to auto-pull Ollama model if not available (uses config if None)
        interactive: If True, prompt user for model selection if needed

    Returns:
        LangChain LLM instance or None if setup failed
    """
    from .llm import load_config

    config = load_config()
    processing_config = config.get("processing_llm", {})
    provider = processing_config.get("provider", "auto")
    temperature = processing_config.get("temperature", 0.1)

    # If provider is not ollama-only, try cloud first
    if provider != "ollama":
        cloud_llm, cloud_provider = _try_cloud_processing_llm(
            provider=provider,
            temperature=temperature,
        )
        if cloud_llm:
            console.print(f"[dim]Using {cloud_provider} for processing[/dim]")
            return cloud_llm
        if provider not in ("auto",):
            # Specific cloud provider requested but unavailable — fall through to Ollama
            console.print(f"[yellow]{provider} not available, falling back to Ollama[/yellow]")

    # Ollama fallback (or provider == "ollama")
    return _get_ollama_processing_llm(
        model_name=model_name,
        auto_pull=auto_pull,
        interactive=interactive,
        temperature=temperature,
    )
