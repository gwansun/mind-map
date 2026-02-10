"""Reasoning LLM (LLM-A) management for final response generation.

This module handles Claude CLI, Antigravity-OAuth (Gemini), Gemini, Anthropic Claude,
and OpenAI GPT setup for reasoning tasks.
Claude CLI is the default provider, leveraging Claude Pro subscription without API costs.
"""

import os
import shutil
import subprocess
import time
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from rich.console import Console

console = Console()


def _find_claude_cli() -> str | None:
    """Find the claude CLI binary, checking common locations."""
    # Check if it's in PATH
    claude_path = shutil.which("claude")
    if claude_path:
        return claude_path

    # Check common nvm locations
    home = os.path.expanduser("~")
    nvm_paths = [
        os.path.join(home, ".nvm/versions/node"),
    ]

    for nvm_base in nvm_paths:
        if os.path.exists(nvm_base):
            # Find node versions and check for claude
            try:
                for version_dir in os.listdir(nvm_base):
                    claude_bin = os.path.join(nvm_base, version_dir, "bin", "claude")
                    if os.path.exists(claude_bin):
                        return claude_bin
            except OSError:
                continue

    # Check global npm bin
    npm_global = os.path.join(home, ".npm-global/bin/claude")
    if os.path.exists(npm_global):
        return npm_global

    return None


def _claude_cli_env() -> dict[str, str]:
    """Build subprocess env for claude CLI, stripping ANTHROPIC_API_KEY.

    When ANTHROPIC_API_KEY is set in the environment (e.g. loaded by dotenv
    for the Anthropic SDK fallback), the claude CLI picks it up and uses the
    API key instead of the local Pro subscription.  Remove it so the CLI
    authenticates via its own session.

    Also ensures nvm/node paths are in PATH for finding the claude binary.
    """
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)

    # Add nvm paths to PATH if not already present
    home = os.path.expanduser("~")
    nvm_base = os.path.join(home, ".nvm/versions/node")
    if os.path.exists(nvm_base):
        try:
            for version_dir in os.listdir(nvm_base):
                bin_path = os.path.join(nvm_base, version_dir, "bin")
                if os.path.exists(bin_path) and bin_path not in env.get("PATH", ""):
                    env["PATH"] = bin_path + ":" + env.get("PATH", "")
                    break  # Use first available node version
        except OSError:
            pass

    return env


# ============== Claude CLI LLM ==============


class ClaudeCLILLM(BaseChatModel):
    """LangChain-compatible wrapper for Claude CLI (Claude Code).

    Uses the locally installed `claude` CLI to generate responses,
    leveraging Claude Pro subscription without API costs.
    """

    model: str = Field(default="sonnet", description="Claude model alias (sonnet, opus, haiku)")
    timeout: int = Field(default=120, description="Timeout in seconds for CLI calls")
    claude_path: str | None = Field(default=None, description="Path to claude CLI binary")

    def model_post_init(self, __context: Any) -> None:
        """Find claude CLI path after initialization."""
        if self.claude_path is None:
            self.claude_path = _find_claude_cli()

    @property
    def _llm_type(self) -> str:
        return "claude-cli"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response using Claude CLI."""
        if not self.claude_path:
            raise RuntimeError(
                "Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
            )

        # Convert LangChain messages to prompt text
        prompt_parts = []
        for msg in messages:
            if msg.type == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.type == "human":
                prompt_parts.append(f"Human: {msg.content}")
            elif msg.type == "ai":
                prompt_parts.append(f"Assistant: {msg.content}")
            else:
                prompt_parts.append(str(msg.content))

        prompt = "\n\n".join(prompt_parts)

        try:
            result = subprocess.run(
                [self.claude_path, "-p", "--model", self.model],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=_claude_cli_env(),
            )

            if result.returncode != 0:
                raise RuntimeError(f"Claude CLI error: {result.stderr}")

            response_text = result.stdout.strip()

            return ChatResult(
                generations=[
                    ChatGeneration(message=AIMessage(content=response_text))
                ]
            )

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Claude CLI timed out after {self.timeout}s")
        except FileNotFoundError:
            raise RuntimeError(
                "Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
            )


def check_claude_cli_installed() -> bool:
    """Fast check - verify CLI binary exists."""
    return _find_claude_cli() is not None


def check_claude_cli_available() -> bool:
    """Full check - verify CLI is installed and authenticated.

    This runs a minimal test prompt to confirm authentication.
    Use check_claude_cli_installed() for fast preliminary checks.
    """
    claude_path = _find_claude_cli()
    if not claude_path:
        return False

    try:
        result = subprocess.run(
            [claude_path, "-p", "--model", "haiku"],
            input="respond with only: OK",
            capture_output=True,
            text=True,
            timeout=30,
            env=_claude_cli_env(),
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


def get_claude_cli_llm(model: str = "sonnet", timeout: int = 120) -> Any:
    """Get Claude CLI LLM for response generation.

    Args:
        model: Model alias (sonnet, opus, haiku)
        timeout: Timeout in seconds

    Returns:
        ClaudeCLILLM instance or None if not available
    """
    if not check_claude_cli_installed():
        console.print("[yellow]Claude CLI not installed.[/yellow]")
        console.print("[dim]Install with: npm install -g @anthropic-ai/claude-code[/dim]")
        return None

    if not check_claude_cli_available():
        console.print("[yellow]Claude CLI not authenticated or not working.[/yellow]")
        console.print("[dim]Run: claude login[/dim]")
        return None

    return ClaudeCLILLM(model=model, timeout=timeout)


# ============== Antigravity-OAuth (Gemini via Google Antigravity IDE) ==============

# Module-level token cache
_antigravity_token_cache: dict[str, Any] = {
    "access_token": None,
    "expires_at": 0.0,
    "refresh_token": None,
}

ANTIGRAVITY_TOKEN_URL = "https://oauth2.googleapis.com/token"
ANTIGRAVITY_SCOPE = "https://www.googleapis.com/auth/generative-language"


def _get_antigravity_token(client_id: str, client_secret: str) -> str | None:
    """Exchange Antigravity-OAuth credentials for an access token.

    Uses OAuth2 client credentials grant against Google's token endpoint.
    Caches the token in memory and refreshes on expiry.

    Args:
        client_id: Antigravity OAuth client ID
        client_secret: Antigravity OAuth client secret

    Returns:
        Access token string, or None if exchange fails
    """
    import requests

    # Return cached token if still valid (with 60s buffer)
    if (
        _antigravity_token_cache["access_token"]
        and time.time() < _antigravity_token_cache["expires_at"] - 60
    ):
        return _antigravity_token_cache["access_token"]

    # Try refresh token first
    if _antigravity_token_cache["refresh_token"]:
        try:
            resp = requests.post(
                ANTIGRAVITY_TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": _antigravity_token_cache["refresh_token"],
                },
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                _antigravity_token_cache["access_token"] = data["access_token"]
                _antigravity_token_cache["expires_at"] = time.time() + data.get(
                    "expires_in", 3600
                )
                if "refresh_token" in data:
                    _antigravity_token_cache["refresh_token"] = data["refresh_token"]
                return _antigravity_token_cache["access_token"]
        except Exception:
            pass  # Fall through to client credentials grant

    # Client credentials grant
    try:
        resp = requests.post(
            ANTIGRAVITY_TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
                "scope": ANTIGRAVITY_SCOPE,
            },
            timeout=15,
        )

        if resp.status_code != 200:
            console.print(
                f"[red]Antigravity-OAuth token exchange failed: {resp.status_code}[/red]"
            )
            return None

        data = resp.json()
        _antigravity_token_cache["access_token"] = data["access_token"]
        _antigravity_token_cache["expires_at"] = time.time() + data.get("expires_in", 3600)
        if "refresh_token" in data:
            _antigravity_token_cache["refresh_token"] = data["refresh_token"]

        return _antigravity_token_cache["access_token"]

    except Exception as e:
        console.print(f"[red]Antigravity-OAuth token exchange error: {e}[/red]")
        return None


def check_antigravity_available() -> bool:
    """Check if Antigravity-OAuth credentials are configured."""
    return bool(os.getenv("ANTIGRAVITY_CLIENT_ID") and os.getenv("ANTIGRAVITY_CLIENT_SECRET"))


def get_antigravity_llm(
    model: str = "gemini-3-flash", temperature: float = 0.7
) -> Any:
    """Get Gemini LLM authenticated via Antigravity-OAuth.

    Uses OAuth2 token exchange with Google's Antigravity IDE credentials
    to access Gemini models without a static GOOGLE_API_KEY.

    Args:
        model: Gemini model name (default: gemini-3-flash)
        temperature: Temperature for response generation

    Returns:
        ChatGoogleGenerativeAI instance or None if credentials missing/exchange fails
    """
    client_id = os.getenv("ANTIGRAVITY_CLIENT_ID")
    client_secret = os.getenv("ANTIGRAVITY_CLIENT_SECRET")
    if not client_id or not client_secret:
        console.print(
            "[yellow]ANTIGRAVITY_CLIENT_ID/SECRET not set. "
            "Cannot use Antigravity-OAuth.[/yellow]"
        )
        return None

    access_token = _get_antigravity_token(client_id, client_secret)
    if not access_token:
        return None

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=access_token,
            temperature=temperature,
        )
    except ImportError:
        console.print("[red]langchain-google-genai not installed[/red]")
        return None
    except Exception as e:
        console.print(f"[red]Failed to initialize Antigravity Gemini: {e}[/red]")
        return None


# ============== Cloud Provider LLMs ==============


def get_gemini_llm(model: str = "gemini-1.5-pro", temperature: float = 0.7) -> Any:
    """Get a LangChain-compatible Google Gemini LLM for response generation.

    Args:
        model: Gemini model name (default: gemini-1.5-pro)
        temperature: Temperature for response generation

    Returns:
        ChatGoogleGenerativeAI instance or None if API key not configured
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        console.print("[yellow]GOOGLE_API_KEY not set. Cannot use Gemini for response generation.[/yellow]")
        return None

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=temperature)
    except ImportError:
        console.print("[red]langchain-google-genai not installed[/red]")
        return None
    except Exception as e:
        console.print(f"[red]Failed to initialize Google Gemini: {e}[/red]")
        return None


def check_gemini_available() -> bool:
    """Check if Google API key is configured."""
    return bool(os.getenv("GOOGLE_API_KEY"))


def get_anthropic_llm(model: str = "claude-sonnet-4-5-20250929", temperature: float = 0.7) -> Any:
    """Get a LangChain-compatible Anthropic Claude LLM for response generation.

    Args:
        model: Anthropic model name (default: claude-sonnet-4-5-20250929)
        temperature: Temperature for response generation

    Returns:
        ChatAnthropic instance or None if API key not configured
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[yellow]ANTHROPIC_API_KEY not set. Cannot use Claude for response generation.[/yellow]")
        return None

    try:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, api_key=api_key, temperature=temperature)
    except ImportError:
        console.print("[red]langchain-anthropic not installed[/red]")
        return None
    except Exception as e:
        console.print(f"[red]Failed to initialize Anthropic Claude: {e}[/red]")
        return None


def check_anthropic_available() -> bool:
    """Check if Anthropic API key is configured."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def get_openai_llm(model: str = "gpt-4o", temperature: float = 0.7) -> Any:
    """Get a LangChain-compatible OpenAI LLM for response generation.

    Args:
        model: OpenAI model name (default: gpt-4o)
        temperature: Temperature for response generation

    Returns:
        ChatOpenAI instance or None if API key not configured
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[yellow]OPENAI_API_KEY not set. Cannot use OpenAI for response generation.[/yellow]")
        return None

    try:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, api_key=api_key, temperature=temperature)
    except ImportError:
        console.print("[red]langchain-openai not installed[/red]")
        return None
    except Exception as e:
        console.print(f"[red]Failed to initialize OpenAI: {e}[/red]")
        return None


def check_openai_available() -> bool:
    """Check if OpenAI API key is configured."""
    return bool(os.getenv("OPENAI_API_KEY"))


def get_reasoning_llm() -> Any:
    """Get LLM for reasoning tasks (LLM-A): final response generation.

    Returns LLM based on reasoning_llm configuration in config.yaml.
    Defaults to Claude CLI (uses Claude Pro subscription without API costs).

    Fallback priority:
        Configured provider → Claude CLI → Antigravity → Gemini → Anthropic → OpenAI

    Returns:
        LangChain LLM instance or None if not available
    """
    from mind_map.core.config import load_config

    config = load_config()
    reasoning_config = config.get("reasoning_llm", {})

    provider = reasoning_config.get("provider", "claude-cli")
    model = reasoning_config.get("model", "sonnet")
    temperature = reasoning_config.get("temperature", 0.7)
    timeout = reasoning_config.get("timeout", 120)

    # Try configured provider first
    if provider == "claude-cli":
        llm = get_claude_cli_llm(model, timeout)
        if llm:
            console.print(f"[dim]Using Claude CLI ({model})[/dim]")
            return llm
        console.print("[yellow]Claude CLI not available, trying fallback...[/yellow]")
    elif provider == "antigravity":
        llm = get_antigravity_llm(model, temperature)
        if llm:
            console.print(f"[dim]Using Antigravity-OAuth Gemini ({model})[/dim]")
            return llm
        console.print("[yellow]Antigravity-OAuth not available, trying fallback...[/yellow]")
    elif provider == "gemini" or provider == "google":
        llm = get_gemini_llm(model, temperature)
        if llm:
            return llm
        console.print("[yellow]Gemini not available, trying fallback...[/yellow]")
    elif provider == "anthropic":
        llm = get_anthropic_llm(model, temperature)
        if llm:
            return llm
        console.print("[yellow]Anthropic not available, trying fallback...[/yellow]")
    elif provider == "openai":
        llm = get_openai_llm(model, temperature)
        if llm:
            return llm
        console.print("[yellow]OpenAI not available, trying fallback...[/yellow]")
    else:
        console.print(f"[yellow]Unknown reasoning_llm provider: {provider}[/yellow]")

    # Fallback chain: Claude CLI -> Antigravity -> Gemini -> Anthropic -> OpenAI
    if provider != "claude-cli" and check_claude_cli_installed():
        llm = get_claude_cli_llm("sonnet", 120)
        if llm:
            console.print("[dim]Using Claude CLI as fallback[/dim]")
            return llm

    if provider != "antigravity" and check_antigravity_available():
        llm = get_antigravity_llm("gemini-3-flash", temperature)
        if llm:
            console.print("[dim]Using Antigravity-OAuth Gemini as fallback[/dim]")
            return llm

    if provider not in ["gemini", "google"] and check_gemini_available():
        llm = get_gemini_llm("gemini-1.5-pro", temperature)
        if llm:
            console.print("[dim]Using Google Gemini as fallback[/dim]")
            return llm

    if provider != "anthropic" and check_anthropic_available():
        llm = get_anthropic_llm("claude-sonnet-4-5-20250929", temperature)
        if llm:
            console.print("[dim]Using Anthropic Claude as fallback[/dim]")
            return llm

    if provider != "openai" and check_openai_available():
        llm = get_openai_llm("gpt-4o", temperature)
        if llm:
            console.print("[dim]Using OpenAI as fallback[/dim]")
            return llm

    console.print("[red]No reasoning LLM available[/red]")
    return None
