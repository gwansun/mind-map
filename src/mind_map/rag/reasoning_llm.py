"""Reasoning LLM (LLM-A) management for final response generation.

This module handles Claude CLI, Gemini, Anthropic Claude, and OpenAI GPT setup for reasoning tasks.
Claude CLI is the default provider, leveraging Claude Pro subscription without API costs.
"""

import os
import re
import shutil
import subprocess
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from rich.console import Console

console = Console()


def _find_openclaw_cli() -> str | None:
    """Find the openclaw CLI binary."""
    return shutil.which("openclaw")


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


# ============== OpenClaw Agent LLM ==============


def _is_openclaw_error_output(text: str) -> bool:
    """Detect if OpenClaw output contains a runtime or error payload.
    
    Checks for common error patterns:
    - JSON error objects with 'error' or 'status: error'
    - Error messages starting with "Error:" or "RuntimeError"
    - Traceback patterns
    - Common error keywords
    """
    if not text:
        return True  # Empty output is an error
    
    text_stripped = text.strip()
    text_lower = text_stripped.lower()
    
    # Check for common error prefixes (including RuntimeError, Error, Failed, etc.)
    error_prefixes = ["error:", "error -", "runtime error", "failed:", "failure:", "runtimeerror"]
    for prefix in error_prefixes:
        if text_lower.startswith(prefix):
            return True
    
    # Also check for RuntimeError (camelCase) at start
    if text_stripped.startswith("RuntimeError"):
        return True
    
    # Check for error keywords
    error_keywords = ["runtime error", "model error", "invalid request", "authentication failed", 
                      "api error", "rate limit", "quota exceeded", "connection error"]
    for keyword in error_keywords:
        if keyword in text_lower:
            return True
    
    # Check for JSON error objects (check both 'error' key and 'status': 'error')
    if text_stripped.startswith('{') and 'error' in text_lower:
        try:
            import json
            data = json.loads(text_stripped)
            if isinstance(data, dict):
                # Check for 'error' key or 'status': 'error'
                if 'error' in data:
                    return True
                if data.get('status') == 'error':
                    return True
        except (json.JSONDecodeError, ValueError):
            pass
    
    # Check for standalone "status: error" pattern (without JSON)
    if 'status: error' in text_lower:
        return True
    
    # Check for traceback patterns
    if 'traceback' in text_lower and ('file "' in text_lower or 'line ' in text_lower):
        return True
    
    return False


class OpenClawAgentLLM(BaseChatModel):
    """LangChain-compatible wrapper for OpenClaw Agent runtime.

    Uses `openclaw agent --agent main --message` to generate responses,
    leveraging the configured OpenClaw agent stack.
    """

    model: str = Field(default="main", description="OpenClaw agent name (default: main)")
    timeout: int = Field(default=120, description="Timeout in seconds for agent calls")
    openclaw_path: str | None = Field(default=None, description="Path to openclaw CLI binary")

    def model_post_init(self, __context: Any) -> None:
        """Find openclaw CLI path after initialization."""
        if self.openclaw_path is None:
            self.openclaw_path = _find_openclaw_cli()

    @property
    def _llm_type(self) -> str:
        return "openclaw-agent"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response using OpenClaw Agent."""
        if not self.openclaw_path:
            raise RuntimeError(
                "OpenClaw CLI not found. Ensure OpenClaw is installed."
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
                [self.openclaw_path, "agent", "--agent", self.model, "--message", prompt],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if result.returncode != 0:
                raise RuntimeError(f"OpenClaw Agent error (exit {result.returncode}): {result.stderr}")

            response_text = result.stdout.strip()
            
            # Check for runtime/model error payloads even on success
            if _is_openclaw_error_output(response_text):
                raise RuntimeError(f"OpenClaw Agent runtime error: {response_text[:500]}")

            return ChatResult(
                generations=[
                    ChatGeneration(message=AIMessage(content=response_text))
                ]
            )

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"OpenClaw Agent timed out after {self.timeout}s")
        except FileNotFoundError:
            raise RuntimeError(
                "OpenClaw CLI not found. Ensure OpenClaw is installed."
            )


def check_openclaw_agent_installed() -> bool:
    """Fast check - verify CLI binary exists."""
    return _find_openclaw_cli() is not None


def check_openclaw_agent_available() -> bool:
    """Full check - verify CLI is installed and working.

    This runs a minimal test prompt to confirm the agent responds.
    Use check_openclaw_agent_installed() for fast preliminary checks.
    """
    openclaw_path = _find_openclaw_cli()
    if not openclaw_path:
        return False

    try:
        result = subprocess.run(
            [openclaw_path, "agent", "--agent", "main", "--message", "respond with only: OK"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


def get_openclaw_agent_llm(model: str = "main", timeout: int = 120) -> Any:
    """Get OpenClaw Agent LLM for response generation.

    Args:
        model: Agent name (default: main)
        timeout: Timeout in seconds

    Returns:
        OpenClawAgentLLM instance or None if not available
    """
    if not check_openclaw_agent_installed():
        console.print("[yellow]OpenClaw not installed.[/yellow]")
        return None

    if not check_openclaw_agent_available():
        console.print("[yellow]OpenClaw agent not available or not working.[/yellow]")
        return None

    return OpenClawAgentLLM(model=model, timeout=timeout)


# ============== MiniMax Direct LLM ==============


def _strip_think_tags(text: str) -> str:
    """Strip MiniMax M-series <think>...</think> reasoning blocks."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


class MiniMaxChatLLM(BaseChatModel):
    """LangChain-compatible wrapper for direct MiniMax API calls.

    Uses POST to https://api.minimax.io/v1/chat/completions with
    the MiniMax-M2.5 model. Replaces the deprecated OpenClawAgentLLM.
    """

    model: str = Field(default="MiniMax-M2.5", description="MiniMax model name")
    timeout: int = Field(default=120, description="Timeout in seconds for API calls")
    api_key: str = Field(
        default_factory=lambda: os.getenv("MINIMAX_API_KEY", ""),
        description="MiniMax API key (from MINIMAX_API_KEY env var)",
    )
    base_url: str = Field(
        default="https://api.minimax.io",
        description="MiniMax API base URL",
    )
    max_tokens: int = Field(default=124000, description="Maximum completion tokens")

    @property
    def _llm_type(self) -> str:
        return "minimax-direct"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response via direct MiniMax API call."""
        if not self.api_key:
            raise RuntimeError(
                "MINIMAX_API_KEY not set. Set it in your environment or .env file."
            )

        try:
            import requests
        except ImportError:
            raise RuntimeError("requests is required for MiniMax API calls")

        # Convert LangChain messages to OpenAI format
        openai_messages = []
        for msg in messages:
            if msg.type == "system":
                openai_messages.append({"role": "system", "content": str(msg.content)})
            elif msg.type == "human":
                openai_messages.append({"role": "user", "content": str(msg.content)})
            elif msg.type == "ai":
                openai_messages.append({"role": "assistant", "content": str(msg.content)})
            else:
                openai_messages.append({"role": "user", "content": str(msg.content)})

        payload = {
            "model": self.model,
            "messages": openai_messages,
            "max_tokens": self.max_tokens,
        }

        try:
            resp = requests.post(
                f"{self.base_url.rstrip('/')}/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.Timeout:
            raise RuntimeError(f"MiniMax API timed out after {self.timeout}s")
        except requests.RequestException as e:
            raise RuntimeError(f"MiniMax API request failed: {e}")

        body = resp.json()
        choices = body.get("choices", [])
        if not choices:
            raise RuntimeError(f"MiniMax API returned no choices: {body}")

        response_text = choices[0].get("message", {}).get("content", "")
        if not response_text:
            raise RuntimeError("MiniMax API returned empty content")

        # Strip <think> tags (M-series models wrap reasoning in them)
        response_text = _strip_think_tags(response_text)

        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=response_text))
            ]
        )


def check_minimax_api_available() -> bool:
    """Check if MiniMax API key is configured."""
    return bool(os.getenv("MINIMAX_API_KEY"))


def get_minimax_llm(model: str = "MiniMax-M2.5", timeout: int = 120) -> Any:
    """Get MiniMax Direct LLM for response generation.

    Args:
        model: MiniMax model name (default: MiniMax-M2.5)
        timeout: Timeout in seconds

    Returns:
        MiniMaxChatLLM instance or None if API key not configured
    """
    if not check_minimax_api_available():
        console.print("[yellow]MINIMAX_API_KEY not set. Cannot use MiniMax API.[/yellow]")
        return None

    api_key = os.getenv("MINIMAX_API_KEY", "")
    return MiniMaxChatLLM(
        model=model,
        timeout=timeout,
        api_key=api_key,
    )


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
    Defaults to OpenClaw Agent (uses configured OpenClaw agent stack).

    Fallback priority: OpenClaw Agent → Claude CLI → Gemini → Anthropic Claude → OpenAI GPT

    Returns:
        LangChain LLM instance or None if not available
    """
    from mind_map.core.config import load_config

    config = load_config()
    reasoning_config = config.get("reasoning_llm", {})

    provider = reasoning_config.get("provider", "minimax-direct")
    model = reasoning_config.get("model", "MiniMax-M2.5")
    temperature = reasoning_config.get("temperature", 0.7)
    timeout = reasoning_config.get("timeout", 120)

    # Try configured provider first
    if provider == "minimax-direct":
        llm = get_minimax_llm(model, timeout)
        if llm:
            console.print(f"[dim]Using MiniMax API ({model})[/dim]")
            return llm
        console.print("[yellow]MiniMax API not available, trying fallback...[/yellow]")
    elif provider == "openclaw-agent":
        llm = get_openclaw_agent_llm(model, timeout)
        if llm:
            console.print(f"[dim]Using OpenClaw Agent ({model})[/dim]")
            return llm
        console.print("[yellow]OpenClaw Agent not available, trying fallback...[/yellow]")
    elif provider == "claude-cli":
        llm = get_claude_cli_llm(model, timeout)
        if llm:
            console.print(f"[dim]Using Claude CLI ({model})[/dim]")
            return llm
        console.print("[yellow]Claude CLI not available, trying fallback...[/yellow]")
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

    # Fallback chain: Claude CLI → Gemini → Anthropic → OpenAI
    if provider != "claude-cli" and check_claude_cli_installed():
        llm = get_openclaw_agent_llm("main", timeout)
        if llm:
            console.print("[dim]Using OpenClaw Agent as fallback[/dim]")
            return llm

    if provider != "claude-cli" and check_claude_cli_installed():
        llm = get_claude_cli_llm("sonnet", 120)
        if llm:
            console.print("[dim]Using Claude CLI as fallback[/dim]")
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
