"""The one external seam: an LLM backend turns a request into a reply."""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Request:
    """One call: the role making it, the model, the prompt and the sampling parameters."""

    role: str
    model: str
    prompt: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Reply:
    """The provider's answer: its text, token usage, cost in US dollars and the model that answered.

    ``params`` are the parameters the backend sent with the prompt (such as
    ``max_tokens``), and ``raw`` is the provider's reply as it came, before
    the text was taken out of it.
    """

    text: str
    usage: dict
    cost: float
    model_id: str
    params: dict = field(default_factory=dict)
    raw: str = ""


class LLMBackend(Protocol):
    """Anything that completes a request.

    A backend may also price a call before it is made with
    ``estimate(input_tokens, output_tokens) -> float | None`` (None: it
    cannot); the cost estimate before a run uses it, and treats a backend
    without it as unknown.
    """

    def complete(self, request: Request) -> Reply: ...


def make_backend(spec: dict) -> LLMBackend:
    """Build the backend a role's config names in ``spec["backend"]``.

    Raises
    ------
    ValueError
        If the backend name is unknown, or the config holds a key: keys come
        only from environment variables (``api_key_env`` names one).
    """
    from factor_agent.llm.anthropic import AnthropicBackend
    from factor_agent.llm.claude_code import ClaudeCodeBackend
    from factor_agent.llm.openai_compatible import OpenAICompatibleBackend
    from factor_agent.llm.replay import ReplayBackend
    from factor_agent.llm.scripted import ScriptedBackend

    backends = {
        "scripted": ScriptedBackend.from_config, "replay": ReplayBackend.from_config,
        "claude_code": ClaudeCodeBackend.from_config, "anthropic": AnthropicBackend.from_config,
        "openai_compatible": OpenAICompatibleBackend.from_config,
    }
    name = spec.get("backend")
    if name not in backends:
        raise ValueError(f"unknown LLM backend {name!r}; known: {sorted(backends)}")
    secrets = [option for option in spec if _is_secret(option)]
    if secrets:
        raise ValueError(
            f"LLM backend {name!r}: {secrets} look like keys; a key is read only from an environment "
            f"variable (name it with api_key_env), never from the config"
        )
    return backends[name](spec)


def _is_secret(option: str) -> bool:
    """Whether a config option's name says it holds a credential."""
    name = option.lower()
    if name == "api_key_env":
        return False
    return name in {"key", "token", "secret", "password"} or name.endswith(("_key", "_token", "_secret", "_password"))
