"""The one external seam: an LLM backend turns a request into a reply."""

import time
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
    """The provider's answer: its text, token usage as the provider reports it and the model that answered.

    ``params`` are the parameters the backend sent with the prompt (such as
    ``max_tokens``), and ``raw`` is the provider's reply as it came, before
    the text was taken out of it.
    """

    text: str
    usage: dict
    model_id: str
    params: dict = field(default_factory=dict)
    raw: str = ""


class ProviderQuotaExhausted(RuntimeError):
    """The provider refuses calls because the balance or quota is exhausted; its message says so in its words.

    Each backend recognises its own provider's refusal. A campaign stops on
    it cleanly, dropping the round in progress, and resumes once topped up.
    """


class ProviderReplyMalformed(RuntimeError):
    """The provider answered, without an error, something that is not a reply; its message says what it lacked.

    Such as a body of ``null`` (DeepSeek, after the blank lines it sends
    while a call waits), a reply without its text or its usage, or output
    that is not the JSON it should be. The SDKs do not retry it, as the
    answer came back as a success; a campaign asks again, a few times
    (``AskingAgain``), before raising it.
    """


class LLMBackend(Protocol):
    """Anything that completes a request."""

    def complete(self, request: Request) -> Reply: ...


#: Seconds waited before each call asked again after an answer that was not a reply: two calls again, then it raises.
ASK_AGAIN_WAITS = (5.0, 30.0)


class AskingAgain:
    """A backend that asks ``backend`` again when it answers something that is not a reply.

    A campaign calls every role's backend through one, its own backends and
    those passed in alike.

    ``ProviderReplyMalformed`` is retried after each wait of
    ``ASK_AGAIN_WAITS`` and raised once they are spent; every other error
    is raised at once.
    """

    def __init__(self, backend: LLMBackend):
        self.backend = backend

    def complete(self, request: Request) -> Reply:
        """Complete ``request`` with ``backend``, asking again after an answer that was not a reply."""
        for wait in ASK_AGAIN_WAITS:
            try:
                return self.backend.complete(request)
            except ProviderReplyMalformed:
                time.sleep(wait)
        return self.backend.complete(request)


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
