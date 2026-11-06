"""The OpenAI-compatible backend: chat completions at any ``base_url`` (OpenAI, DeepSeek, DashScope,
Gemini's compatible endpoint, a local vLLM or Ollama), the key from the environment variable the config names."""

import openai

from factor_agent.llm._common import key_from_environment
from factor_agent.llm.base import ProviderQuotaExhausted, ProviderReplyMalformed, Reply, Request


class OpenAICompatibleBackend:
    """Complete a request with one chat-completions call.

    Parameters
    ----------
    model : str
        The model of every call.
    base_url : str
        The provider's endpoint, such as ``https://api.deepseek.com/v1``.
    api_key_env : str, optional
        The environment variable holding the key; None for a local server without one.
    max_tokens : int, optional
        The ``max_tokens`` of every call; the provider's default if None.
    max_completion_tokens : int, optional
        The ``max_completion_tokens`` of every call, in place of
        ``max_tokens``: OpenAI's reasoning models take only this one, and
        count the reasoning tokens in it.
    reasoning_effort : str, optional
        The ``reasoning_effort`` of every call (such as ``"low"``,
        ``"medium"``, ``"high"``), for a reasoning model; the provider's
        default if None.
    timeout_seconds : float
        Seconds a call may take.

    Raises
    ------
    ValueError
        If both ``max_tokens`` and ``max_completion_tokens`` are given.
    """

    def __init__(
        self, model: str, base_url: str, api_key_env: str | None = None,
        max_tokens: int | None = None, max_completion_tokens: int | None = None,
        reasoning_effort: str | None = None, timeout_seconds: float = 600.0,
    ):
        if max_tokens is not None and max_completion_tokens is not None:
            raise ValueError("openai_compatible: give max_tokens or max_completion_tokens, not both")
        self.model = model
        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort
        key = key_from_environment(api_key_env) if api_key_env else "none"
        self._client = openai.OpenAI(api_key=key, base_url=base_url, timeout=timeout_seconds)

    #: The options a role's config may give besides ``backend`` and ``model``; ``base_url`` is needed.
    OPTIONS = ("base_url", "api_key_env", "max_tokens", "max_completion_tokens", "reasoning_effort", "timeout_seconds")

    @classmethod
    def from_config(cls, spec: dict) -> "OpenAICompatibleBackend":
        """Build from a role's config: ``model``, ``base_url`` and optionally ``api_key_env``,
        ``max_tokens`` or ``max_completion_tokens``, ``reasoning_effort`` and ``timeout_seconds``."""
        options = {k: spec[k] for k in cls.OPTIONS if k in spec and k != "base_url"}
        return cls(spec["model"], spec["base_url"], **options)

    @property
    def params(self) -> dict:
        """The parameters every call sends with the prompt."""
        params = {
            "max_tokens": self.max_tokens, "max_completion_tokens": self.max_completion_tokens,
            "reasoning_effort": self.reasoning_effort,
        }
        return {k: v for k, v in params.items() if v is not None}

    def complete(self, request: Request) -> Reply:
        """Send ``request``'s prompt as one user message.

        Raises
        ------
        ProviderQuotaExhausted
            If the provider answers HTTP 402 (Payment Required): the balance or quota is exhausted.
        openai.APIError
            If the call fails otherwise, after the SDK's retries (which never retry a 402).
        ProviderReplyMalformed
            If the provider answers something that is not a reply: no body
            (``null``), no choice, no message or no token usage.
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": request.prompt}], **self.params,
            )
        except openai.APIStatusError as exc:
            if exc.status_code == 402:
                raise ProviderQuotaExhausted(f"openai_compatible: {self.model}: {exc.message}") from exc
            raise
        lacking = _lacking(response)
        if lacking:
            raise ProviderReplyMalformed(f"openai_compatible: {self.model} answered a reply without {lacking}")
        usage = response.usage
        return Reply(
            text=response.choices[0].message.content or "",
            usage={"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens},
            model_id=response.model, params=self.params, raw=response.to_json(indent=None),
        )


def _lacking(response) -> str | None:
    """What ``response`` lacks to be a reply (its body, a choice, the message, the usage), or None."""
    if response is None:
        return "a body"
    if not getattr(response, "choices", None):
        return "a choice"
    if getattr(response.choices[0], "message", None) is None:
        return "a message"
    if getattr(response, "usage", None) is None:
        return "token usage"
    return None
