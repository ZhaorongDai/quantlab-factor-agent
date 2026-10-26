"""The Anthropic API backend: the Messages API through the ``anthropic`` SDK, its key from ``ANTHROPIC_API_KEY``."""

import anthropic

from factor_agent.llm._common import key_from_environment
from factor_agent.llm.base import ProviderQuotaExhausted, Reply, Request


class AnthropicBackend:
    """Complete a request with one Messages API call.

    Parameters
    ----------
    model : str
        The model of every call.
    max_tokens : int
        The ``max_tokens`` of every call.
    effort : str, optional
        The ``output_config.effort`` of every call; the model's default if None.
    base_url : str, optional
        Another endpoint than Anthropic's.
    timeout_seconds : float
        Seconds a call may take.
    """

    def __init__(
        self, model: str, max_tokens: int = 16000, effort: str | None = None,
        base_url: str | None = None, timeout_seconds: float = 600.0,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self._client = anthropic.Anthropic(
            api_key=key_from_environment("ANTHROPIC_API_KEY"), base_url=base_url, timeout=timeout_seconds,
        )

    @classmethod
    def from_config(cls, spec: dict) -> "AnthropicBackend":
        """Build from a role's config: ``model`` and optionally ``max_tokens``, ``effort``, ``base_url``, ``timeout_seconds``."""
        options = {k: spec[k] for k in ("max_tokens", "effort", "base_url", "timeout_seconds") if k in spec}
        return cls(spec["model"], **options)

    @property
    def params(self) -> dict:
        """The parameters every call sends with the prompt."""
        return {"max_tokens": self.max_tokens, **({"effort": self.effort} if self.effort else {})}

    def complete(self, request: Request) -> Reply:
        """Send ``request``'s prompt as one user message.

        Raises
        ------
        ProviderQuotaExhausted
            If Anthropic refuses the call for billing: an error of type
            ``billing_error`` (HTTP 402), or the request error saying the
            credit balance is too low.
        anthropic.APIError
            If the call fails otherwise, after the SDK's retries.
        """
        extra = {"output_config": {"effort": self.effort}} if self.effort else {}
        try:
            message = self._client.messages.create(
                model=self.model, max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": request.prompt}], **extra,
            )
        except anthropic.APIStatusError as exc:
            if _is_billing(exc):
                raise ProviderQuotaExhausted(f"anthropic: {self.model}: {exc.message}") from exc
            raise
        usage = message.usage
        input_tokens = usage.input_tokens + (usage.cache_creation_input_tokens or 0) + (usage.cache_read_input_tokens or 0)
        return Reply(
            text="".join(block.text for block in message.content if block.type == "text"),
            usage={"input_tokens": input_tokens, "output_tokens": usage.output_tokens},
            model_id=message.model, params=self.params, raw=message.to_json(indent=None),
        )


def _is_billing(exc: "anthropic.APIStatusError") -> bool:
    """Whether Anthropic refused the call for billing: a ``billing_error``, or a request error on the credit balance."""
    error = exc.body.get("error", {}) if isinstance(exc.body, dict) else {}
    kind, message = error.get("type"), str(error.get("message", ""))
    return kind == "billing_error" or (kind == "invalid_request_error" and "credit balance" in message.lower())
