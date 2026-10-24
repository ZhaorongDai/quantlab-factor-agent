"""The OpenAI-compatible backend: chat completions at any ``base_url`` (OpenAI, DeepSeek, DashScope,
Gemini's compatible endpoint, a local vLLM or Ollama), the key from the environment variable the config names."""

import openai

from factor_agent.llm._common import Prices, key_from_environment
from factor_agent.llm.base import Reply, Request


class OpenAICompatibleBackend:
    """Complete a request with one chat-completions call.

    Parameters
    ----------
    model : str
        The model of every call.
    base_url : str
        The provider's endpoint, such as ``https://api.deepseek.com/v1``.
    prices : Prices
        What the model's tokens cost (zero for a local server).
    api_key_env : str, optional
        The environment variable holding the key; None for a local server without one.
    max_tokens : int, optional
        The ``max_tokens`` of every call; the provider's default if None.
    timeout_seconds : float
        Seconds a call may take.
    """

    def __init__(
        self, model: str, base_url: str, prices: Prices, api_key_env: str | None = None,
        max_tokens: int | None = None, timeout_seconds: float = 600.0,
    ):
        self.model = model
        self.prices = prices
        self.max_tokens = max_tokens
        key = key_from_environment(api_key_env) if api_key_env else "none"
        self._client = openai.OpenAI(api_key=key, base_url=base_url, timeout=timeout_seconds)

    @classmethod
    def from_config(cls, spec: dict) -> "OpenAICompatibleBackend":
        """Build from a role's config: ``model``, ``base_url``, the prices, and optionally ``api_key_env``, ``max_tokens``, ``timeout_seconds``."""
        options = {k: spec[k] for k in ("api_key_env", "max_tokens", "timeout_seconds") if k in spec}
        return cls(spec["model"], spec["base_url"], Prices.from_spec(spec, "openai_compatible"), **options)

    @property
    def params(self) -> dict:
        """The parameters every call sends with the prompt."""
        return {"max_tokens": self.max_tokens} if self.max_tokens else {}

    def estimate(self, input_tokens: int, output_tokens: int) -> float:
        """What a call with these token counts would cost."""
        return self.prices.cost(input_tokens, output_tokens)

    def complete(self, request: Request) -> Reply:
        """Send ``request``'s prompt as one user message; the cost is the usage at ``prices``.

        Raises
        ------
        openai.APIError
            If the call fails after the SDK's retries.
        RuntimeError
            If the reply reports no token usage, so its cost cannot be counted.
        """
        response = self._client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": request.prompt}], **self.params,
        )
        usage = response.usage
        if usage is None:
            raise RuntimeError(f"openai_compatible: {self.model} reported no token usage, so the budget cannot count it")
        input_tokens, output_tokens = usage.prompt_tokens, usage.completion_tokens
        return Reply(
            text=response.choices[0].message.content or "",
            usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
            cost=self.prices.cost(input_tokens, output_tokens),
            model_id=response.model, params=self.params, raw=response.to_json(indent=None),
        )
