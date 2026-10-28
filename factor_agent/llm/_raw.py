"""What a provider's raw reply says beyond the reply's text and usage: cache hits, reasoning, why it stopped.

Every backend keeps the provider's reply as it came (``Reply.raw``): an
Anthropic message, an OpenAI-compatible chat completion (DeepSeek's among
them) or the Claude Code CLI's result. ``raw_facts`` reads the few facts
the events file and the frontend need from any of them, and nothing of
the reply is read anywhere else. A raw reply that is empty, not JSON or of
another shape gives no fact.
"""

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class RawFacts:
    """What a provider's raw reply reports; ``None`` for anything it does not.

    ``cache_hits`` are input tokens read from the provider's cache,
    ``reasoning_tokens`` output tokens the model spent reasoning, and
    ``reasoning`` the model's reasoning itself. ``truncated`` is true when
    the reply was cut off at the output limit, false when it stopped for
    another reason.
    """

    cache_hits: int | None = None
    reasoning_tokens: int | None = None
    truncated: bool | None = None
    reasoning: str | None = None


def raw_facts(raw: str) -> RawFacts:
    """The facts of the provider's raw reply ``raw`` (see the module's notes)."""
    try:
        value = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return RawFacts()
    if not isinstance(value, dict):
        return RawFacts()
    usage = value.get("usage") if isinstance(value.get("usage"), dict) else {}
    choices = value.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    stop = _first(value.get("stop_reason"), choice.get("finish_reason"))
    return RawFacts(
        cache_hits=_first(
            usage.get("cache_read_input_tokens"),
            _dig(usage, "prompt_tokens_details", "cached_tokens"),
            usage.get("prompt_cache_hit_tokens"),
        ),
        reasoning_tokens=_dig(usage, "completion_tokens_details", "reasoning_tokens"),
        truncated=None if stop is None else stop in ("max_tokens", "length"),
        reasoning=_reasoning(value, choice),
    )


def _reasoning(value: dict, choice: dict) -> str | None:
    """The model's reasoning: an OpenAI-compatible message's, or an Anthropic message's thinking blocks."""
    if choice:
        message = choice.get("message")
        text = (message.get("reasoning_content") or message.get("reasoning")) if isinstance(message, dict) else None
        return text if isinstance(text, str) and text else None
    blocks = value.get("content")
    if isinstance(blocks, list):
        thinking = [b.get("thinking") for b in blocks if isinstance(b, dict) and b.get("type") == "thinking"]
        return "\n\n".join(t for t in thinking if isinstance(t, str) and t) or None
    return None


def _first(*values):
    """The first of ``values`` that is not ``None``."""
    return next((v for v in values if v is not None), None)


def _dig(value, *keys):
    """``value[keys[0]][keys[1]]...``, ``None`` where a level is not a dict."""
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
    return value
