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
    """The provider's answer: its text, token usage, cost in US dollars and the model that answered."""

    text: str
    usage: dict
    cost: float
    model_id: str


class LLMBackend(Protocol):
    """Anything that completes a request."""

    def complete(self, request: Request) -> Reply: ...


def make_backend(spec: dict) -> LLMBackend:
    """Build the backend a role's config names in ``spec["backend"]``.

    Raises
    ------
    ValueError
        If the backend name is unknown.
    """
    from factor_agent.llm.scripted import ScriptedBackend

    backends = {"scripted": ScriptedBackend.from_config}
    name = spec.get("backend")
    if name not in backends:
        raise ValueError(f"unknown LLM backend {name!r}; known: {sorted(backends)}")
    return backends[name](spec)
