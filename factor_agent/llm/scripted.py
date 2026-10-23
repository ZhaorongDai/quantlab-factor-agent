"""A scripted backend: replies read from a file, in order. For tests and offline smoke runs."""

import json
from pathlib import Path

from factor_agent.llm.base import Reply, Request


class ScriptedBackend:
    """Answer each request with the next reply of a fixed list.

    ``requests`` keeps every request received, in order.

    Raises
    ------
    RuntimeError
        From ``complete`` when every reply has been used.
    """

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.requests: list[Request] = []

    @classmethod
    def from_config(cls, spec: dict) -> "ScriptedBackend":
        """Build from ``{"replies": <path to a JSON list of reply texts>}``."""
        return cls(json.loads(Path(spec["replies"]).read_text()))

    def complete(self, request: Request) -> Reply:
        """Return the next scripted reply and keep ``request``."""
        if len(self.requests) >= len(self._replies):
            raise RuntimeError(f"ScriptedBackend: no reply left for request {len(self.requests) + 1}")
        text = self._replies[len(self.requests)]
        self.requests.append(request)
        return Reply(text=text, usage={}, cost=0.0, model_id="scripted")
