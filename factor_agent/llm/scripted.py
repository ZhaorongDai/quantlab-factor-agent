"""A scripted backend: replies read from a file, in order. For tests and offline smoke runs."""

import json
import threading
from pathlib import Path

from factor_agent.llm.base import Reply, Request


class ScriptedBackend:
    """Answer each request with the next reply of a fixed list.

    ``requests`` keeps every request received, in order; calls from several
    threads take the replies one at a time, in the order they arrive.

    Raises
    ------
    RuntimeError
        From ``complete`` when every reply has been used.
    """

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.requests: list[Request] = []
        self._taking = threading.Lock()

    #: The options a role's config may give besides ``backend`` and ``model``; ``replies`` is needed.
    OPTIONS = ("replies",)

    @classmethod
    def from_config(cls, spec: dict) -> "ScriptedBackend":
        """Build from ``{"replies": <path to a JSON list of reply texts>}``."""
        return cls(json.loads(Path(spec["replies"]).read_text()))

    def complete(self, request: Request) -> Reply:
        """Return the next scripted reply and keep ``request``."""
        with self._taking:
            if len(self.requests) >= len(self._replies):
                raise RuntimeError(f"ScriptedBackend: no reply left for request {len(self.requests) + 1}")
            text = self._replies[len(self.requests)]
            self.requests.append(request)
        return Reply(text=text, usage={}, model_id="scripted")
