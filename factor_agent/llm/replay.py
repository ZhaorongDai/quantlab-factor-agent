"""The replay backend: a recorded run's replies served back, so a campaign reproduces without a provider."""

from collections import defaultdict, deque

from factor_agent.llm.base import Reply, Request
from factor_agent.llm.log import LLMLog


class ReplayExhausted(RuntimeError):
    """The recorded run holds no reply for a request: it ran out of calls, or the run asked something else."""


class ReplayBackend:
    """Answer each request with the recorded reply to the same role, model and prompt.

    A request asked several times gets its recorded replies in order.

    Raises
    ------
    ValueError
        If the log holds no call (a wrong path, or a run that made none).
    """

    def __init__(self, log: LLMLog):
        self.log = log
        self._replies: dict[tuple[str, str, str], deque] = defaultdict(deque)
        calls = log.calls()
        if not calls:
            raise ValueError(f"replay: no recorded call in {log.path}")
        for call in calls:
            request = call["request"]
            self._replies[request["role"], request["model"], request["prompt"]].append(call)

    @classmethod
    def from_config(cls, spec: dict) -> "ReplayBackend":
        """Build from ``{"log": <the llm_log directory of the recorded run>}``."""
        return cls(LLMLog(spec["log"]))

    def estimate(self, input_tokens: int, output_tokens: int) -> float:
        """Nothing: no provider is called."""
        return 0.0

    def complete(self, request: Request) -> Reply:
        """Return the recorded reply to ``request``.

        Raises
        ------
        ReplayExhausted
            If the log holds no reply left for this role, model and prompt.
        """
        queue = self._replies.get((request.role, request.model, request.prompt))
        if not queue:
            raise ReplayExhausted(
                f"replay of {self.log.path}: no recorded reply left for this {request.role} request "
                f"(the recorded run ran out of calls, or this run asked something it never did); "
                f"the prompt begins {request.prompt[:200]!r}"
            )
        call = queue.popleft()
        reply = call["reply"]
        return Reply(
            text=reply["text"], usage=reply["usage"], cost=reply["cost"], model_id=reply["model_id"],
            params=call["request"]["params"], raw=reply["raw"],
        )
