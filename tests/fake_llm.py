"""A scripted fake of every LLM role, answering from what each request asks for.

The Researcher's prompts end with one fenced JSON block holding what the
role must answer about: the quota per mechanism type (planner), the
directions (proposer), the refused candidates (a fix request) or the
candidates to judge (reviewer). The fake reads that block and the
``Round: N`` line, and answers through the test's functions, so a script
holds whatever the quotas turn out to be. Every request is kept in
``requests``.
"""

import json
import re
from collections.abc import Callable

from factor_agent.config import DEFAULT_MECHANISMS
from factor_agent.config import ROLES
from factor_agent.llm.base import Reply, Request

_BLOCK = re.compile(r"```json\s*(.*?)```", re.DOTALL)
_ROUND = re.compile(r"^Round: (\d+)$", re.MULTILINE)


def last_block(prompt: str):
    """The last fenced JSON block of ``prompt``."""
    return json.loads(_BLOCK.findall(prompt)[-1])


def round_of(prompt: str) -> int:
    """The round a prompt belongs to."""
    return int(_ROUND.search(prompt).group(1))


def reply(value) -> str:
    return "Here it is.\n```json\n" + json.dumps(value, indent=2) + "\n```\n"


class FakeLLM:
    """Every role of the campaign.

    Parameters
    ----------
    propose : callable
        ``propose(round, directions)`` returns one candidate item per direction.
    fix : callable, optional
        ``fix(round, failures)`` returns one item per refused candidate;
        by default each comes back unchanged.
    consistent : callable, optional
        ``consistent(candidate)`` is the reviewer's verdict; by default
        every candidate whose description does not say "inconsistent" passes.
    cost : float
        The cost of every reply.
    interrupt : callable, optional
        ``interrupt(request)`` is true for the request at which the fake
        raises ``Interrupted``, as a crash mid-campaign would.

    An item whose ``mechanism_type`` is one of the default types is given its
    direction's type, whatever the config's types, so a script need not know
    the quotas; any other type is left for the parser to refuse.
    """

    def __init__(
        self, propose: Callable, fix: Callable | None = None, consistent: Callable | None = None,
        cost: float = 0.0, interrupt: Callable | None = None,
    ):
        self._propose = propose
        self._fix = fix or (lambda number, failures: [f["candidate"] for f in failures])
        self._consistent = consistent or (lambda candidate: "inconsistent" not in candidate["description"])
        self._cost = cost
        self._interrupt = interrupt or (lambda request: False)
        self.requests: list[Request] = []
        self.distilled = 0

    def backends(self) -> dict:
        """The backend of every role: this fake."""
        return dict.fromkeys(ROLES, self)

    def complete(self, request: Request) -> Reply:
        if self._interrupt(request):
            raise Interrupted(request.role)
        self.requests.append(request)
        text = getattr(self, f"_{request.role}")(request.prompt)
        return Reply(text=text, usage={}, cost=self._cost, model_id=f"fake-{request.role}")

    def of_role(self, role: str) -> list[Request]:
        return [r for r in self.requests if r.role == role]

    def _planner(self, prompt: str) -> str:
        quota = last_block(prompt)
        return reply([
            {"mechanism_type": kind, "action": "new", "parent": "", "direction": f"a {kind} direction"}
            for kind, count in quota.items() for _ in range(count)
        ])

    def _proposer(self, prompt: str) -> str:
        number, block = round_of(prompt), last_block(prompt)
        if isinstance(block, list) and block and "refused_because" in block[0]:
            items = self._fix(number, block)
            directions = [failure["direction"] for failure in block]
        else:
            directions = block
            items = self._propose(number, directions)
        return reply([_typed(item, d) for item, d in zip(items, directions)])

    def _reviewer(self, prompt: str) -> str:
        return reply([
            {"number": c["number"], "consistent": self._consistent(c),
             "reason": "consistent" if self._consistent(c) else "the code does not compute what the description says"}
            for c in last_block(prompt)
        ])

    def _distiller(self, prompt: str) -> str:
        self.distilled += 1
        return f"Distilled memory number {self.distilled}: volume surprises keep working."


class Interrupted(RuntimeError):
    """The campaign was stopped from outside mid-round."""


def _typed(item, direction: dict):
    if isinstance(item, dict) and item.get("mechanism_type") in {m.name for m in DEFAULT_MECHANISMS}:
        return {**item, "mechanism_type": direction["mechanism_type"]}
    return item
