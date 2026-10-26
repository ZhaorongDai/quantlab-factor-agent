"""A scripted fake of every LLM role, answering from what each request asks for.

The Researcher's prompts hold what the role must answer about as JSON inside
an XML tag on lines of its own: ``<quota>`` (planner), ``<directions>``
(proposer), ``<refused_candidates>`` (a fix request) or ``<candidates>``
(reviewer). The fake reads that tag and the ``Round: N`` line, and answers
through the test's functions, so a script holds whatever the quotas turn
out to be, and the replies do not depend on the order in which calls sent
side by side arrive. Every request is kept in ``requests``, in arrival
order; the fake is safe to call from several threads.
"""

import json
import re
import threading
from collections.abc import Callable

from factor_agent.config import DEFAULT_MECHANISMS
from factor_agent.config import ROLES
from factor_agent.llm.base import ProviderQuotaExhausted, Reply, Request

#: What the fake says when it refuses a request for the quota.
QUOTA_MESSAGE = "fake: Insufficient Balance"

_ROUND = re.compile(r"^Round: (\d+)$", re.MULTILINE)


def tagged(prompt: str, tag: str):
    """The JSON between the lines ``<tag>`` and ``</tag>`` of ``prompt``."""
    [body] = re.findall(rf"^<{tag}>\n(.*?)\n</{tag}>$", prompt, re.DOTALL | re.MULTILINE)
    return json.loads(body)


def has_tag(prompt: str, tag: str) -> bool:
    """Whether ``prompt`` has the line ``<tag>``."""
    return re.search(rf"^<{tag}>$", prompt, re.MULTILINE) is not None


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
        ``fix(round, failures)`` returns one item per refused entry (each
        with its ``direction``, ``refused_version`` and ``refused_because``);
        by default each refused version comes back unchanged.
    consistent : callable, optional
        ``consistent(candidate)`` is the reviewer's verdict, on the first
        review and on the review of a rewrite alike (a rewrite reads as the
        fix made it); by default every candidate whose description does not
        say "inconsistent" passes.
    interrupt : callable, optional
        ``interrupt(request)`` is true for the request at which the fake
        raises ``Interrupted``, as a crash mid-campaign would.
    quota : callable, optional
        ``quota(request)`` is true for every request the fake refuses with
        ``ProviderQuotaExhausted``, as a provider whose balance ran out would.
    plan : callable, optional
        ``plan(round, quota, prompt)`` returns the planner's directions; by
        default one new direction per slot.
    distill : callable, optional
        ``distill(call)`` returns the distiller's reply to its ``call``-th
        request (from 1); by default a memory in the three parts whose first
        item reads "Distilled memory number <call>".

    An item whose ``mechanism_type`` is one of the default types is given its
    direction's type, whatever the config's types, so a script need not know
    the quotas; any other type is left for the parser to refuse.
    """

    def __init__(
        self, propose: Callable, fix: Callable | None = None, consistent: Callable | None = None,
        interrupt: Callable | None = None, plan: Callable | None = None, distill: Callable | None = None,
        quota: Callable | None = None,
    ):
        self._propose = propose
        self._fix = fix or (lambda number, failures: [f["refused_version"] for f in failures])
        self._consistent = consistent or (lambda candidate: "inconsistent" not in candidate["description"])
        self._interrupt = interrupt or (lambda request: False)
        self._quota = quota or (lambda request: False)
        self._plan = plan or new_directions
        self._distill = distill or memory
        self.requests: list[Request] = []
        self.distilled = 0
        self._lock = threading.Lock()

    def backends(self) -> dict:
        """The backend of every role: this fake."""
        return dict.fromkeys(ROLES, self)

    def complete(self, request: Request) -> Reply:
        if self._interrupt(request):
            raise Interrupted(request.role)
        if self._quota(request):
            raise ProviderQuotaExhausted(QUOTA_MESSAGE)
        with self._lock:
            self.requests.append(request)
        text = getattr(self, f"_{request.role}")(request.prompt)
        return Reply(text=text, usage={}, model_id=f"fake-{request.role}")

    def of_role(self, role: str) -> list[Request]:
        return [r for r in self.requests if r.role == role]

    def _planner(self, prompt: str) -> str:
        return reply(self._plan(round_of(prompt), tagged(prompt, "quota"), prompt))

    def _proposer(self, prompt: str) -> str:
        number = round_of(prompt)
        if has_tag(prompt, "refused_candidates"):
            failures = tagged(prompt, "refused_candidates")
            items = self._fix(number, failures)
            directions = [failure["direction"] for failure in failures]
        else:
            directions = tagged(prompt, "directions")
            items = self._propose(number, directions)
        return reply([_typed(item, d) for item, d in zip(items, directions)])

    def _reviewer(self, prompt: str) -> str:
        return reply([
            {"number": c["number"], "consistent": self._consistent(c),
             "reason": "consistent" if self._consistent(c) else "the code does not compute what the description says"}
            for c in tagged(prompt, "candidates")
        ])

    def _distiller(self, prompt: str) -> str:
        with self._lock:
            self.distilled += 1
            call = self.distilled
        return self._distill(call)


class Interrupted(RuntimeError):
    """The campaign was stopped from outside mid-round."""


def memory(number: int) -> str:
    """A memory in the three parts, told apart by ``number``."""
    return (
        f"Working mechanism patterns\n- Distilled memory number {number}: volume surprises keep working.\n"
        "Failure constraints\n- Write ratios: levels rank by price.\n"
        "Directional findings\n- Volume surges were followed by outperformance."
    )


def new_directions(number: int, quota: dict, prompt: str) -> list[dict]:
    """The default plan: one new direction per slot."""
    return [
        {"mechanism_type": kind, "action": "new", "parent": "", "direction": f"a {kind} direction"}
        for kind, count in quota.items() for _ in range(count)
    ]


def _typed(item, direction: dict):
    if isinstance(item, dict) and item.get("mechanism_type") in {m.name for m in DEFAULT_MECHANISMS}:
        return {**item, "mechanism_type": direction["mechanism_type"]}
    return item
