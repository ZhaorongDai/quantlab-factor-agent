"""The Researcher: the LLM roles of a round, from prompt to checked candidates.

Four roles, each with its own backend and model: the planner plans the
round's directions to the quota of each mechanism type, the proposer writes
one candidate per direction (and fixes those the parser or the Gate
refuses), the reviewer judges each gated candidate's consistency in a
separate call (and the proposer rewrites, once, each it finds
inconsistent: ``revise``), and the distiller turns the ledger into the memory. Every
call's cost is added to ``spent``; once ``spent`` reaches the budget, the
next call raises ``BudgetExhausted`` instead of being sent.

What the prompts say comes from ``Prompts``: the templates (built in or
replaced by the config), the market, the label's horizon, the scoring, the
vocabulary and the mechanism types, which also set the plan's slots and
the types a candidate may have.
"""

import dataclasses
import json
import re
from dataclasses import asdict, dataclass

from factor_agent.candidate import Candidate
from factor_agent.gate import Gate, Passed
from factor_agent.llm.base import LLMBackend, Request
from factor_agent.prompts import SECTIONS, Prompts
from factor_agent.zoo import Zoo

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

#: The planner's actions: a new mechanism, an orthogonal mutation of an accepted one, a refinement of one.
ACTIONS = ("new", "mutation", "refinement")

#: The context sections each role reads (see ``factor_agent.prompts.SECTIONS``).
_PLANNER_SECTIONS = ("seed", "families", "memory", "history", "last_round", "best")
_PROPOSER_SECTIONS = ("families", "memory", "last_round", "best")


@dataclass(frozen=True)
class Direction:
    """One planned slot of a round: its mechanism type, the action, the parent factor and the idea.

    ``parent`` names an accepted factor for a mutation or a refinement, and
    is empty for a new direction.
    """

    mechanism_type: str
    action: str
    parent: str
    direction: str

    def to_dict(self) -> dict:
        """The direction as the ledger and the prompts hold it."""
        return asdict(self)


@dataclass(frozen=True)
class Verdict:
    """The reviewer's judgement of one candidate."""

    consistent: bool
    reason: str

    @property
    def refusal(self) -> str:
        """The reason an inconsistent candidate is sent back, and rejected, with: the reviewer's own words."""
        return f"the reviewer found it inconsistent: {self.reason}"


@dataclass(frozen=True)
class Attempt:
    """One refused version of a candidate: what the proposer wrote, where it failed and why.

    ``stage`` is ``"parse"`` (not a valid candidate for its direction), ``"gate"``
    or ``"review"`` (the reviewer found it inconsistent).
    """

    item: object
    stage: str
    reasons: tuple[str, ...]
    #: How to write it instead, from the vocabulary's fixes (``Rejected.hints``); sent with the reasons.
    hints: tuple[str, ...] = ()


@dataclass(frozen=True)
class Proposal:
    """One direction of a round after its retries.

    ``checked`` is set when the last version passed the Gate; otherwise the
    slot is a failed trial and the last of ``attempts`` says why. ``item``
    is the last version the proposer wrote, ``candidate`` its parsed form
    (None if it did not parse), and ``attempts`` every refused version in order.
    """

    direction: Direction
    item: object
    candidate: Candidate | None
    checked: Passed | None
    attempts: tuple[Attempt, ...]


def output_unit(checked: Passed) -> str:
    """The unit of a gated graph's output, as the Gate's dimension inference names it: "price", "pure number"...

    "unknown" where the inference cannot say, as for the log of a price.
    """
    return checked.stats.get("dimension", "unknown")


class BudgetExhausted(RuntimeError):
    """The calls have cost the budget; no further call is sent."""


class Researcher:
    """The LLM roles of a campaign.

    Parameters
    ----------
    roles : dict
        Maps each role (``planner``, ``proposer``, ``reviewer``,
        ``distiller``) to its ``(backend, model)``.
    prompts : Prompts
        The templates and what fills them; its mechanism types are the plan's, in order.
    spent : float
        US dollars already spent, by the rounds a resumed campaign ran.
    budget : float, optional
        US dollars the calls may cost; None for no limit.
    on_cost : callable, optional
        Called with ``spent`` after every call, so the spending survives a crash.
    """

    def __init__(
        self, roles: dict[str, tuple[LLMBackend, str]], prompts: Prompts, spent: float = 0.0,
        budget: float | None = None, on_cost=None,
    ):
        self.roles = dict(roles)
        self.prompts = prompts
        self._types = prompts.mechanism_types
        #: US dollars spent on calls so far.
        self.spent = spent
        self._budget = budget
        self._on_cost = on_cost or (lambda spent: None)

    def plan(
        self, number: int, quota: dict[str, int], context: dict[str, str], parents: list[str],
    ) -> tuple[list[Direction], list[str]]:
        """Return round ``number``'s directions, ``quota[type]`` of each type in type order, and the plan's issues.

        ``parents`` are the accepted factors a mutation or a refinement may
        name. A direction the planner leaves out, or writes with an unknown
        type, action or parent, is replaced by a new direction of its type;
        each such repair is one issue.
        """
        prompt = self.prompts.render(
            "plan", round=number, context=_render(context, _PLANNER_SECTIONS), quota=json.dumps(quota),
        )
        issues, by_type = [], {kind: [] for kind in self._types}
        try:
            items = _json_array(self._ask("planner", prompt))
        except ValueError as exc:
            items, issues = [], [f"planner: {exc}"]
        for i, item in enumerate(items, 1):
            problem = _direction_problem(item, parents, self._types)
            if problem:
                issues.append(f"planner: direction {i} {problem}")
            else:
                by_type[item["mechanism_type"]].append(Direction(**{k: item[k] for k in Direction.__annotations__}))
        directions = []
        for kind in self._types:
            if len(by_type[kind]) > quota[kind]:
                issues.append(f"planner: {len(by_type[kind])} {kind} directions for {quota[kind]} slots; the first are kept")
            planned = by_type[kind][:quota[kind]]
            if len(planned) < quota[kind]:
                issues.append(f"planner: {len(planned)} of {quota[kind]} {kind} directions; the rest are new directions")
            planned += [
                Direction(kind, "new", "", f"a {kind} mechanism not tried before")
                for _ in range(quota[kind] - len(planned))
            ]
            directions += planned
        return directions, issues

    def propose(
        self, number: int, directions: list[Direction], context: dict[str, str], gate: Gate, zoo: Zoo, retries: int,
    ) -> list[Proposal]:
        """Ask for one candidate per direction and return one ``Proposal`` per direction, in order.

        A candidate must have its direction's mechanism type. Every
        candidate the parser or ``gate`` (checking originality against
        ``zoo``) refuses is sent back with its reasons, all of a round's in
        one request, up to ``retries`` times.
        """
        prompt = self.prompts.render(
            "propose", round=number, context=_render(context, _PROPOSER_SECTIONS), n=len(directions),
            directions=json.dumps([{"number": i + 1, **d.to_dict()} for i, d in enumerate(directions)], indent=2),
        )
        items = _items(self._ask("proposer", prompt), [None] * len(directions))
        attempts: list[list[Attempt]] = [[] for _ in directions]
        proposals: list[Proposal | None] = [None] * len(directions)
        pending = list(range(len(directions)))
        for retry in range(retries + 1):
            for slot in pending:
                proposals[slot] = self._check(directions[slot], items[slot], gate, zoo, attempts[slot])
            pending = [slot for slot in pending if proposals[slot].checked is None]
            if not pending or retry == retries:
                break
            fixed = self._ask_to_fix(number, [(directions[slot], items[slot], attempts[slot][-1]) for slot in pending])
            for slot, item in zip(pending, fixed):
                items[slot] = item
        return proposals

    def review(self, number: int, proposals: list[Proposal]) -> list[Verdict]:
        """Judge the consistency of each gated proposal's candidate, in one call.

        Each candidate is listed with its output unit, as the Gate inferred
        it (``output_unit``). A reply with no usable verdict at all is asked
        for once more; a candidate still without a verdict is inconsistent.
        """
        if not proposals:
            return []
        listed = [
            {"number": i + 1, **p.candidate.to_dict(), "output_unit": output_unit(p.checked)}
            for i, p in enumerate(proposals)
        ]
        prompt = self.prompts.render("review", round=number, candidates=json.dumps(listed, indent=2))
        verdicts: dict[int, Verdict] = {}
        for _ in range(2):
            try:
                items = _json_array(self._ask("reviewer", prompt))
            except ValueError:
                items = []
            for item in items:
                if isinstance(item, dict) and isinstance(item.get("consistent"), bool) and isinstance(item.get("number"), int):
                    verdicts.setdefault(item["number"], Verdict(item["consistent"], str(item.get("reason", ""))))
            if verdicts:
                break
        missing = Verdict(False, "the reviewer gave no verdict on this candidate")
        return [verdicts.get(i + 1, missing) for i in range(len(proposals))]

    def revise(
        self, number: int, inconsistent: list[tuple[Proposal, Verdict]], gate: Gate, zoo: Zoo,
    ) -> list[Proposal]:
        """Send each candidate the reviewer found inconsistent back to be rewritten once; return the rewrites, in order.

        Every candidate goes, with the reviewer's reason and its output unit, in one rewrite request,
        and its first review becomes an attempt of stage ``"review"``. Each
        rewrite is checked by the parser and ``gate`` once, with no further
        retries: a refused rewrite is a proposal without ``checked``. The
        refused versions' graphs are withdrawn from ``gate`` first, so a
        rewrite that keeps its graph is not its own duplicate.
        """
        if not inconsistent:
            return []
        attempts = []
        for proposal, verdict in inconsistent:
            gate.withdraw(proposal.checked.graph)
            attempts.append([*proposal.attempts, Attempt(proposal.item, "review", (verdict.refusal,))])
        fixed = self._ask_to_fix(
            number, [(proposal.direction, proposal.item, tried[-1]) for (proposal, _), tried in zip(inconsistent, attempts)],
            units=[output_unit(proposal.checked) for proposal, _ in inconsistent],
        )
        return [
            self._check(proposal.direction, item, gate, zoo, tried)
            for (proposal, _), item, tried in zip(inconsistent, fixed, attempts)
        ]

    def distill(self, number: int, history: str) -> str:
        """The memory the distiller writes from ``history``, the trials before round ``number``."""
        return self._ask("distiller", self.prompts.render("distill", round=number, history=history)).strip()

    def _check(self, direction: Direction, item: object, gate: Gate, zoo: Zoo, attempts: list[Attempt]) -> Proposal:
        """Parse and gate one item; a refusal is appended to ``attempts``."""
        def refused(stage, reasons, candidate=None, shown=item, hints=()):
            attempts.append(Attempt(shown, stage, tuple(reasons), tuple(hints)))
            return Proposal(direction, shown, candidate, None, tuple(attempts))

        if isinstance(item, _Missing):
            return refused("parse", [item.reason], shown=item.previous)
        try:
            candidate = Candidate.from_dict(item, self._types)
        except ValueError as exc:
            return refused("parse", [str(exc)])
        if candidate.mechanism_type != direction.mechanism_type:
            return refused("parse", [
                f"mechanism_type {candidate.mechanism_type!r} is not its direction's {direction.mechanism_type!r}"
            ])
        checked = gate.check(candidate, zoo)
        if isinstance(checked, Passed):
            return Proposal(direction, item, candidate, checked, tuple(attempts))
        return refused("gate", checked.reasons, candidate, hints=checked.hints)

    def _ask_to_fix(
        self, number: int, failures: list[tuple[Direction, object, Attempt]], units: list[str] | None = None,
    ) -> list:
        """Send the refused items back; return one fixed item per failure, in order.

        ``units`` gives each refused version's output unit, for versions that
        passed the Gate (those the reviewer sent back); without it no entry has one.
        """
        items = [item.previous if isinstance(item, _Missing) else item for _, item, _ in failures]
        units = units or [None] * len(failures)
        listed = [
            {
                "number": i + 1, "direction": direction.to_dict(), "refused_version": item,
                **({"output_unit": unit} if unit is not None else {}),
                "refused_because": list(attempt.reasons), **({"how_to_write_it": list(attempt.hints)} if attempt.hints else {}),
            }
            for i, ((direction, _, attempt), item, unit) in enumerate(zip(failures, items, units, strict=True))
        ]
        prompt = self.prompts.render("retry", round=number, failures=json.dumps(listed, indent=2), n=len(failures))
        return _items(self._ask("proposer", prompt), items)

    def _ask(self, role: str, prompt: str) -> str:
        """Send ``prompt`` to ``role``'s backend, count its cost and return the reply's text.

        Raises
        ------
        BudgetExhausted
            If the calls so far have cost the budget.
        """
        if self._budget is not None and self.spent >= self._budget:
            raise BudgetExhausted(f"the LLM calls cost ${self.spent:.2f}, the budget is ${self._budget:.2f}")
        backend, model = self.roles[role]
        reply = backend.complete(Request(role=role, model=model, prompt=prompt))
        self.spent += reply.cost
        self._on_cost(self.spent)
        return reply.text


@dataclass(frozen=True)
class _Missing:
    """No usable item came back for a slot; ``previous`` is its last version (None before the first)."""

    reason: str
    previous: object


def _items(text: str, previous: list) -> list:
    """One item per slot of ``previous`` from a reply's JSON array; a slot without one is ``_Missing``."""
    try:
        items = _json_array(text)
    except ValueError as exc:
        return [_Missing(str(exc), item) for item in previous]
    short = f"the reply had {len(items)} items for {len(previous)} candidates"
    return [_unwrapped(items[i]) if i < len(items) else _Missing(short, item) for i, item in enumerate(previous)]


#: The fields of a candidate, as the proposer writes them.
_CANDIDATE_FIELDS = frozenset(field.name for field in dataclasses.fields(Candidate))
#: The keys a reply may nest a candidate under when it echoes the entry it answers.
_NESTED = ("candidate", "refused_version")


def _unwrapped(item: object) -> object:
    """``item``, or the candidate it nests under "candidate" or "refused_version" when it lacks a candidate's fields.

    A reply to a rewrite request sometimes echoes the refused entry with the
    corrected candidate inside it; only that shape is unwrapped, so any other
    item reaches the parser as it came.
    """
    if not isinstance(item, dict) or _CANDIDATE_FIELDS <= item.keys():
        return item
    nested = [item[key] for key in _NESTED if isinstance(item.get(key), dict)]
    return nested[0] if nested else item


def _direction_problem(item: object, parents: list[str], types: tuple[str, ...]) -> str:
    """Why a planner item is not a usable direction, or an empty string."""
    if not isinstance(item, dict) or not all(isinstance(item.get(k), str) for k in Direction.__annotations__):
        return f"needs the string fields {list(Direction.__annotations__)}"
    if item["mechanism_type"] not in types:
        return f"has the unknown mechanism type {item['mechanism_type']!r}"
    if item["action"] not in ACTIONS:
        return f"has the unknown action {item['action']!r}"
    if item["action"] != "new" and item["parent"] not in parents:
        return f"names {item['parent']!r}, which is not an accepted factor"
    return ""


def _render(context: dict[str, str], keys: tuple[str, ...]) -> str:
    """The non-empty sections of ``context`` among ``keys``, each under its heading inside the tag of its key."""
    return "\n\n".join(
        f"## {heading}\n<{key}>\n{context[key]}\n</{key}>"
        for key, heading in SECTIONS if key in keys and context.get(key)
    )


def _json_array(text: str) -> list:
    """Return the JSON array in ``text``: the last fenced block if there is one, else the whole text."""
    fenced = _FENCE.findall(text)
    body = fenced[-1] if fenced else text
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"the reply is not JSON ({exc}): {text[:200]!r}") from exc
    if not isinstance(value, list):
        raise ValueError(f"the reply is a JSON {type(value).__name__}, not an array")
    return value
