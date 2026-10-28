"""The Researcher: the LLM roles of a round, from prompt to checked candidates.

Four roles, each with its own backend and model: the planner plans the
round's directions to the quota of each mechanism type, the proposer writes
one candidate per direction (and fixes those the parser or the Gate
refuses), the reviewer judges each gated candidate's consistency in a
separate call (and the proposer rewrites, once, each it finds
inconsistent: ``revise``), and the distiller turns the ledger into the memory.

What the prompts say comes from ``Prompts``: the templates (built in or
replaced by the config), the market, the label's horizon, the scoring, the
vocabulary and the mechanism types, which also set the plan's slots and
the types a candidate may have.

A round's directions are split into proposal groups: contiguous runs of
near-equal size, in the plan's type order. Each group's first proposal,
fixes, review, rewrite and second review are calls of their own, and every
group's call of a stage is sent at once on threads and waited for; the
parser and the Gate then run over the whole round in slot order, so the
outcome does not depend on which reply lands first. The planner and the
distiller make one call a round.
"""

import dataclasses
import json
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from factor_agent.candidate import Candidate
from factor_agent.events import CallEnd, CallStart, Event, Events, GateOutcome, GateStep, ReviewVerdict, reply_facts
from factor_agent.gate import Gate, Passed
from factor_agent.llm.base import LLMBackend, ProviderQuotaExhausted, Request
from factor_agent.prompts import MEMORY_ITEMS, MEMORY_PARTS, SECTIONS, Prompts
from factor_agent.zoo import Zoo

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

#: The planner's actions: a new mechanism, an orthogonal mutation of an accepted one, a refinement of one.
ACTIONS = ("new", "mutation", "refinement")

#: The context sections each role reads (see ``factor_agent.prompts.SECTIONS``); ``render_context`` shows them.
PLANNER_SECTIONS = ("seed", "families", "memory", "earlier_summary", "recent_history", "last_round", "best")
PROPOSER_SECTIONS = ("families", "memory", "last_round", "best")


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
    (None if it did not parse), ``attempts`` every refused version in order,
    ``group`` the proposal group whose calls write, fix and review it, and
    ``slot`` its place in the round, from 0.
    """

    direction: Direction
    item: object
    candidate: Candidate | None
    checked: Passed | None
    attempts: tuple[Attempt, ...]
    group: int
    slot: int


@dataclass(frozen=True)
class _Scope:
    """What an LLM call concerns, as its start and end in the events file say: its round, proposal group and slots.

    The planner's and the distiller's calls concern their round only.
    """

    round: int
    group: int | None = None
    slots: tuple[int, ...] = ()


def output_unit(checked: Passed) -> str:
    """The unit of a gated graph's output, as the Gate's dimension inference names it: "price", "pure number"...

    "unknown" where the inference cannot say, as for the log of a price.
    """
    return checked.stats.get("dimension", "unknown")


class Researcher:
    """The LLM roles of a campaign.

    Parameters
    ----------
    roles : dict
        Maps each role (``planner``, ``proposer``, ``reviewer``,
        ``distiller``) to its ``(backend, model)``.
    prompts : Prompts
        The templates and what fills them; its mechanism types are the plan's, in order.
    groups : int, default 1
        The proposal groups a round's directions are split into (at most one per direction).
    events : Events, optional
        The run's events file, which every call, every check of a version
        and every verdict is written to (``factor_agent.events``).
    """

    def __init__(
        self, roles: dict[str, tuple[LLMBackend, str]], prompts: Prompts, groups: int = 1,
        events: Events | None = None,
    ):
        self.roles = dict(roles)
        self.prompts = prompts
        self.groups = groups
        self.events = events
        self._types = prompts.mechanism_types

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
            "plan", round=number, context=render_context(context, PLANNER_SECTIONS), quota=json.dumps(quota),
        )
        issues, by_type = [], {kind: [] for kind in self._types}
        try:
            items = _json_array(self._ask("planner", prompt, _Scope(number)))
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

        Each proposal group's directions go in a request of their own, all
        sent at once. A candidate must have its direction's mechanism type.
        Every candidate the parser or ``gate`` (checking originality against
        ``zoo``) refuses is sent back with its reasons, all of a group's in
        one request, up to ``retries`` times; the candidates are checked in
        slot order once every group's reply is back.
        """
        directions_by_group = _split(len(directions), self.groups)
        context = render_context(context, PROPOSER_SECTIONS)
        items = self._ask_groups(
            "proposer", number, dict(enumerate(directions_by_group)), list(range(len(directions))),
            lambda slots: self.prompts.render(
                "propose", round=number, context=context, n=len(slots),
                directions=json.dumps([{"number": i + 1, **directions[s].to_dict()} for i, s in enumerate(slots)], indent=2),
            ),
            lambda slots, text: _items(text, [None] * len(slots)),
        )
        group_of = {slot: group for group, slots in enumerate(directions_by_group) for slot in slots}
        attempts: list[list[Attempt]] = [[] for _ in directions]
        proposals: list[Proposal | None] = [None] * len(directions)
        pending = list(range(len(directions)))
        for retry in range(retries + 1):
            for slot in pending:
                proposals[slot] = self._check(
                    number, slot, "fixed" if retry else "proposed", directions[slot], items[slot], gate, zoo,
                    attempts[slot], group_of[slot],
                )
            pending = [slot for slot in pending if proposals[slot].checked is None]
            if not pending or retry == retries:
                break
            for slot, item in zip(pending, self._ask_to_fix(number, [proposals[slot] for slot in pending])):
                items[slot] = item
        return proposals

    def review(self, number: int, proposals: list[Proposal]) -> list[Verdict]:
        """Judge the consistency of each gated proposal's candidate, one call per proposal group, all sent at once.

        Each candidate is listed with its output unit, as the Gate inferred
        it (``output_unit``). A group whose reply holds no usable verdict at
        all is asked once more; a candidate still without a verdict is inconsistent.
        """
        def render(positions):
            return self.prompts.render("review", round=number, candidates=json.dumps([
                {"number": i + 1, **proposals[p].candidate.to_dict(), "output_unit": output_unit(proposals[p].checked)}
                for i, p in enumerate(positions)
            ], indent=2))

        def read(positions, text):
            try:
                items = _json_array(text)
            except ValueError:
                items = []
            found: dict[int, Verdict] = {}
            for item in items:
                if isinstance(item, dict) and isinstance(item.get("consistent"), bool) and isinstance(item.get("number"), int):
                    found.setdefault(item["number"], Verdict(item["consistent"], str(item.get("reason", ""))))
            return [found.get(i + 1) for i in range(len(positions))]

        judged: dict[int, Verdict | None] = {}
        asking = _positions_by_group([p.group for p in proposals])
        slots = [p.slot for p in proposals]
        for _ in range(2):
            judged |= self._ask_groups("reviewer", number, asking, slots, render, read)
            asking = {group: positions for group, positions in asking.items() if all(judged[p] is None for p in positions)}
        missing = Verdict(False, "the reviewer gave no verdict on this candidate")
        verdicts = [judged[p] or missing for p in range(len(proposals))]
        for proposal, verdict in zip(proposals, verdicts):
            self._event(ReviewVerdict(
                round=number, slot=proposal.slot, consistent=verdict.consistent, reason=verdict.reason,
                rewrite=any(attempt.stage == "review" for attempt in proposal.attempts),
            ))
        return verdicts

    def revise(
        self, number: int, inconsistent: list[tuple[Proposal, Verdict]], gate: Gate, zoo: Zoo,
    ) -> list[Proposal]:
        """Send each candidate the reviewer found inconsistent back to be rewritten once; return the rewrites, in order.

        Every candidate goes, with the reviewer's reason and its output unit, in one rewrite request
        per proposal group, all sent at once, and its first review becomes an attempt of stage ``"review"``. Each
        rewrite is checked by the parser and ``gate`` once, with no further
        retries: a refused rewrite is a proposal without ``checked``. The
        refused versions' graphs are withdrawn from ``gate`` first, so a
        rewrite that keeps its graph is not its own duplicate.
        """
        if not inconsistent:
            return []
        sent_back = []
        for proposal, verdict in inconsistent:
            gate.withdraw(proposal.checked.graph)
            review = Attempt(proposal.item, "review", (verdict.refusal,))
            sent_back.append(dataclasses.replace(proposal, attempts=(*proposal.attempts, review)))
        return [
            self._check(
                number, proposal.slot, "rewritten", proposal.direction, item, gate, zoo, list(proposal.attempts),
                proposal.group,
            )
            for proposal, item in zip(sent_back, self._ask_to_fix(number, sent_back))
        ]

    def distill(self, number: int, memory: str, history: str) -> str | None:
        """The memory the distiller folds from the previous ``memory`` and ``history``, the trials since it.

        The reply must hold the three parts under their headings, in order;
        each part keeps its first ``MEMORY_ITEMS`` items. An empty reply, or
        one without the three headings (cut short, or not a memory), gives
        None, and the previous memory stays.
        """
        reply = self._ask("distiller", self.prompts.render(
            "distill", round=number, memory=memory or "(none: this is the first memory)", history=history,
        ), _Scope(number))
        return _memory(reply)

    def _check(
        self, number: int, slot: int, step: GateStep, direction: Direction, item: object, gate: Gate, zoo: Zoo,
        attempts: list[Attempt], group: int,
    ) -> Proposal:
        """Parse and gate one item, of ``slot`` in proposal group ``group``; a refusal is appended to ``attempts``.

        ``step`` says which version of the slot's candidate it is, as the
        events file names it: ``"proposed"``, ``"fixed"`` or ``"rewritten"``.
        """
        def refused(stage, reasons, candidate=None, shown=item, hints=()):
            attempts.append(Attempt(shown, stage, tuple(reasons), tuple(hints)))
            self._event(GateOutcome(
                round=number, slot=slot, step=step, item=shown, passed=False, stage=stage,
                reasons=list(reasons), hints=list(hints),
            ))
            return Proposal(direction, shown, candidate, None, tuple(attempts), group, slot)

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
            self._event(GateOutcome(round=number, slot=slot, step=step, item=item, passed=True))
            return Proposal(direction, item, candidate, checked, tuple(attempts), group, slot)
        return refused("gate", checked.reasons, candidate, hints=checked.hints)

    def _ask_to_fix(self, number: int, refused: list[Proposal]) -> list:
        """Send each refused proposal's last version back with its last attempt, one request per proposal group, all at once.

        Returns one fixed item per proposal, in order. A version that passed
        the Gate (one the reviewer sent back) is listed with its output unit.
        """
        listed = [
            {
                "direction": proposal.direction.to_dict(), "refused_version": proposal.item,
                **({"output_unit": output_unit(proposal.checked)} if proposal.checked is not None else {}),
                "refused_because": list(attempt.reasons), **({"how_to_write_it": list(attempt.hints)} if attempt.hints else {}),
            }
            for proposal in refused
            for attempt in [proposal.attempts[-1]]
        ]
        fixed = self._ask_groups(
            "proposer", number, _positions_by_group([proposal.group for proposal in refused]),
            [proposal.slot for proposal in refused],
            lambda positions: self.prompts.render(
                "retry", round=number, n=len(positions),
                failures=json.dumps([{"number": i + 1, **listed[p]} for i, p in enumerate(positions)], indent=2),
            ),
            lambda positions, text: _items(text, [refused[p].item for p in positions]),
        )
        return [fixed[p] for p in range(len(refused))]

    def _ask(self, role: str, prompt: str, scope: _Scope) -> str:
        """Send ``prompt`` to ``role``'s backend and return the reply's text.

        The call's start and end in the events file say what it concerns: ``scope``.
        """
        backend, model = self.roles[role]
        request = Request(role=role, model=model, prompt=prompt)
        if self.events is None:
            return backend.complete(request).text
        call = dict(call=self.events.next_call(), round=scope.round, role=role, model=model, group=scope.group,
                    slots=list(scope.slots))
        self.events.write(CallStart(**call))
        started = time.monotonic()
        try:
            # A recording backend says which call file it wrote; any other backend writes none.
            reply, path = backend.record(request) if hasattr(backend, "record") else (backend.complete(request), None)
        except BaseException as exc:
            self.events.write(CallEnd(**call, seconds=time.monotonic() - started, error=f"{type(exc).__name__}: {exc}"))
            raise
        self.events.write(CallEnd(
            **call, file=path.name if path else None, seconds=time.monotonic() - started, **reply_facts(reply),
        ))
        return reply.text

    def _event(self, event: Event) -> None:
        if self.events is not None:
            self.events.write(event)

    def _ask_groups(
        self, role: str, number: int, positions_by_group: dict[int, list[int]], slots: list[int],
        render: Callable[[list[int]], str], read: Callable[[list[int], str], list],
    ) -> dict[int, object]:
        """Ask ``role`` one call of round ``number`` per proposal group, all at once; scatter the replies to their positions.

        ``positions_by_group`` holds each group's positions by its number and
        ``slots`` each position's slot; ``render(positions)`` is the group's
        prompt and ``read(positions, text)`` one value per position from its reply.
        """
        groups = list(positions_by_group.items())
        replies = self._ask_all(role, [
            (render(positions), _Scope(number, group, tuple(slots[p] for p in positions))) for group, positions in groups
        ])
        return {
            p: value
            for (_, positions), text in zip(groups, replies)
            for p, value in zip(positions, read(positions, text), strict=True)
        }

    def _ask_all(self, role: str, asked: list[tuple[str, _Scope]]) -> list[str]:
        """Make ``asked`` calls to ``role`` side by side, one thread each; return the replies' texts in order.

        Each call is its prompt and what it concerns. Every call is waited for, so each that succeeds is
        logged; then a ``ProviderQuotaExhausted`` among the errors is raised,
        so the campaign stops for the quota, and otherwise the first error in
        the order of ``asked``.
        """
        if not asked:
            return []
        with ThreadPoolExecutor(max_workers=len(asked)) as pool:
            calls = [pool.submit(self._ask, role, prompt, scope) for prompt, scope in asked]
        errors = [call.exception() for call in calls if call.exception() is not None]
        if errors:
            raise next((error for error in errors if isinstance(error, ProviderQuotaExhausted)), errors[0])
        return [call.result() for call in calls]


def _split(count: int, parts: int) -> list[list[int]]:
    """The slots ``0..count-1`` cut into ``parts`` contiguous runs of near-equal size, the longer first; none empty."""
    parts = min(parts, count)
    if not parts:
        return []
    size, longer = divmod(count, parts)
    starts = [g * size + min(g, longer) for g in range(parts + 1)]
    return [list(range(a, b)) for a, b in zip(starts, starts[1:])]


def _positions_by_group(group_ids: list[int]) -> dict[int, list[int]]:
    """The positions of ``group_ids`` by proposal group: the groups in order, each one's positions in order."""
    return {group: [p for p, g in enumerate(group_ids) if g == group] for group in sorted(set(group_ids))}


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


def _memory(reply: str) -> str | None:
    """The memory in a distiller's ``reply``, with plain headings, "- " bullets and at most ``MEMORY_ITEMS`` items a part.

    A heading may come with a leading "#", surrounding "**" and a trailing
    ":", in any case; an item may start with "- ", "* ", "• ", "1." or "1)".
    None if a heading is missing or out of order, or if no part holds an item.
    """
    lines = reply.strip().removeprefix("```text").removesuffix("```").strip().splitlines()
    starts = [next((i for i, line in enumerate(lines) if _heading(line) == part.lower()), None) for part in MEMORY_PARTS]
    if None in starts or starts != sorted(starts):
        return None
    kept, found = [], 0
    for part, start, end in zip(MEMORY_PARTS, starts, [*starts[1:], len(lines)]):
        items = [match.group(1).strip() for line in lines[start + 1:end] if (match := _BULLET.match(line))]
        found += len(items)
        kept += [part, *(f"- {item}" for item in items[:MEMORY_ITEMS])]
    return "\n".join(kept) if found else None


#: An item of the memory: its bullet, then its text.
_BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+(\S.*)$")


def _heading(line: str) -> str:
    """``line`` as a bare lower-case heading: without leading "#", surrounding "**" and a trailing ":"."""
    text = line.strip().lstrip("#").strip().strip("*").strip().removesuffix(":").strip().strip("*").strip()
    return text.lower()


def render_context(context: dict[str, str], keys: tuple[str, ...]) -> str:
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
