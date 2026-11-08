"""The round in progress, replayed from the events since the run was last opened, and one LLM call read in full.

A round a resume dropped left its events before the resume, so it is not
among them; a run without events has no round in progress to show. An LLM
call is read by the number of its call file, from the file itself: tokens,
truncation and the model's reasoning are the provider's reply's; what it
concerned (round, slots) only the events file says.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from factor_agent import events
from factor_agent.llm._raw import raw_facts
from factor_agent.llm.base import Reply
from factor_agent.web._files import Replayed, loads, utc
from factor_agent.web._health import Tokens
from factor_agent.web._plans import picked_slots


@dataclass(frozen=True)
class Check:
    """One check of a pending trial's candidate: the parser and the Gate on one version of it, or the reviewer.

    ``by`` is ``"gate"`` or ``"reviewer"``. For the Gate, ``step`` is the
    version's: ``"proposed"`` (the first), ``"fixed"`` (sent back by the
    parser or the Gate) or ``"rewritten"`` (sent back by the reviewer), and
    ``stage`` is ``"parse"`` or ``"gate"`` when it refused the version,
    ``None`` when it passed it. ``passed`` is true when the Gate passed the
    version or the reviewer found it consistent. ``reasons`` are the Gate's
    reasons for a refusal, or the reviewer's reason for its verdict either
    way; ``hints`` are the Gate's.
    """

    by: str
    step: str | None
    passed: bool
    stage: str | None
    reasons: tuple[str, ...]
    hints: tuple[str, ...]
    time: datetime


@dataclass(frozen=True)
class PendingTrial:
    """A pending trial: what is known so far of one slot of the round in progress.

    ``stage`` is the furthest it went: ``"planned"`` (its research direction
    is chosen), ``"proposed"``, ``"fixed"`` (after the Gate refused it),
    ``"rewritten"`` (after the reviewer found it inconsistent), and
    ``"consistent"`` or ``"inconsistent"`` as the reviewer last found it.
    ``direction`` is its research direction as the plan gives it. ``latest``
    is its latest version as the proposer wrote it, ``None`` before the
    first. ``checks`` are the Gate's outcomes and the reviewer's verdicts
    so far, the earliest first. ``parent_picked`` is true when the code
    picked its parent (a Picked parent). It becomes a trial when its round
    is written.
    """

    slot: int
    stage: str
    direction: dict
    latest: object
    checks: tuple[Check, ...]
    parent_picked: bool = False


@dataclass(frozen=True)
class RoundCall:
    """An LLM call of the round in progress, as the call timeline shows it.

    ``number`` is its call file's number, by which ``read_call`` reads it;
    ``None`` while it has not come back, or when it failed (``error`` says
    how). ``group`` is its proposal group and ``slots`` the slots it
    concerns; both are empty for the planner, whose call concerns the round.
    ``sent`` and ``returned`` are when it was sent and came back (UTC).
    """

    number: int | None
    role: str
    model: str
    group: int | None
    slots: tuple[int, ...]
    sent: datetime
    returned: datetime | None
    error: str | None


@dataclass(frozen=True)
class RunningRound:
    """The round in progress, as its events tell since the run was last opened.

    ``started`` is when the round before it was written, or the run was
    opened when it is the first since. ``calls`` are its LLM calls in the
    order they were sent, and ``evaluation`` when the Evaluator started on
    it (``None`` before). ``pending`` holds a pending trial per slot of its
    plan, in slot order; it is empty until the plan is made.
    """

    round: int
    started: datetime
    calls: tuple[RoundCall, ...]
    evaluation: datetime | None
    pending: tuple[PendingTrial, ...]


@dataclass(frozen=True)
class LLMCall:
    """One LLM call of a run, in full, as its call file holds it.

    ``number`` is the number of its call file. ``round``, ``group`` and
    ``slots`` are the round, proposal group and slots it concerned, as the
    events file says (``None`` and empty without one). ``time`` is when it
    came back (UTC) and ``seconds`` how long it took (``None`` without an
    events file). ``tokens``, ``truncated`` (cut off at the output limit,
    ``None`` when the provider's reply does not say) and ``empty`` are read
    from the provider's reply. ``reasoning`` is the model's reasoning when
    the provider returns it, ``None`` otherwise.
    """

    number: int
    role: str
    model: str
    round: int | None
    group: int | None
    slots: tuple[int, ...]
    time: datetime
    seconds: float | None
    tokens: Tokens
    truncated: bool | None
    empty: bool
    prompt: str
    reply: str
    reasoning: str | None


def running(number: int, replayed: Replayed) -> RunningRound | None:
    """Round ``number``, in progress, from the events since the run was last opened; ``None`` without events."""
    if replayed.opened is None:
        return None
    started = replayed.opened.time
    calls: dict[int, RoundCall] = {}
    evaluation = None
    directions: list[dict] = []
    picked: set[int] = set()
    stages: dict[int, str] = {}
    latest: dict[int, object] = {}
    checks: dict[int, list[Check]] = {}
    for event in replayed.current:
        if isinstance(event, events.RoundEnd):
            started = event.time
        if getattr(event, "round", None) != number:
            continue
        if isinstance(event, events.CallEnd):
            sent = calls[event.call].sent if event.call in calls else event.time - timedelta(seconds=event.seconds)
            calls[event.call] = RoundCall(
                number=_call_number(event.file), role=event.role, model=event.model, group=event.group,
                slots=tuple(event.slots), sent=sent, returned=event.time, error=event.error,
            )
        elif isinstance(event, events.CallStart):
            calls[event.call] = RoundCall(number=None, role=event.role, model=event.model, group=event.group,
                                          slots=tuple(event.slots), sent=event.time, returned=None, error=None)
        elif isinstance(event, events.EvaluationStart):
            evaluation = event.time
        elif isinstance(event, events.Plan):
            directions, picked = event.directions, picked_slots(event)
            stages = dict.fromkeys(range(len(directions)), "planned")
        elif isinstance(event, events.GateOutcome):
            stages[event.slot], latest[event.slot] = event.step, event.item
            checks.setdefault(event.slot, []).append(Check(
                by="gate", step=event.step, passed=event.passed, stage=event.stage, reasons=tuple(event.reasons),
                hints=tuple(event.hints), time=event.time,
            ))
        elif isinstance(event, events.ReviewVerdict):
            stages[event.slot] = "consistent" if event.consistent else "inconsistent"
            checks.setdefault(event.slot, []).append(Check(
                by="reviewer", step=None, passed=event.consistent, stage=None, reasons=(event.reason,), hints=(),
                time=event.time,
            ))
    pending = tuple(
        PendingTrial(slot=slot, stage=stages[slot], direction=direction, latest=latest.get(slot),
                     checks=tuple(checks.get(slot, ())), parent_picked=slot in picked)
        for slot, direction in enumerate(directions)
    )
    return RunningRound(round=number, started=started, calls=tuple(calls.values()), evaluation=evaluation,
                        pending=pending)


def _call_number(file: str | None) -> int | None:
    """The number of the call file named ``file``, as ``<number>-<role>.json``."""
    return int(file.partition("-")[0]) if file else None


def read_call(run_dir: str | Path, number: int) -> LLMCall | None:
    """The LLM call of the run in ``run_dir`` whose call file has ``number``, ``None`` when it has none.

    A call of a round a resume dropped is not read: its file is no longer
    in the run's sequence of calls.
    """
    run_dir = Path(run_dir)
    files = list((run_dir / "llm_log").glob(f"{number:06d}-*.json")) if number > 0 else []
    if not files:
        return None
    path = files[0]
    try:
        call = loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    request, reply = call.get("request") or {}, call.get("reply") or {}
    text, raw = reply.get("text") or "", reply.get("raw") or ""
    facts = events.reply_facts(Reply(text=text, usage=reply.get("usage") or {}, model_id=reply.get("model_id") or "",
                                     raw=raw))
    end = next((e for e in reversed(events.read(run_dir / events.FILE))
                if isinstance(e, events.CallEnd) and e.file == path.name), None)
    return LLMCall(
        number=number, role=request.get("role") or path.stem.partition("-")[2], model=request.get("model") or "",
        round=end.round if end else None, group=end.group if end else None, slots=tuple(end.slots) if end else (),
        time=end.time if end else utc(path.stat().st_mtime), seconds=end.seconds if end else None,
        tokens=Tokens(facts["input_tokens"], facts["output_tokens"], facts["cache_hits"], facts["reasoning"]),
        truncated=facts["truncated"], empty=facts["empty"], prompt=request.get("prompt") or "", reply=text,
        reasoning=raw_facts(raw).reasoning,
    )
