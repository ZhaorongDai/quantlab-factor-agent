"""The events file: what a campaign is doing, appended as it happens, for anyone watching the run.

A campaign appends one event per line to ``events.jsonl`` in its run
directory: a JSON object with the file's schema version (``"schema"``), the
event's type (``"type"``), the time it happened (``"time"``, UTC, ISO 8601)
and the event's fields. The file is append-only, and the campaign never
reads it: it takes no part in a resume, a replay or the run's snapshot, so
deleting it loses the record of what happened but changes no result. A
resume that drops a round a crash left half written says so in its
``resume`` event instead of removing that round's events.

The event types, defined here once and read back by ``read``:

- ``start`` and ``resume`` (``Start``, ``Resume``): the campaign's process
  id and host, the code versions, the rounds already written, the stop
  conditions (round and trial limits, patience) and the admission limits;
  a resume also names the round it dropped;
- ``plan`` (``Plan``): a round's research directions in slot order, its
  quota and the plan's issues;
- ``call_start`` and ``call_end`` (``CallStart``, ``CallEnd``): every LLM
  call, with its role, the proposal group and the slots it concerns; its
  end names the call file it wrote, how long it took, its tokens as far
  as the provider's reply reports them, and whether the reply was cut off
  at the output limit or empty;
- ``gate`` (``GateOutcome``): every version of a slot's candidate the
  parser and the Gate checked, passed or refused with the reasons and hints;
- ``review`` (``ReviewVerdict``): the reviewer's verdict on a slot, with its reason;
- ``evaluation`` (``EvaluationStart``): the Evaluator starts on the round's
  candidates the reviewer found consistent;
- ``admission`` (``AdmissionOutcome``): a slot's outcome after evaluation,
  with the t threshold in force and the metrics;
- ``round_end`` (``RoundEnd``): a round's trials are written;
- ``stop`` (``Stopped``): the run stopped, as the state file records it.

A slot is a candidate's place in its round, from 0, in the order of the
round's research directions (and of the round's lines in the ledger). A
value the campaign holds as NaN or infinite is written as ``null``.
"""

import itertools
import json
import math
import threading
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

#: The events file's name in a run directory.
FILE = "events.jsonl"

#: The version of the events file's format: it changes when an event loses or changes a field.
SCHEMA = 1


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, kw_only=True)
class Event:
    """What every event has: the time it happened (UTC)."""

    #: The event's type in the file; each event class sets its own.
    type = ""
    time: datetime = field(default_factory=_now)


@dataclass(frozen=True, kw_only=True)
class Start(Event):
    """A campaign opened its run directory, with what it runs on and when it stops.

    ``pid`` and ``host`` are the campaign's process id and host name;
    ``versions`` the code versions the run's snapshot records for this
    opening; ``rounds`` the rounds already written. ``max_rounds``,
    ``max_trials`` and ``patience`` are the stop conditions, ``candidates``
    the slots of a round; ``t_threshold``, ``max_correlation``,
    ``min_coverage`` and ``min_monotonicity`` the admission settings.
    """

    type = "start"
    pid: int
    host: str
    versions: dict
    rounds: int
    candidates: int
    max_rounds: int
    max_trials: int
    patience: int
    t_threshold: float
    max_correlation: float
    min_coverage: float
    min_monotonicity: float


@dataclass(frozen=True, kw_only=True)
class Resume(Start):
    """A campaign opened a run directory it had opened before; ``dropped`` is the round it drops, if any.

    The events of the dropped round stay in the file before this one; the
    round runs again under the same number after it.
    """

    type = "resume"
    dropped: int | None = None


@dataclass(frozen=True, kw_only=True)
class Plan(Event):
    """Round ``round``'s research directions, in slot order, the quota by mechanism type and the plan's issues."""

    type = "plan"
    round: int
    directions: list[dict]
    quota: dict[str, int]
    issues: list[str]


@dataclass(frozen=True, kw_only=True)
class CallStart(Event):
    """An LLM call was sent.

    ``call`` tells the calls of one opening of the run apart (``CallEnd``
    names the same). ``group`` is the proposal group and ``slots`` the
    slots the call concerns; both are empty for the planner and the
    distiller, whose calls concern the whole round.
    """

    type = "call_start"
    call: int
    round: int
    role: str
    model: str
    group: int | None = None
    slots: list[int] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True)
class CallEnd(CallStart):
    """An LLM call came back, or failed.

    ``file`` is the call file's name in ``llm_log/``, ``None`` when the call
    failed (``error`` says how) and wrote none. ``seconds`` is how long it
    took. The token counts are the provider's, ``None`` where its reply
    does not report them: ``cache_hits`` are input tokens read from the
    provider's cache, ``reasoning`` output tokens spent reasoning.
    ``truncated`` is true when the reply was cut off at the output limit,
    ``None`` when the provider's reply does not say; ``empty`` when the
    reply holds no text.
    """

    type = "call_end"
    file: str | None = None
    seconds: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_hits: int | None = None
    reasoning: int | None = None
    truncated: bool | None = None
    empty: bool = False
    error: str | None = None


@dataclass(frozen=True, kw_only=True)
class GateOutcome(Event):
    """One version of a slot's candidate checked by the parser and the Gate.

    ``step`` is ``"proposed"`` (the first version), ``"fixed"`` (a version
    sent back by the parser or the Gate) or ``"rewritten"`` (a version the
    reviewer sent back). ``item`` is the version as the proposer wrote it.
    ``stage`` is ``"parse"`` or ``"gate"`` for a refusal, with its
    ``reasons`` and ``hints``, and ``None`` when it ``passed``.
    """

    type = "gate"
    round: int
    slot: int
    step: str
    item: object
    passed: bool
    stage: str | None = None
    reasons: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True)
class ReviewVerdict(Event):
    """The reviewer's verdict on a slot's candidate, with its reason; ``rewrite`` when it judged a rewrite."""

    type = "review"
    round: int
    slot: int
    consistent: bool
    reason: str
    rewrite: bool = False


@dataclass(frozen=True, kw_only=True)
class EvaluationStart(Event):
    """The Evaluator starts on round ``round``'s ``slots``: the candidates the reviewer found consistent."""

    type = "evaluation"
    round: int
    slots: list[int]


@dataclass(frozen=True, kw_only=True)
class AdmissionOutcome(Event):
    """A slot's outcome after evaluation.

    ``status`` is ``"accepted"`` or ``"rejected"`` (``stage`` ``"admission"``),
    or ``"failed"`` (``stage`` ``"evaluate"``) when its graph did not
    compile. ``threshold`` is the partial-IC t it had to reach; ``metrics``
    are the Evaluator's (``None`` for a failure).
    """

    type = "admission"
    round: int
    slot: int
    status: str
    stage: str
    reasons: list[str]
    threshold: float | None
    metrics: dict | None


@dataclass(frozen=True, kw_only=True)
class RoundEnd(Event):
    """Round ``round``'s ``trials`` are written, ``accepted`` of them accepted."""

    type = "round_end"
    round: int
    trials: int
    accepted: int


@dataclass(frozen=True, kw_only=True)
class Stopped(Event):
    """The run stopped: ``by`` the stop condition, ``reason`` its sentence, as the state file records them."""

    type = "stop"
    by: str
    reason: str


#: Every event class by its type.
TYPES: dict[str, type[Event]] = {
    cls.type: cls
    for cls in (Start, Resume, Plan, CallStart, CallEnd, GateOutcome, ReviewVerdict, EvaluationStart,
                AdmissionOutcome, RoundEnd, Stopped)
}


class Events:
    """The events file of one run directory, appended to by a campaign; safe to use from several threads."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._writing = threading.Lock()
        self._calls = itertools.count(1)

    def write(self, event: Event) -> None:
        """Append ``event`` as one line."""
        line = {"schema": SCHEMA, "type": event.type, **_plain(asdict(event))}
        text = json.dumps(line, ensure_ascii=False, allow_nan=False) + "\n"
        with self._writing:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as out:
                out.write(text)

    def next_call(self) -> int:
        """A number for the next call, unique in this opening of the run."""
        with self._writing:
            return next(self._calls)


def read(path: str | Path) -> list[Event]:
    """The events in the file at ``path``, in order; an empty list when there is none.

    A line a crash cut short, a line of a later schema and an event of an
    unknown type are skipped; a field this version does not know is ignored.
    """
    try:
        lines = Path(path).read_text().splitlines()
    except FileNotFoundError:
        return []
    events = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        cls = TYPES.get(value.get("type")) if isinstance(value, dict) else None
        if cls is None or value.get("schema") != SCHEMA:
            continue
        known = {f.name for f in fields(cls)}
        try:
            events.append(cls(**{k: v for k, v in value.items() if k in known and k != "time"},
                              time=datetime.fromisoformat(value["time"])))
        except (TypeError, KeyError, ValueError):
            continue
    return events


def reply_facts(reply) -> dict:
    """What a call's ``CallEnd`` says of its reply: tokens, truncation and emptiness, as far as the reply reports them.

    The input and output tokens are the reply's ``usage``; the cache hits,
    the reasoning tokens and the stop reason come from the provider's raw
    reply: an Anthropic message, an OpenAI-compatible chat completion or
    the Claude Code CLI's result.
    """
    usage = reply.usage or {}
    try:
        raw = json.loads(reply.raw) if reply.raw else {}
    except json.JSONDecodeError:
        raw = {}
    raw = raw if isinstance(raw, dict) else {}
    raw_usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    choice = (raw.get("choices") or [{}])[0] if isinstance(raw.get("choices"), list) else {}
    cache_hits = _first(
        raw_usage.get("cache_read_input_tokens"),
        _dig(raw_usage, "prompt_tokens_details", "cached_tokens"),
        raw_usage.get("prompt_cache_hit_tokens"),
    )
    reasoning = _dig(raw_usage, "completion_tokens_details", "reasoning_tokens")
    stop = _first(raw.get("stop_reason"), choice.get("finish_reason") if isinstance(choice, dict) else None)
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cache_hits": cache_hits,
        "reasoning": reasoning,
        "truncated": None if stop is None else stop in ("max_tokens", "length"),
        "empty": not reply.text.strip(),
    }


def _first(*values):
    return next((v for v in values if v is not None), None)


def _dig(value, *keys):
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def _plain(value):
    """``value`` as strict JSON holds it: NaN and infinities as ``None``, times as ISO 8601 text."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value
