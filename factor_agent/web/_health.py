"""What a run is doing now: whether it stopped, its phase, its process, its calls and tokens, the time it has left.

A resumed run keeps the stop of its previous open in its state file until
it stops again. With events, a stop counts only when a stop event follows
the latest start or resume. Without, it counts only when the state file
was written after the run's latest open and no LLM call ended after it.
The process that opened the run last is looked for only on the host it
ran on; on another host it is unknown, never gone.
"""

import socket
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import psutil

from factor_agent import events
from factor_agent.web._files import Replayed, call_files, utc


@dataclass(frozen=True)
class Stop:
    """Why a run stopped, as the campaign recorded it: ``by`` is the stop condition, ``reason`` its sentence."""

    by: str
    reason: str


@dataclass(frozen=True)
class Tokens:
    """Tokens of LLM calls as the providers report them; a count no reply reported is ``None``.

    ``cache_hits`` are input tokens the provider read from its cache, and
    ``reasoning`` output tokens the model spent reasoning.
    """

    input: int | None
    output: int | None
    cache_hits: int | None
    reasoning: int | None


@dataclass(frozen=True)
class RoundTokens:
    """The tokens of a round's calls."""

    round: int
    tokens: Tokens


@dataclass(frozen=True)
class Health:
    """What a run is doing now, as its events file and its process tell.

    ``phase`` is ``"planning"`` (a round is written, or the run was just
    opened, and no call of the next round has been sent), ``"asking"`` (the
    LLM is being asked for the round in progress) or ``"evaluating"`` (the
    Evaluator scores the round's candidates); ``None`` when the run has
    stopped or crashed, or has no events file. ``round`` is the round in
    progress and ``max_rounds`` the round limit, ``None`` when the run has
    stopped or the limit is unknown. ``process`` is ``"running"`` or
    ``"gone"`` when the campaign's process is checked on this host, and
    ``"unknown"`` when the campaign ran on another host or the run has no
    events file; ``crashed`` is true when the process is gone and the run
    recorded no stop. ``last_call`` is when the latest LLM call came back
    (UTC). ``stalled`` is true when the run looks stuck: no call was sent
    or came back for ``STALL_ASKING`` while asking, or the evaluation has
    lasted ``STALL_EVALUATING``. ``calls`` counts the calls of the round in
    progress by role, since the run was last opened; a run without an
    events file has only these counts, from its call files. ``tokens`` are
    the tokens of every call of the run, a round a resume dropped
    included, and ``round_tokens`` the same by round; ``None`` and empty
    without an events file. ``time_left`` estimates the seconds until the
    round or trial limit stops the run, from the mean length of its last
    ``RECENT_ROUNDS`` rounds; ``None`` while no round was timed, and once
    the run has stopped or crashed.
    """

    phase: str | None
    round: int | None
    max_rounds: int | None
    process: str
    crashed: bool
    last_call: datetime | None
    stalled: bool
    calls: dict[str, int]
    time_left: float | None
    tokens: Tokens | None
    round_tokens: tuple[RoundTokens, ...]


#: The events of each phase of a round: the first a phase starts with, the later ones it goes on with.
_PHASES = {
    events.Start: "planning", events.Resume: "planning", events.RoundEnd: "planning",
    events.CallStart: "asking", events.CallEnd: "asking", events.Plan: "asking", events.GateOutcome: "asking",
    events.ReviewVerdict: "asking",
    events.EvaluationStart: "evaluating", events.AdmissionOutcome: "evaluating",
}

#: How long a run asking the LLM may go without a call sent or back before it looks stuck.
STALL_ASKING = timedelta(minutes=15)

#: How long an evaluation may last before the run looks stuck.
STALL_EVALUATING = timedelta(minutes=30)

#: How many of the latest timed rounds the time left is estimated from.
RECENT_ROUNDS = 3


def stop(run_dir: Path, state: dict, opened_at: str | None, replayed: Replayed) -> Stop | None:
    """The state's stop, unless it was recorded before the latest open (see the module's notes)."""
    if not state.get("stop"):
        return None
    if replayed.opened is not None:
        if not any(isinstance(event, events.Stopped) for event in replayed.current):
            return None
        return Stop(by=state["stop"]["by"], reason=state["stop"]["reason"])
    written = (run_dir / "state.json").stat().st_mtime_ns
    if opened_at and written < datetime.fromisoformat(opened_at).timestamp() * 1e9:
        return None
    if any(p.stat().st_mtime_ns > written for p in call_files(run_dir)):
        return None
    return Stop(by=state["stop"]["by"], reason=state["stop"]["reason"])


def health(
    run_dir: Path, state: dict, rounds: int, trials: int, stopped: bool, replayed: Replayed, now: datetime,
) -> Health:
    """The health of a run with ``rounds`` rounds and ``trials`` trials written, ``stopped`` or not, at ``now``."""
    opened = replayed.opened
    process = _process(opened)
    crashed = not stopped and process == "gone"
    phase = None
    if opened is not None and not stopped and not crashed:
        phase = _PHASES.get(type(replayed.current[-1]))
    running = None if stopped else rounds + 1
    ends = [e for e in replayed.all if isinstance(e, events.CallEnd) and e.error is None]
    if opened is None:
        files = sorted(call_files(run_dir))
        last_call = utc(max(p.stat().st_mtime for p in files)) if files else None
        roles = [] if running is None else [p.stem.partition("-")[2] for p in files[state.get("calls", 0):]]
    else:
        last_call = ends[-1].time if ends else None
        roles = [e.role for e in replayed.current if isinstance(e, events.CallEnd) and e.file and e.round == running]
    by_round = {}
    for end in ends:
        by_round.setdefault(end.round, []).append(end)
    return Health(
        phase=phase, round=running, max_rounds=opened.max_rounds if opened else None, process=process,
        crashed=crashed, last_call=last_call, stalled=_stalled(phase, replayed, now),
        calls={role: roles.count(role) for role in dict.fromkeys(roles)},
        time_left=None if running is None or crashed else _time_left(rounds, trials, replayed, now),
        tokens=_tokens(ends) if opened else None,
        round_tokens=tuple(RoundTokens(number, _tokens(calls)) for number, calls in sorted(by_round.items())),
    )


def _time_left(rounds: int, trials: int, replayed: Replayed, now: datetime) -> float | None:
    """The seconds the rounds the limits still allow would take at the recent rounds' pace, the round begun included.

    A round is timed from the end of the round before, or the opening of
    the run when it was the first after it, to its own end.
    """
    opened = replayed.opened
    if opened is None:
        return None
    lengths, since = [], None
    for event in replayed.all:
        if isinstance(event, (events.Start, events.RoundEnd)):
            if isinstance(event, events.RoundEnd) and since is not None:
                lengths.append((event.time - since).total_seconds())
            since = event.time
    if not lengths:
        return None
    left = min(opened.max_rounds - rounds, (opened.max_trials - trials) // max(opened.candidates, 1))
    pace = sum(lengths[-RECENT_ROUNDS:]) / len(lengths[-RECENT_ROUNDS:])
    return max(0.0, pace * left - (now - since).total_seconds())


def _tokens(ends: list[events.CallEnd]) -> Tokens:
    """The tokens of the calls that ``ends`` end, each count ``None`` when no call reported it."""
    def total(key):
        counts = [getattr(end, key) for end in ends if getattr(end, key) is not None]
        return sum(counts) if counts else None
    return Tokens(total("input_tokens"), total("output_tokens"), total("cache_hits"), total("reasoning"))


def _stalled(phase: str | None, replayed: Replayed, now: datetime) -> bool:
    """Whether a run asking has had no call sent or back for ``STALL_ASKING``, or its evaluation lasted ``STALL_EVALUATING``."""
    if phase == "asking":
        # The opening counts as the first sign of life, a failed call's end as one too.
        activity = [e.time for e in replayed.current if isinstance(e, (events.Start, events.CallStart, events.CallEnd))]
        return now - activity[-1] > STALL_ASKING
    if phase == "evaluating":
        started = [e.time for e in replayed.current if isinstance(e, events.EvaluationStart)]
        return now - started[-1] > STALL_EVALUATING
    return False


def _process(opened: events.Start | None) -> str:
    """Whether the process that opened the run last still runs, when it ran on this host."""
    if opened is None or opened.host != socket.gethostname():
        return "unknown"
    try:
        started = psutil.Process(opened.pid).create_time()
    except psutil.NoSuchProcess:
        return "gone"
    except psutil.Error:
        return "unknown"
    # A process started after the run was opened has only been given the same id.
    return "running" if started <= opened.time.timestamp() + 1 else "gone"
