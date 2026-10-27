"""Read a run directory as the frontend shows it, without writing to it.

``list_runs(root)`` finds every run directory under a folder: a directory
holding a ``snapshot.json``, which a campaign writes on its first start,
before its first round. ``RunView.read(run_dir)`` reads one run. Callers
never see the run directory's files; this module alone knows them, and
reads them while a campaign may be writing them:

- The state file decides how many rounds are written. The campaign appends
  a round's trials to the ledger before it rewrites the state, so ledger
  rounds beyond the state's count are not read until the state catches up,
  and a ledger line a crash cut short is skipped.
- NaN and infinite values in the ledger (a correlation with an empty pool,
  say) read as ``None``, so a view converts to strict JSON.
- The events file (``factor_agent.events``) tells what the run is doing:
  its phase, its calls and their tokens, its process, and the settings it
  was opened with. A run a campaign without events wrote has none of this
  but the calls of its round in progress, counted from its call files.
- A resumed run keeps the stop of its previous open in its state file until
  it stops again. With events, a stop counts only when a stop event follows
  the latest start or resume. Without, it counts only when the state file
  was written after the run's latest open and no LLM call ended after it: a
  resumed run that has asked the LLM since, or has written no state since
  it was opened, has not stopped.
- A round's t threshold (one per round, since a round's candidates are
  judged at the same trial count) is the one its admission events give.
  Without them it is the one its refusals quote, carried over from the
  round before when the round quotes none. The correlation limit is the
  one the run was last opened with, or without events the one a refusal
  quotes. Either is ``None`` while nothing gives it.
- The process that opened the run last is looked for only on the host it
  ran on; on another host it is unknown, never gone.
- The memory's versions are the distiller's files ``memory/round_NN.md``,
  each written before round ``NN``, so after round ``NN - 1``. A version
  after the first is compared with the one before it line by line.
"""

import difflib
import json
import re
import socket
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psutil

from factor_agent import events


@dataclass(frozen=True)
class Stop:
    """Why a run stopped, as the campaign recorded it: ``by`` is the stop condition, ``reason`` its sentence."""

    by: str
    reason: str


@dataclass(frozen=True)
class Trial:
    """One trial of a written round, as the ledger holds it, with NaN read as ``None``.

    ``slot`` is the trial's place in its round, from 0, in the order of the
    round's research directions. ``status``, ``stage``, ``reasons``,
    ``candidate``, ``direction``, ``attempts``, ``stats`` and ``metrics`` are
    the trial record's (``factor_agent.ledger.TrialRecord``); ``metrics`` is
    ``None`` for a trial the Evaluator never scored.
    """

    round: int
    slot: int
    status: str
    stage: str
    reasons: tuple[str, ...]
    candidate: dict
    direction: dict | None
    attempts: tuple[dict, ...]
    stats: dict | None
    metrics: dict | None


@dataclass(frozen=True)
class Neighbour:
    """The factor closest to an accepted one in ranking, and their correlation.

    ``label`` names it as the Evaluator does. ``name`` is how a page names
    it: a factor accepted earlier in the run as ``"round N: "`` and the first
    words of its description, any other by its label. ``accepted_round`` is
    N for a factor accepted in the run, ``None`` for a reference or known
    factor. ``correlation`` is the raw cross-sectional correlation, which
    admission compares in absolute value with its limit.
    """

    label: str
    name: str
    accepted_round: int | None
    correlation: float | None


@dataclass(frozen=True)
class Factor:
    """An accepted factor, written up with the facts that say how far to trust it.

    ``label`` is the name the Evaluator and Acceptance give it,
    ``"<description> (accepted in round N)"``. ``round``, ``slot`` and
    ``mechanism_type`` place it in the run; ``hypothesis``, ``description``
    and ``code`` are its candidate's. ``factor_direction`` is +1 when a
    higher value went with a higher training return, -1 when with a lower
    one. ``ic_mean``, ``ic_t``, ``partial_ic_mean`` and ``partial_ic_t`` are
    the training IC and the partial IC and their Newey-West t, in the
    factor's direction, as admission judged them; ``metrics`` holds every
    statistic as the ledger records it, the raw values among them.
    ``decile_returns`` and ``coverage`` are the Evaluator's.
    """

    label: str
    round: int
    slot: int
    mechanism_type: str | None
    hypothesis: str
    description: str
    code: str
    factor_direction: int
    ic_mean: float | None
    ic_t: float | None
    partial_ic_mean: float | None
    partial_ic_t: float | None
    threshold: float | None
    monotonicity: float | None
    nearest: Neighbour | None
    nearest_known: Neighbour | None
    decile_returns: tuple[float | None, ...]
    coverage: float | None
    metrics: dict


@dataclass(frozen=True)
class MemoryLine:
    """One line of a memory version compared with the version before it.

    ``change`` is ``"kept"``, ``"added"`` (not in the version before) or
    ``"removed"`` (in the version before, not in this one).
    """

    change: str
    line: str


@dataclass(frozen=True)
class MemoryVersion:
    """One version of the memory the distiller wrote.

    ``round`` is the round after which it was written. ``changes`` holds the
    lines of this version and of the one before, in order, each marked kept,
    added or removed; it is ``None`` for the run's first version.
    """

    round: int
    text: str
    changes: tuple[MemoryLine, ...] | None


@dataclass(frozen=True)
class RunSummary:
    """What the run list shows of a run.

    ``name`` is the run directory's path relative to the listed folder
    (the folder's own name when the folder is the run directory).
    ``code_version`` is the commit of this package the run was last opened
    with, ``"unknown"`` when it could not be read. ``rounds`` counts the
    written rounds, ``trials`` and ``accepted`` their trials and accepted
    ones. ``stop`` is ``None`` while the run has not stopped, or crashed
    without stopping. ``updated`` is the time of the run's last write (UTC).
    """

    name: str
    code_version: str
    rounds: int
    trials: int
    accepted: int
    stop: Stop | None
    updated: datetime


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
    tokens: "Tokens | None"
    round_tokens: tuple["RoundTokens", ...]


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
class RunView:
    """A read-only snapshot of one run: its summary, the trials of its written rounds, its accepted factors and its memory.

    ``factors`` holds the accepted factors in the order they were accepted
    (by round, then slot). ``correlation_limit`` is the largest absolute
    correlation with an existing factor that admission allows: the one the
    run was last opened with, or for a run without events the one its
    latest refusal for it quotes, ``None`` while nothing gives it.
    ``memory`` holds the versions of the memory by the round after which
    they were written, the earliest first; it is empty while the distiller
    has written none. ``health`` is what the run is doing now. ``now`` is
    the time the view describes (UTC).
    """

    summary: RunSummary
    trials: tuple[Trial, ...]
    factors: tuple[Factor, ...]
    correlation_limit: float | None
    memory: tuple[MemoryVersion, ...]
    health: Health
    now: datetime

    @classmethod
    def read(cls, run_dir: str | Path, now: datetime | None = None, name: str | None = None) -> "RunView":
        """Read the run in ``run_dir``.

        Parameters
        ----------
        run_dir : str or Path
            A run directory: one holding a ``snapshot.json``.
        now : datetime, optional
            The time the view describes; the current time by default.
        name : str, optional
            The run's name in the view; the directory's name by default.

        Returns
        -------
        RunView

        Raises
        ------
        FileNotFoundError
            When ``run_dir`` holds no ``snapshot.json``.
        """
        run_dir = Path(run_dir)
        if not (run_dir / "snapshot.json").is_file():
            raise FileNotFoundError(f"web: {run_dir} is not a run directory: it holds no snapshot.json")
        now = now or datetime.now(timezone.utc)
        state = _json(run_dir / "state.json") or {}
        rounds = state.get("rounds", 0)
        trials = tuple(_trials(run_dir / "ledger.jsonl", rounds))
        versions = (_json(run_dir / "snapshot.json") or {}).get("versions") or [{}]
        replayed = _Events(events.read(run_dir / events.FILE))
        summary = RunSummary(
            name=name or run_dir.name,
            code_version=versions[-1].get("factor_agent") or "unknown",
            rounds=rounds,
            trials=len(trials),
            accepted=sum(t.status == "accepted" for t in trials),
            stop=_stop(run_dir, state, versions[-1].get("opened_at"), replayed),
            updated=_time(max(p.stat().st_mtime for p in _written(run_dir))),
        )
        return cls(summary=summary, trials=trials, factors=_factors(trials, replayed),
                   correlation_limit=_correlation_limit(trials, replayed),
                   memory=_memory(run_dir / "memory"), health=_health(run_dir, state, summary, replayed, now),
                   now=now)


def run_version(run_dir: str | Path) -> str:
    """A text that changes whenever the run in ``run_dir`` writes, cheap to read: a page polls it before reading the run.

    It is made of the modification times and sizes of the files a campaign
    writes while it runs, and the number of its call files.
    """
    run_dir = Path(run_dir)
    parts = []
    for name in ("snapshot.json", "state.json", "ledger.jsonl", events.FILE, "llm_log", "memory"):
        try:
            stat = (run_dir / name).stat()
        except FileNotFoundError:
            parts.append("-")
            continue
        parts.append(f"{stat.st_mtime_ns}:{stat.st_size}")
    return " ".join(parts)


def list_runs(root: str | Path) -> list[RunSummary]:
    """The runs under ``root``, the latest written first.

    Parameters
    ----------
    root : str or Path
        A folder of run directories, at any depth, or a run directory itself.

    Returns
    -------
    list of RunSummary
        One per run directory, named by its path relative to ``root``.
    """
    root = Path(root)
    summaries = [RunView.read(d, name=name).summary for name, d in _run_dirs(root).items()]
    return sorted(summaries, key=lambda s: s.updated, reverse=True)


def find_run(root: str | Path, name: str) -> Path | None:
    """The run directory ``list_runs(root)`` names ``name``, ``None`` when it names none so."""
    return _run_dirs(Path(root)).get(name)


def _run_dirs(root: Path) -> dict[str, Path]:
    """The run directories under ``root`` by their names."""
    found = sorted({p.parent for p in root.rglob("snapshot.json") if p.is_file()})
    return {d.relative_to(root).as_posix() if d != root else root.name: d for d in found}


def _loads(text: str):
    # The ledger and the Acceptance reports write NaN where a value is undefined.
    return json.loads(text, parse_constant=lambda _: None)


def _json(path: Path):
    try:
        return _loads(path.read_text())
    except FileNotFoundError:
        return None


def _trials(ledger: Path, rounds: int) -> list[Trial]:
    """The trials of the first ``rounds`` rounds; later rounds and a line a crash cut short are skipped."""
    try:
        lines = ledger.read_text().splitlines()
    except FileNotFoundError:
        return []
    trials, slots = [], {}
    for line in lines:
        try:
            record = _loads(line)
        except json.JSONDecodeError:
            continue
        if record["round"] > rounds:
            continue
        slot = slots[record["round"]] = slots.get(record["round"], -1) + 1
        trials.append(Trial(
            round=record["round"], slot=slot, status=record["status"], stage=record["stage"],
            reasons=tuple(record["reasons"]), candidate=record["candidate"], direction=record.get("direction"),
            attempts=tuple(record.get("attempts") or ()), stats=record.get("stats"), metrics=record.get("metrics"),
        ))
    return trials


#: The partial-IC t threshold an admission refusal quotes.
_THRESHOLD = re.compile(r"is below the threshold ([\d.]+) at \d+ evaluated trial")


#: The correlation limit an admission refusal quotes.
_LIMIT = re.compile(r"^cross-sectional correlation .* is above ([\d.]+) in absolute value$", re.DOTALL)


def _correlation_limit(trials: tuple[Trial, ...], replayed: "_Events") -> float | None:
    """The correlation limit the run was last opened with, else the latest refusal quoting it, else ``None``."""
    if replayed.opened is not None:
        return replayed.opened.max_correlation
    quoted = [found.group(1) for trial in trials for reason in trial.reasons if (found := _LIMIT.match(reason))]
    return float(quoted[-1]) if quoted else None


def _thresholds(trials: tuple[Trial, ...], replayed: "_Events") -> dict[int, float | None]:
    """Each round's threshold as its admission events give it, else as its refusals quote it, else the previous round's.

    The admission events of a round a resume dropped come before those of
    the round run again under the same number, so the later ones count.
    """
    quoted = {}
    for trial in trials:
        for reason in trial.reasons:
            if found := _THRESHOLD.search(reason):
                quoted[trial.round] = float(found.group(1))
    for event in replayed.all:
        if isinstance(event, events.AdmissionOutcome) and event.threshold is not None:
            quoted[event.round] = event.threshold
    thresholds, last = {}, None
    for number in sorted({trial.round for trial in trials}):
        thresholds[number] = last = quoted.get(number, last)
    return thresholds


#: An accepted factor's label: its description and the round that accepted it.
_ACCEPTED = re.compile(r"^(.*) \(accepted in round (\d+)\)$", re.DOTALL)

#: How many words of an earlier accepted factor's description name it.
_FIRST_WORDS = 6


def _neighbour(label: str | None, correlation: float | None) -> Neighbour | None:
    """The nearest factor the Evaluator recorded, ``None`` when there was none to compare with."""
    if not label:
        return None
    if accepted := _ACCEPTED.match(label):
        words = accepted.group(1).split()
        first = " ".join(words[:_FIRST_WORDS]) + ("…" if len(words) > _FIRST_WORDS else "")
        number = int(accepted.group(2))
        return Neighbour(label=label, name=f"round {number}: {first}", accepted_round=number, correlation=correlation)
    return Neighbour(label=label, name=label, accepted_round=None, correlation=correlation)


def _ranks(values: list[float]) -> list[float]:
    """Each value's rank from 1, ties given their mean rank."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for i in order[start:end + 1]:
            ranks[i] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def _monotonicity(deciles: tuple[float | None, ...], sign: int) -> float | None:
    """The Spearman correlation of the decile number with the decile returns, signed by ``sign``, as admission's."""
    if len(deciles) < 2 or any(r is None for r in deciles) or len(set(deciles)) == 1:
        return None
    ranks, n = _ranks(list(deciles)), len(deciles)
    mean = (n + 1) / 2
    spread = sum((r - mean) ** 2 for r in ranks) * sum((i + 1 - mean) ** 2 for i in range(n))
    return sign * sum((r - mean) * (i + 1 - mean) for i, r in enumerate(ranks)) / spread ** 0.5


def _factors(trials: tuple[Trial, ...], replayed: "_Events") -> tuple[Factor, ...]:
    thresholds = _thresholds(trials, replayed)
    factors = []
    for trial in trials:
        if trial.status != "accepted":
            continue
        metrics, candidate = trial.metrics, trial.candidate
        sign = 1 if (metrics.get("ic_mean") or 0) > 0 else -1
        deciles = tuple(metrics.get("decile_returns") or ())
        oriented = {key: None if metrics.get(key) is None else sign * metrics[key]
                    for key in ("ic_mean", "ic_t", "partial_ic_mean", "partial_ic_t")}
        factors.append(Factor(
            label=f"{candidate['description']} (accepted in round {trial.round})",
            round=trial.round, slot=trial.slot, mechanism_type=(trial.direction or {}).get("mechanism_type"),
            hypothesis=candidate.get("hypothesis", ""), description=candidate["description"], code=candidate["code"],
            factor_direction=sign, **oriented, threshold=thresholds[trial.round],
            monotonicity=_monotonicity(deciles, sign),
            nearest=_neighbour(metrics.get("nearest"), metrics.get("nearest_correlation")),
            nearest_known=_neighbour(metrics.get("nearest_known"), metrics.get("nearest_known_correlation")),
            decile_returns=deciles, coverage=metrics.get("coverage"), metrics=metrics,
        ))
    return tuple(factors)


def _memory(folder: Path) -> tuple[MemoryVersion, ...]:
    """The memory's versions, the earliest first, each but the first compared with the one before."""
    files = {}
    for path in folder.glob("round_*.md"):
        if match := re.fullmatch(r"round_(\d+)\.md", path.name):
            files[int(match[1]) - 1] = path.read_text().rstrip("\n")
    versions, before = [], None
    for number, text in sorted(files.items()):
        versions.append(MemoryVersion(round=number, text=text, changes=None if before is None else _changes(before, text)))
        before = text
    return tuple(versions)


def _changes(before: str, after: str) -> tuple[MemoryLine, ...]:
    old, new = before.splitlines(), after.splitlines()
    lines = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes():
        if tag == "equal":
            lines += [MemoryLine("kept", line) for line in new[j1:j2]]
            continue
        lines += [MemoryLine("removed", line) for line in old[i1:i2]]
        lines += [MemoryLine("added", line) for line in new[j1:j2]]
    return tuple(lines)


def _calls(run_dir: Path) -> list[Path]:
    return list((run_dir / "llm_log").glob("*.json"))


def _written(run_dir: Path) -> list[Path]:
    """The run's files that change while it runs."""
    files = [run_dir / name for name in ("snapshot.json", "state.json", "ledger.jsonl", events.FILE)]
    return [p for p in files if p.exists()] + _calls(run_dir)


class _Events:
    """A run's events, and those since its latest opening (``current``), the opening first."""

    def __init__(self, replayed: list[events.Event]):
        self.all = replayed
        openings = [i for i, event in enumerate(replayed) if isinstance(event, events.Start)]
        self.current = replayed[openings[-1]:] if openings else []
        self.opened: events.Start | None = self.current[0] if self.current else None


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


def _health(run_dir: Path, state: dict, summary: RunSummary, replayed: _Events, now: datetime) -> Health:
    opened = replayed.opened
    process = _process(opened)
    crashed = summary.stop is None and process == "gone"
    phase = None
    if opened is not None and summary.stop is None and not crashed:
        phase = _PHASES.get(type(replayed.current[-1]))
    running = None if summary.stop else summary.rounds + 1
    ends = [e for e in replayed.all if isinstance(e, events.CallEnd) and e.error is None]
    if opened is None:
        files = sorted(_calls(run_dir))
        last_call = _time(max(p.stat().st_mtime for p in files)) if files else None
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
        time_left=None if running is None or crashed else _time_left(summary, replayed, now),
        tokens=_tokens(ends) if opened else None,
        round_tokens=tuple(RoundTokens(number, _tokens(calls)) for number, calls in sorted(by_round.items())),
    )


#: How many of the latest timed rounds the time left is estimated from.
RECENT_ROUNDS = 3


def _time_left(summary: RunSummary, replayed: _Events, now: datetime) -> float | None:
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
    rounds = min(opened.max_rounds - summary.rounds, (opened.max_trials - summary.trials) // max(opened.candidates, 1))
    pace = sum(lengths[-RECENT_ROUNDS:]) / len(lengths[-RECENT_ROUNDS:])
    return max(0.0, pace * rounds - (now - since).total_seconds())


def _tokens(ends: list[events.CallEnd]) -> Tokens:
    """The tokens of the calls that ``ends`` end, each count ``None`` when no call reported it."""
    def total(key):
        counts = [getattr(end, key) for end in ends if getattr(end, key) is not None]
        return sum(counts) if counts else None
    return Tokens(total("input_tokens"), total("output_tokens"), total("cache_hits"), total("reasoning"))


def _stalled(phase: str | None, replayed: _Events, now: datetime) -> bool:
    if phase == "asking":
        calls = [e.time for e in replayed.current if isinstance(e, (events.CallStart, events.Start))]
        return now - calls[-1] > STALL_ASKING
    if phase == "evaluating":
        started = [e.time for e in replayed.current if isinstance(e, events.EvaluationStart)]
        return now - started[-1] > STALL_EVALUATING
    return False


def _process(opened: "events.Start | None") -> str:
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


def _stop(run_dir: Path, state: dict, opened_at: str | None, replayed: _Events) -> Stop | None:
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
    if any(p.stat().st_mtime_ns > written for p in _calls(run_dir)):
        return None
    return Stop(by=state["stop"]["by"], reason=state["stop"]["reason"])


def _time(seconds: float) -> datetime:
    return datetime.fromtimestamp(seconds, timezone.utc)
