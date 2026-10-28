"""Read a run directory as the frontend shows it, without writing to it.

``list_runs(root)`` finds every run directory under a folder: a directory
holding a ``snapshot.json``, which a campaign writes on its first start,
before its first round. ``RunView.read(run_dir)`` reads one run. Callers
never see the run directory's files; this module and its private modules
alone know them, and read them while a campaign may be writing them:

- The state file decides how many rounds are written. The campaign appends
  a round's trials to the ledger before it rewrites the state, so ledger
  rounds beyond the state's count are not read until the state catches up,
  and a ledger line a crash cut short is skipped.
- NaN and infinite values in the ledger (a correlation with an empty pool,
  say) read as ``None``, so a view converts to strict JSON.
- The run list reads of each run only what it shows; a run whose files
  cannot be read is listed as unreadable, with the reason.
- The events file (``factor_agent.events``) tells what the run is doing:
  its phase, its calls and their tokens, its process, and the settings it
  was opened with. A run a campaign without events wrote has none of this
  but the calls of its round in progress, counted from its call files.
- The round in progress is replayed from the events since the run was
  last opened: its calls, and a pending trial per slot of its plan. A
  round a resume dropped left its events before the resume, so it is not
  among them. A run without events has no round in progress to show.
- An LLM call is read by the number of its call file, from the file
  itself: tokens, truncation and the model's reasoning are the provider's
  reply's; what it concerned (round, slots) only the events file says.
- A resumed run keeps the stop of its previous open in its state file until
  it stops again. With events, a stop counts only when a stop event follows
  the latest start or resume. Without, it counts only when the state file
  was written after the run's latest open and no LLM call ended after it: a
  resumed run that has asked the LLM since, or has written no state since
  it was opened, has not stopped.
- A round's t threshold (one per round, since a round's candidates are
  judged at the same trial count) is the one its admission events give.
  Without them it is the one its refusals quote, carried over from the
  round before when the round quotes none; an accepted factor says which.
  The correlation limit is the one the run was last opened with, or without
  events the one a refusal quotes. Either is ``None`` while nothing gives it.
- The process that opened the run last is looked for only on the host it
  ran on; on another host it is unknown, never gone.
- An Acceptance report is ``acceptance/<segment>[_<digest>].json``; one that
  records opens of the hold-out is a hold-out report. The opens are the hold-out
  lock's. A report written before the contribution table or the selection
  flag existed reads with ``None`` for them: nothing it lacks is made up.
- A collision is read from a refusal's text: the Gate's rewrite, refinement
  of parameters only or shared subtree is a code collision with each factor
  it names, admission's correlation above the limit a ranking collision. The
  Gate's refusal of a duplicate of the run's own earlier candidate is not.
- The diagnosis is read from the trials by rules private to this package,
  their thresholds fixed there, each finding counting trials (not attempts:
  a trial is counted once whichever of its versions met the problem).
- A trial the Evaluator scored is placed on the admission map by its
  partial-IC t in its direction (the sign of its training IC, as admission
  reads it) and its largest absolute correlation with an existing factor.
  The trials it never scored are grouped by the stage that refused them and
  the reason: reasons that differ only in their numbers, or in the reply or
  code they quote, are one reason.
- The memory's versions are the distiller's files ``memory/round_NN.md``,
  each written before round ``NN``, so after round ``NN - 1``. A version
  after the first is compared with the one before it line by line.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from factor_agent import events
from factor_agent.web._acceptance import AcceptanceReports, Comparison, Contribution, HoldoutOpen, acceptance
from factor_agent.web._diagnosis import Finding, diagnosis
from factor_agent.web._files import Replayed, read_json, utc, written
from factor_agent.web._health import (
    RECENT_ROUNDS, STALL_ASKING, STALL_EVALUATING, Health, RoundTokens, Stop, Tokens, health, stop,
)
from factor_agent.web._ledger import Collision, Neighbour, Trial, correlation_limit, read_trials, statuses
from factor_agent.web._memory import MemoryLine, MemoryVersion, memory
from factor_agent.web._results import Factor, factors
from factor_agent.web._round import Check, LLMCall, PendingTrial, RoundCall, RunningRound, read_call, running
from factor_agent.web._trials import Placed, Refusal, TrialRef, trials_page

__all__ = [
    "RECENT_ROUNDS", "STALL_ASKING", "STALL_EVALUATING", "AcceptanceReports", "Check", "Collision", "Comparison",
    "Contribution", "Factor", "Finding", "Health", "HoldoutOpen", "LLMCall", "MemoryLine", "MemoryVersion", "Neighbour",
    "PendingTrial", "Placed", "Refusal", "RoundCall", "RoundTokens", "RunSummary", "RunView", "RunningRound", "Stop",
    "Tokens", "Trial", "TrialRef", "find_run", "list_runs", "read_call", "run_version",
]


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
    ``unreadable`` says which file of the run could not be read and why,
    for a run the list could not read (its counts are then 0); ``None``
    for any other.
    """

    name: str
    code_version: str
    rounds: int
    trials: int
    accepted: int
    stop: Stop | None
    updated: datetime
    unreadable: str | None = None


@dataclass(frozen=True)
class RunView:
    """A read-only snapshot of one run: its summary, the trials of its written rounds, its accepted factors and its memory.

    ``factors`` holds the accepted factors in the order they were accepted
    (by round, then slot). ``correlation_limit`` is the largest absolute
    correlation with an existing factor that admission allows: the one the
    run was last opened with, or for a run without events the one its
    latest refusal for it quotes, ``None`` while nothing gives it.
    ``evaluated`` places every trial the Evaluator scored, in round and slot
    order, and ``threshold`` is the t threshold of the latest round (see the
    module's notes), ``None`` while nothing gives it.
    ``never_evaluated`` groups the trials it never scored, by stage (parse,
    Gate, review, then evaluation) and then by the first round of each reason.
    ``memory`` holds the versions of the memory by the round after which
    they were written, the earliest first; it is empty while the distiller
    has written none. ``acceptance`` holds the run's Acceptance reports and
    hold-out opens. ``diagnosis`` holds at most three findings, the one
    concerning the most trials first; it is empty when no rule holds.
    ``health`` is what the run is doing now, and ``running`` the round in
    progress: ``None`` when the run has stopped, or has no events file (its
    health then counts the round's calls by role). ``now`` is the time the
    view describes (UTC).
    """

    summary: RunSummary
    trials: tuple[Trial, ...]
    factors: tuple[Factor, ...]
    correlation_limit: float | None
    evaluated: tuple[Placed, ...]
    threshold: float | None
    never_evaluated: tuple[Refusal, ...]
    memory: tuple[MemoryVersion, ...]
    acceptance: AcceptanceReports
    diagnosis: tuple[Finding, ...]
    health: Health
    running: RunningRound | None
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
        state = read_json(run_dir / "state.json") or {}
        trials = read_trials(run_dir / "ledger.jsonl", state.get("rounds", 0))
        replayed = Replayed.read(run_dir)
        summary = _summary(run_dir, name or run_dir.name, state, replayed, [t.status for t in trials])
        page = trials_page(trials, replayed)
        return cls(
            summary=summary, trials=trials, factors=factors(trials, replayed),
            correlation_limit=correlation_limit(trials, replayed), evaluated=page.evaluated, threshold=page.threshold,
            never_evaluated=page.never_evaluated, memory=memory(run_dir / "memory"), acceptance=acceptance(run_dir),
            diagnosis=diagnosis(trials),
            health=health(run_dir, state, summary.rounds, summary.trials, summary.stop is not None, replayed, now),
            running=None if summary.stop else running(summary.rounds + 1, replayed), now=now,
        )


def _summary(run_dir: Path, name: str, state: dict, replayed: Replayed, written_statuses: list[str]) -> RunSummary:
    """The summary of a run whose state file holds ``state`` and whose written trials have ``written_statuses``."""
    versions = (read_json(run_dir / "snapshot.json") or {}).get("versions") or [{}]
    return RunSummary(
        name=name, code_version=versions[-1].get("factor_agent") or "unknown", rounds=state.get("rounds", 0),
        trials=len(written_statuses), accepted=written_statuses.count("accepted"),
        stop=stop(run_dir, state, versions[-1].get("opened_at"), replayed),
        updated=utc(max(p.stat().st_mtime for p in written(run_dir))),
    )


class _Unreadable(Exception):
    """A run file the run list cannot read; the message names it and says why."""


@contextmanager
def _reading(path: Path) -> Iterator[None]:
    """Turn a failure to read ``path`` (missing while listed, not JSON, of another shape) into ``_Unreadable``."""
    try:
        yield
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise _Unreadable(f"{path.name} cannot be read: {exc}") from exc


def _listed(run_dir: Path, name: str) -> RunSummary:
    """The run list's summary of a run: its state, ledger statuses and events only, or unreadable."""
    try:
        with _reading(run_dir / "state.json"):
            state = read_json(run_dir / "state.json") or {}
            rounds = state.get("rounds", 0)
        with _reading(run_dir / "ledger.jsonl"):
            found = statuses(run_dir / "ledger.jsonl", rounds)
        with _reading(run_dir / events.FILE):
            replayed = Replayed.read(run_dir)
        with _reading(run_dir / "snapshot.json"):
            return _summary(run_dir, name, state, replayed, found)
    except _Unreadable as exc:
        return RunSummary(name=name, code_version="unknown", rounds=0, trials=0, accepted=0, stop=None,
                          updated=utc((run_dir / "snapshot.json").stat().st_mtime), unreadable=str(exc))


def run_version(run_dir: str | Path) -> str:
    """A text that changes whenever the run in ``run_dir`` writes, cheap to read: a page polls it before reading the run.

    It is made of the modification times and sizes of the files a campaign
    writes while it runs, and the number of its call files.
    """
    run_dir = Path(run_dir)
    parts = []
    for name in ("snapshot.json", "state.json", "ledger.jsonl", events.FILE, "llm_log", "memory", "acceptance",
                 "holdout.lock"):
        try:
            stat = (run_dir / name).stat()
        except FileNotFoundError:
            parts.append("-")
            continue
        parts.append(f"{stat.st_mtime_ns}:{stat.st_size}")
    return " ".join(parts)


def list_runs(root: str | Path) -> list[RunSummary]:
    """The runs under ``root``, the latest written first.

    Each run is read only as far as its summary needs. A run whose files
    cannot be read is listed with ``unreadable`` saying why, so one broken
    run does not hide the others.

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
    summaries = [_listed(d, name) for name, d in _run_dirs(root).items()]
    return sorted(summaries, key=lambda s: s.updated, reverse=True)


def find_run(root: str | Path, name: str) -> Path | None:
    """The run directory ``list_runs(root)`` names ``name``, ``None`` when it names none so."""
    return _run_dirs(Path(root)).get(name)


def _run_dirs(root: Path) -> dict[str, Path]:
    """The run directories under ``root`` by their names."""
    found = sorted({p.parent for p in root.rglob("snapshot.json") if p.is_file()})
    return {d.relative_to(root).as_posix() if d != root else root.name: d for d in found}
