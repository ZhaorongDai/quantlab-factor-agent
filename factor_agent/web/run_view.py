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
- A resumed run keeps the stop of its previous open in its state file until
  it stops again. A stop counts only when the state file was written after
  the run's latest open and no LLM call ended after it: a resumed run that
  has asked the LLM since, or has written no state since it was opened, has
  not stopped.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


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
class RunView:
    """A read-only snapshot of one run: its summary and the trials of its written rounds.

    ``now`` is the time the view describes (UTC).
    """

    summary: RunSummary
    trials: tuple[Trial, ...]
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
        state = _json(run_dir / "state.json") or {}
        rounds = state.get("rounds", 0)
        trials = tuple(_trials(run_dir / "ledger.jsonl", rounds))
        versions = (_json(run_dir / "snapshot.json") or {}).get("versions") or [{}]
        summary = RunSummary(
            name=name or run_dir.name,
            code_version=versions[-1].get("factor_agent") or "unknown",
            rounds=rounds,
            trials=len(trials),
            accepted=sum(t.status == "accepted" for t in trials),
            stop=_stop(run_dir, state, versions[-1].get("opened_at")),
            updated=_time(max(p.stat().st_mtime for p in _written(run_dir))),
        )
        return cls(summary=summary, trials=trials, now=now or datetime.now(timezone.utc))


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
    found = sorted({p.parent for p in root.rglob("snapshot.json") if p.is_file()})
    summaries = [RunView.read(d, name=d.relative_to(root).as_posix() if d != root else root.name).summary for d in found]
    return sorted(summaries, key=lambda s: s.updated, reverse=True)


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


def _calls(run_dir: Path) -> list[Path]:
    return list((run_dir / "llm_log").glob("*.json"))


def _written(run_dir: Path) -> list[Path]:
    """The run's files that change while it runs."""
    files = [run_dir / name for name in ("snapshot.json", "state.json", "ledger.jsonl")]
    return [p for p in files if p.exists()] + _calls(run_dir)


def _stop(run_dir: Path, state: dict, opened_at: str | None) -> Stop | None:
    """The state's stop, unless it was recorded before the latest open (see the module's notes)."""
    if not state.get("stop"):
        return None
    written = (run_dir / "state.json").stat().st_mtime_ns
    if opened_at and written < datetime.fromisoformat(opened_at).timestamp() * 1e9:
        return None
    if any(p.stat().st_mtime_ns > written for p in _calls(run_dir)):
        return None
    return Stop(by=state["stop"]["by"], reason=state["stop"]["reason"])


def _time(seconds: float) -> datetime:
    return datetime.fromtimestamp(seconds, timezone.utc)
