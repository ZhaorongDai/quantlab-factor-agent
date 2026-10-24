"""The trial ledger: one JSON line per candidate, accepted or not."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TrialRecord:
    """One candidate's outcome.

    ``status`` is ``"accepted"``, ``"rejected"`` (refused by the reviewer,
    unevaluated, or evaluated and refused by admission) or ``"failed"``
    (never evaluated: refused by the parser or the Gate after every retry,
    or its graph did not compile). Only an evaluated trial, one with
    ``metrics``, counts toward the evaluated-trial count. ``stage`` is
    where it ended: ``"parse"``, ``"gate"``, ``"review"``, ``"evaluate"``
    or ``"admission"``. ``candidate`` is the last version the proposer
    wrote, ``attempts`` every refused version before it (``item``,
    ``stage``, ``reasons``). ``direction`` is the planned direction the
    candidate was written for (``mechanism_type``, ``action``, ``parent``,
    ``direction``). ``stats`` holds the Gate's statistics, ``graph`` the
    graph it built (``Graph.to_json``) and ``metrics`` the evaluation, once
    there are any.
    """

    round: int
    candidate: dict
    status: str
    stage: str
    reasons: list[str]
    fingerprint: str | None = None
    metrics: dict | None = None
    stats: dict | None = None
    attempts: list[dict] = field(default_factory=list)
    direction: dict | None = None
    graph: list | None = None

    @property
    def evaluated(self) -> bool:
        """Whether the Evaluator scored the candidate: it counts toward the evaluated-trial count."""
        return self.metrics is not None


class Ledger:
    """Append and read the trial records of a run directory."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, records: list[TrialRecord]) -> None:
        """Append ``records``; the file and its directory are created when missing."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as out:
            for record in records:
                out.write(json.dumps(asdict(record)) + "\n")

    def keep(self, rounds: int) -> None:
        """Drop the records of every round after ``rounds``, such as a round a crash left half written."""
        if not self.path.exists():
            return
        lines = [line for line in self.path.read_text().splitlines() if line]
        kept = []
        for line in lines:
            try:
                record = TrialRecord(**json.loads(line))
            except (json.JSONDecodeError, TypeError):  # a line a crash cut short belongs to an unfinished round
                continue
            if record.round <= rounds:
                kept.append(record)
        if len(kept) == len(lines):
            return
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(asdict(r)) + "\n" for r in kept))
        tmp.replace(self.path)

    def records(self) -> list[TrialRecord]:
        """Every record written so far, in order."""
        if not self.path.exists():
            return []
        return [TrialRecord(**json.loads(line)) for line in self.path.read_text().splitlines() if line]
