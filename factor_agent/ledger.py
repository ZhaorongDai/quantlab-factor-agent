"""The trial ledger: one JSON line per candidate, accepted or not."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TrialRecord:
    """One candidate's outcome.

    ``status`` is ``"accepted"``, ``"rejected"`` (evaluated, refused by
    admission) or ``"failed"`` (never evaluated: refused by the parser or
    the Gate after every retry, or its graph did not compile); a failed
    trial does not count toward the evaluated-trial count. ``stage`` is
    where it ended: ``"parse"``, ``"gate"``, ``"evaluate"`` or
    ``"admission"``. ``candidate`` is the last version the proposer wrote,
    ``attempts`` every refused version before it (``item``, ``stage``,
    ``reasons``). ``stats`` holds the Gate's statistics and ``metrics`` the
    evaluation, once there are any.
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

    def records(self) -> list[TrialRecord]:
        """Every record written so far, in order."""
        if not self.path.exists():
            return []
        return [TrialRecord(**json.loads(line)) for line in self.path.read_text().splitlines() if line]
