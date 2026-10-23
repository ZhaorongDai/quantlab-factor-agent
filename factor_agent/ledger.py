"""The trial ledger: one JSON line per candidate, accepted or not."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrialRecord:
    """One candidate's outcome.

    ``stage`` is where it ended: ``"parse"``, ``"gate"``, ``"evaluate"``
    (its graph did not compile) or ``"admission"``; an accepted candidate
    passed admission. ``metrics`` is set once it was evaluated.
    """

    round: int
    candidate: dict
    status: str
    stage: str
    reasons: list[str]
    fingerprint: str | None = None
    metrics: dict | None = None


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
