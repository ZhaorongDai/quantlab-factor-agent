"""The run directory's files as every part of the reader opens them, and its events replayed once."""

import json
from datetime import datetime, timezone
from pathlib import Path

from factor_agent import events


def loads(text: str):
    """JSON text read with NaN as ``None``: the ledger and the Acceptance reports write NaN where a value is undefined."""
    return json.loads(text, parse_constant=lambda _: None)


def read_json(path: Path):
    """The JSON file at ``path``, ``None`` when there is none."""
    try:
        return loads(path.read_text())
    except FileNotFoundError:
        return None


def utc(seconds: float) -> datetime:
    """The UTC time of a file time."""
    return datetime.fromtimestamp(seconds, timezone.utc)


def call_files(run_dir: Path) -> list[Path]:
    """The run's call files, in no order."""
    return list((run_dir / "llm_log").glob("*.json"))


def written(run_dir: Path) -> list[Path]:
    """The run's files that change while it runs."""
    files = [run_dir / name for name in ("snapshot.json", "state.json", "ledger.jsonl", events.FILE)]
    return [p for p in files if p.exists()] + call_files(run_dir)


class Replayed:
    """A run's events, and those since its latest opening (``current``), the opening first."""

    def __init__(self, replayed: list[events.Event]):
        self.all = replayed
        openings = [i for i, event in enumerate(replayed) if isinstance(event, events.Start)]
        self.current = replayed[openings[-1]:] if openings else []
        self.opened: events.Start | None = self.current[0] if self.current else None

    @classmethod
    def read(cls, run_dir: Path) -> "Replayed":
        """The events of the run in ``run_dir``; none when it has no events file."""
        return cls(events.read(run_dir / events.FILE))
