"""A trial's daily IC, as the campaign keeps it in ``ic/round_NN.json`` for each evaluated trial."""

from dataclasses import dataclass
from pathlib import Path

from factor_agent.web._files import read_json


@dataclass(frozen=True)
class TrialIC:
    """An evaluated trial's daily rank IC and partial IC on each day of the label, ``None`` where unscored.

    The values are raw, as the Evaluator computed them; the page turns them to the factor's direction.
    """

    days: tuple[str, ...]
    ic: tuple[float | None, ...]
    partial_ic: tuple[float | None, ...]


def read_ic(run_dir: str | Path, round: int, slot: int) -> TrialIC | None:
    """The daily IC of trial ``slot`` (from 0) of ``round``; ``None`` when it was not evaluated or the run kept none."""
    kept = read_json(Path(run_dir) / "ic" / f"round_{round:02d}.json")
    series = (kept or {}).get("slots", {}).get(str(slot))
    if series is None:
        return None
    return TrialIC(tuple(kept["days"]), tuple(series["ic"]), tuple(series["partial_ic"]))
