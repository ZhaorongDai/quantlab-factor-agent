"""Evaluation segments: the date ranges the campaign and the acceptance read, the purge and the hold-out lock.

The training segment is the only one the campaign evaluates; the validation
segment is read by ``Acceptance.validate`` and the hold-out by
``Acceptance.open_holdout``, once.

The purge: a forward label at t reads prices up to ``lookahead`` bars past t,
so wherever labels are read the last bars of a segment are dropped and no
label reads a bar of the next segment. At least ``purge_bars`` (five)
trading days are dropped, more when the label reads further (``ret_5``,
whose label at t reads the open of t + 6, drops six).

The hold-out lock: ``Segments.open_holdout`` writes a lock file recording
when and with which code version the hold-out was opened. A second open is
refused unless forced; every open, forced or not, is appended to the lock.
"""

import datetime
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

#: Trading days dropped at the end of a segment at least (Q35).
PURGE_BARS = 5


@dataclass(frozen=True)
class Segment:
    """A named, inclusive date range."""

    name: str
    start: str
    end: str

    def __post_init__(self):
        if pd.Timestamp(self.start) > pd.Timestamp(self.end):
            raise ValueError(f"segment {self.name}: start {self.start} is after end {self.end}")


def purge(lookahead: int, purge_bars: int = PURGE_BARS) -> int:
    """The bars dropped at the end of a segment for a label that reads ``lookahead`` bars past t."""
    return max(purge_bars, lookahead)


class HoldoutLocked(RuntimeError):
    """The hold-out was opened before and the open was not forced."""


@dataclass(frozen=True)
class HoldoutOpen:
    """One open of the hold-out: when (UTC), with which code, and whether it was forced past the lock."""

    opened_at: str
    code_version: dict
    forced: bool


@dataclass(frozen=True)
class HoldoutLock:
    """The lock after an open: ``opens`` holds every open so far, this one last."""

    path: Path
    opens: tuple[HoldoutOpen, ...]

    @property
    def last(self) -> HoldoutOpen:
        """This open."""
        return self.opens[-1]


@dataclass(frozen=True)
class Segments:
    """The training, validation and hold-out segments, in that order, and the purge between them.

    Parameters
    ----------
    training : Segment
        The segment the campaign evaluates.
    validation, holdout : Segment, optional
        The segments ``Acceptance`` reads; each starts after the one before ends.
    purge_bars : int
        The fewest trading days dropped at the end of a segment.
    """

    training: Segment
    validation: Segment | None = None
    holdout: Segment | None = None
    purge_bars: int = PURGE_BARS

    def __post_init__(self):
        order = [s for s in (self.training, self.validation, self.holdout) if s is not None]
        for before, after in zip(order, order[1:]):
            if pd.Timestamp(after.start) <= pd.Timestamp(before.end):
                raise ValueError(f"segment {after.name} starts on {after.start}, before {before.name} ends ({before.end})")
        if self.purge_bars < 0:
            raise ValueError(f"segments: purge_bars must be non-negative, got {self.purge_bars}")

    def scored(self, segment: Segment, calendar: pd.DatetimeIndex, lookahead: int) -> tuple[str, str]:
        """The first and last dates of ``segment`` whose label lies inside it.

        Parameters
        ----------
        segment : Segment
            The segment.
        calendar : pd.DatetimeIndex
            The trading days, covering the segment.
        lookahead : int
            Bars past t the label at t reads (``lookahead_bars()``).

        Raises
        ------
        ValueError
            If the purge leaves the segment no date.
        """
        bars = calendar[(calendar >= pd.Timestamp(segment.start)) & (calendar <= pd.Timestamp(segment.end))]
        dropped = purge(lookahead, self.purge_bars)
        kept = bars[: max(len(bars) - dropped, 0)]
        if not len(kept):
            raise ValueError(f"segment {segment.name}: the purge of {dropped} bars leaves no date")
        return kept[0].strftime("%Y-%m-%d"), kept[-1].strftime("%Y-%m-%d")

    def open_holdout(self, lock_path: str | Path, force: bool = False) -> HoldoutLock:
        """Record an open of the hold-out in the lock at ``lock_path`` and return the lock.

        Raises
        ------
        ValueError
            If there is no hold-out segment.
        HoldoutLocked
            If the hold-out was opened before and ``force`` is false; the lock is unchanged.
        """
        if self.holdout is None:
            raise ValueError("segments: there is no hold-out segment to open")
        path = Path(lock_path)
        opens = [HoldoutOpen(**o) for o in json.loads(path.read_text())["opens"]] if path.exists() else []
        if opens and not force:
            first = opens[0]
            raise HoldoutLocked(
                f"the hold-out was opened on {first.opened_at} (code {first.code_version.get('factor_agent')}), "
                f"{len(opens)} open(s) so far; pass force=True to open it again, which the report states"
            )
        opens.append(HoldoutOpen(
            opened_at=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            code_version=code_version(), forced=bool(opens),
        ))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"segment": asdict(self.holdout), "opens": [asdict(o) for o in opens]}, indent=2) + "\n")
        tmp.replace(path)
        return HoldoutLock(path, tuple(opens))


def code_version() -> dict:
    """The commit (``+dirty`` with uncommitted changes) of this package and of quantlab, and quantlab's version."""
    import quantlab

    import factor_agent

    return {
        "factor_agent": _commit(Path(factor_agent.__file__).parent),
        "quantlab": _commit(Path(quantlab.__file__).parent),
        "quantlab_version": quantlab.__version__,
    }


def _commit(directory: Path) -> str:
    """The git commit of the checkout holding ``directory``, or ``"unknown"`` outside one."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(directory), *args], capture_output=True, text=True, check=True).stdout

    try:
        head = git("rev-parse", "HEAD").strip()
        dirty = bool(git("status", "--porcelain", "--untracked-files=no").strip())
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("+dirty" if dirty else "")
