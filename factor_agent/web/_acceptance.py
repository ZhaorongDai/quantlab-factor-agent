"""What Acceptance has said of a run: its reports ``acceptance/<segment>[_<digest>].json`` and the hold-out lock's opens.

A report that records opens of the hold-out is a hold-out report. A report
written before the contribution table or the selection flag existed reads
with ``None`` for them: nothing it lacks is made up.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from factor_agent.web._files import read_json, utc


@dataclass(frozen=True)
class Contribution:
    """One accepted factor's contribution to the model in one mode, from an Acceptance report's contribution table.

    ``factor`` is the factor's label. ``mode`` is ``"drop_one"`` (the whole
    selection against the selection without it) or ``"add_one"`` (the
    baseline with only it against the baseline). ``difference`` holds, per
    compared statistic, with the factor minus without it, seed-averaged;
    ``agreeing`` the seeds whose own difference has the sign of that mean.
    """

    factor: str
    mode: str
    difference: dict[str, float | None]
    agreeing: dict[str, int]


@dataclass(frozen=True)
class Comparison:
    """One Acceptance report: the user's model with and without accepted factors, on validation or the hold-out.

    ``name`` is the report's file name without ``.json`` (``validation``, or
    ``validation_<digest>`` for a selection). ``holdout`` is true for a
    report on the hold-out. ``factors`` holds the labels of the factors
    added to the model. ``selection`` is true for a selection of fewer than
    every accepted factor, ``None`` for a report written before the flag.
    ``baseline`` and ``with_accepted`` hold each compared statistic averaged
    over ``seeds``, ``difference`` the second minus the first. ``verdict``
    is ``"improves"`` when the rank IC and the top-k excess return both
    rise with the factors, ``"does_not_improve"`` when neither does,
    ``"mixed"`` otherwise, ``None`` when either is undefined. ``forced`` is
    true for a hold-out opened again past its lock. ``contribution`` is the
    report's contribution table, the largest rank IC difference first:
    empty when no contribution was asked for, ``None`` for a report written
    before the table. ``written`` is the report file's time (UTC).
    """

    name: str
    segment: str
    holdout: bool
    factors: tuple[str, ...]
    selection: bool | None
    seeds: tuple[int, ...]
    top_k: int | None
    train_window: tuple[str, ...]
    test_window: tuple[str, ...]
    baseline: dict[str, float | None]
    with_accepted: dict[str, float | None]
    difference: dict[str, float | None]
    verdict: str | None
    forced: bool
    contribution: tuple[Contribution, ...] | None
    code_version: str
    written: datetime


@dataclass(frozen=True)
class HoldoutOpen:
    """One open of the hold-out, as the hold-out lock records it: ``forced`` for every open after the first."""

    opened_at: str
    code_version: str
    forced: bool


@dataclass(frozen=True)
class AcceptanceReports:
    """What Acceptance has said of a run's accepted factors.

    ``validation`` is the validation report on every accepted factor, or
    when there is none the latest validation report of a selection;
    ``holdout`` is the report of the latest open of the hold-out. Either is
    ``None`` while Acceptance has not written it. ``reports`` holds every
    report, the earliest written first; ``holdout_opens`` every open the
    hold-out lock records, which may include an open that wrote no report.
    """

    validation: Comparison | None
    holdout: Comparison | None
    reports: tuple[Comparison, ...]
    holdout_opens: tuple[HoldoutOpen, ...]


def _verdict(difference: dict) -> str | None:
    """Whether the rank IC and the top-k excess return both rose, neither did, or one did."""
    rank_ic, excess = difference.get("rank_ic"), difference.get("top_k_excess")
    if rank_ic is None or excess is None:
        return None
    if rank_ic > 0 and excess > 0:
        return "improves"
    return "does_not_improve" if rank_ic <= 0 and excess <= 0 else "mixed"


def _comparison(path: Path) -> Comparison | None:
    """The report at ``path``, ``None`` when it is not an Acceptance report."""
    report = read_json(path)
    if not isinstance(report, dict) or "difference" not in report:
        return None
    mean, difference = report.get("mean") or {}, report["difference"]
    table = report.get("contribution")
    return Comparison(
        name=path.stem, segment=report.get("segment", path.stem), holdout=bool(report.get("holdout_opens")),
        factors=tuple(report.get("accepted") or ()), selection=report.get("selection"),
        seeds=tuple(report.get("seeds") or ()), top_k=report.get("top_k"),
        train_window=tuple(report.get("train_window") or ()), test_window=tuple(report.get("test_window") or ()),
        baseline=mean.get("baseline") or {}, with_accepted=mean.get("with_accepted") or {}, difference=difference,
        verdict=_verdict(difference), forced=bool(report.get("forced")),
        contribution=None if table is None else tuple(
            Contribution(factor=row["factor"], mode=row["mode"], difference=row["difference"], agreeing=row["agreeing"])
            for row in table
        ),
        code_version=(report.get("code_version") or {}).get("factor_agent") or "unknown",
        written=utc(path.stat().st_mtime),
    )


def acceptance(run_dir: Path) -> AcceptanceReports:
    """The Acceptance reports, the earliest written first, and the hold-out lock's opens."""
    found = [(path.stat().st_mtime_ns, path) for path in (run_dir / "acceptance").glob("*.json")]
    reports = tuple(c for _, path in sorted(found) if (c := _comparison(path)))
    validations = [c for c in reports if not c.holdout]
    validation = next((c for c in validations if c.name == c.segment), validations[-1] if validations else None)
    holdouts = [c for c in reports if c.holdout]
    lock = read_json(run_dir / "holdout.lock") or {}
    opens = tuple(
        HoldoutOpen(opened_at=o["opened_at"], code_version=(o.get("code_version") or {}).get("factor_agent") or "unknown",
                    forced=bool(o.get("forced")))
        for o in lock.get("opens") or ()
    )
    return AcceptanceReports(validation=validation, holdout=holdouts[-1] if holdouts else None, reports=reports,
                             holdout_opens=opens)
