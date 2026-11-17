"""The admission policy: a pure function of a candidate's metrics, its confirmation and the trial count.

A candidate states no direction: its direction is the sign of its training
IC (``direction``), and the statistics below are read in it. The admission
checks run in order and the first that fails is the reason; each is off when
its threshold in ``AdmissionConfig`` is None, but the direction is always
needed:

1. on an average training day the candidate has a value on at least
   ``min_coverage`` of the members (its other statistics speak only for
   the members it covers); the campaign adds to the refusal where the
   graph loses its values (``coverage_cause``);
2. its training IC is a nonzero number, so it has a direction;
3. its correlation with every reference and accepted factor is at most
   ``max_correlation`` in absolute value, and so is its sampled correlation
   with every known factor (on every ``KNOWN_SAMPLE_EVERY``-th scored date
   of the Evaluator);
4. its partial IC, in its direction, has a Newey-West t at or above the
   threshold for the number of evaluated trials so far;
5. its decile returns, in its direction, rise with the decile: their
   Spearman correlation with the decile number is at least
   ``min_monotonicity``;
6. its training IC, in its direction, is at least ``min_ic``;
7. its training partial IC, in its direction, is at least ``min_partial_ic``;
8. its yearly stability: of the full training years (``FULL_YEAR_DAYS``
   scored days or more) at least the share ``min_year_share`` had a mean IC
   in its direction (``Metrics.years_in_direction`` of ``full_years``); without a
   full year it fails;
9. with a confirmation segment, its partial IC there, in its direction,
   has a Newey-West t of at least ``confirm_t`` (``confirmed``); the
   reason says so without a number, since no prompt shows a statistic of
   that segment.

The refusals of the other checks give the statistic and its threshold.
Each training check is defined once (``_COVERAGE``, ``_IN_DIRECTION``):
``reasons`` gives the first that fails, ``shortfalls`` every one that
fails and how far short.

The threshold is a Bonferroni correction anchored at ``t_threshold``
(Harvey, Liu and Zhu 2016). The direction is read from the data, so each
trial is two tests, one per direction: the one-sided p-value of
``t_threshold`` is divided by twice the trial count n (3.0 becomes about
3.21 at one trial, 3.82 at 10, 4.35 at 100 and 4.73 at 600). With
``trial_exponent`` a below 1 it is divided by twice n to the a instead, a
threshold rising more slowly than Bonferroni's (at 0.5, 3.82 at 100 and
4.03 at 600).
"""

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from factor_agent.config import AdmissionConfig
from factor_agent.evaluator import (
    FULL_YEAR_DAYS, KNOWN_SAMPLE_EVERY, CompileFailure, Metrics, NodeCoverage,
)

#: The dates a known factor's correlation is sampled on, as the reason says.
_SAMPLED_DATES = f"one training date in {KNOWN_SAMPLE_EVERY}"


def t_threshold(config: AdmissionConfig, trials: int) -> float | None:
    """The partial-IC Newey-West t a candidate needs as the ``trials``-th evaluated trial; None when the check is off."""
    if config.t_threshold is None:
        return None
    return float(stats.norm.isf(stats.norm.sf(config.t_threshold) / (2 * max(trials, 1) ** config.trial_exponent)))


def direction(ic_mean: float) -> int:
    """A factor's direction: +1 when a higher value went with a higher training return, -1 when lower, 0 when neither."""
    return int(np.sign(ic_mean)) if math.isfinite(ic_mean) else 0


def directional_t(metrics: dict) -> float:
    """An evaluated trial's residual (partial IC) Newey-West t in its direction, from its record's ``metrics``."""
    return direction(metrics["ic_mean"]) * metrics["partial_ic_t"]


def monotonicity(decile_returns, sign: int) -> float:
    """The Spearman correlation of the decile number with the decile returns, signed by ``sign``."""
    returns = np.asarray(decile_returns, dtype=float)
    if not np.isfinite(returns).all() or np.ptp(returns) == 0:
        return float("nan")
    return sign * float(stats.spearmanr(np.arange(len(returns)), returns).statistic)


#: What each admission reason matches, from its start, and the kind of refusal it is (``reason_kind``).
_KINDS = (
    (re.compile(r"coverage "), "coverage"),
    (re.compile(r"the training IC \S+ gives the factor no direction"), "no direction"),
    (re.compile(r"cross-sectional correlation "), "correlation"),
    (re.compile(r"the partial IC's "), "partial IC"),
    (re.compile(r"decile monotonicity "), "monotonicity"),
    (re.compile(r"the training IC \S+ in the factor's direction is below min_ic "), "IC size"),
    (re.compile(r"the training partial IC "), "partial IC size"),
    (re.compile(r"the training IC went in the factor's direction in "), "yearly stability"),
    (re.compile(r"the factor did not hold on the confirmation period"), "confirmation"),
)

#: Why a candidate that passed every other check is refused when its partial IC does not hold on the
#: confirmation segment: without a number, since no prompt may show a statistic of that segment.
NOT_CONFIRMED = (
    "the factor did not hold on the confirmation period, a later part of the training data whose "
    "statistics the search is not shown: its residual IC there was too weak or went the other way"
)


def confirmed(ic_mean: float, confirmation_partial_t: float, config: AdmissionConfig) -> bool:
    """Whether the partial IC on the confirmation segment holds in the direction of the training ``ic_mean``."""
    return direction(ic_mean) * confirmation_partial_t >= config.confirm_t


def reason_kind(reason: str) -> str:
    """The kind of an admission reason ``reasons`` wrote, one per admission check (``_KINDS``), or ``"other"``."""
    return next((kind for pattern, kind in _KINDS if pattern.match(reason)), "other")


def coverage_cause(trace: Sequence[NodeCoverage], render: Callable[[int], str], min_coverage: float) -> str:
    """Where a candidate refused for its coverage loses its values, from its ``trace`` (``Evaluator.trace_coverage``).

    The step named is the first traced node below ``min_coverage`` whose
    traced inputs are all at or above it, with the coverage of those inputs
    and the share on which the step or an input is infinite; a data column
    below it is named as the data it reads. ``render`` writes a node as
    candidate code writes it. The text continues an admission reason; it is
    empty when no such node is traced.
    """
    by_node = {t.node: t for t in trace}
    for t in trace:
        given = [by_node[j] for j in t.inputs]
        if t.coverage >= min_coverage or any(g.coverage < min_coverage for g in given):
            continue
        if t.column:
            return f"; the data column {render(t.node)} has a value on {t.coverage:.2f} of the member days"
        cause = f"; it falls at {render(t.node)}, to {t.coverage:.2f}"
        if given:
            cause += " from " + ", ".join(f"{g.coverage:.2f} for {render(g.node)}" for g in given)
        infinite = [(f"{render(g.node)} is", g) for g in given if g.infinite > 0]
        if t.infinite > 0:
            infinite = [("it is", t)]
        for subject, g in infinite:
            cause += (
                f"; {subject} infinite on {_share(g.infinite)} of the member days and on {_share(g.infinite_days)} "
                f"of the days at least once"
            )
        if infinite:
            cause += " (a division by zero is infinite)"
        return cause
    return ""


def _share(fraction: float) -> str:
    """A share as a percentage, with a decimal below 1%, so a small nonzero share does not read 0%."""
    return f"{fraction:.1%}" if fraction < 0.01 else f"{fraction:.0%}"


@dataclass(frozen=True)
class Shortfall:
    """An admission check a candidate fails (by its threshold's name) and how far it falls short, relative to the threshold."""

    check: str
    gap: float


@dataclass(frozen=True)
class _Check:
    """One training admission check: its threshold's name, the statistic it reads in the factor's direction, whether
    that statistic is a floor (at least the threshold) or a ceiling (at most), and its refusal reason."""

    name: str
    value: Callable[[Metrics, int], float]
    floor: bool
    reason: Callable[[Metrics, int, float, int], str]

    def passes(self, value: float, bar: float) -> bool:
        return value >= bar if self.floor else value <= bar

    def gap(self, value: float, bar: float) -> float:
        if not math.isfinite(value):
            return math.inf
        return (bar - value) / bar if self.floor else (value - bar) / bar


def _correlation(metrics: Metrics) -> float:
    """The larger absolute correlation with a pool factor or a known factor; 0 without either."""
    found = [abs(c) for c in (metrics.nearest_correlation, metrics.nearest_known_correlation) if math.isfinite(c)]
    return max(found, default=0.0)


def _correlation_reason(metrics: Metrics, sign: int, bar: float, trials: int) -> str:
    if math.isfinite(metrics.nearest_correlation) and abs(metrics.nearest_correlation) > bar:
        return (
            f"cross-sectional correlation {metrics.nearest_correlation:+.2f} with {metrics.nearest} "
            f"is above {bar:g} in absolute value"
        )
    return (
        f"cross-sectional correlation {metrics.nearest_known_correlation:+.2f} with {metrics.nearest_known}, "
        f"sampled on {_SAMPLED_DATES}, is above {bar:g} in absolute value"
    )


def _year_share(metrics: Metrics, sign: int) -> float:
    return metrics.years_in_direction / metrics.full_years if metrics.full_years else 0.0


#: The coverage check, the one admission makes before it reads the factor's direction.
_COVERAGE = _Check(
    "min_coverage", lambda m, sign: m.coverage, True,
    lambda m, sign, bar, trials: (
        f"coverage {m.coverage:.2f} (the mean daily share of the members with a value) is below {bar:g}"
    ),
)

#: The training checks admission makes once the factor has a direction, in order.
_IN_DIRECTION = (
    _Check("max_correlation", lambda m, sign: _correlation(m), False, _correlation_reason),
    _Check(
        "t_threshold", lambda m, sign: sign * m.partial_ic_t, True,
        lambda m, sign, bar, trials: (
            f"the partial IC's Newey-West t {sign * m.partial_ic_t:.2f} (residualised against the reference set and "
            f"the accepted factors, in the factor's direction) is below the threshold {bar:.2f} "
            f"at {trials} evaluated trial(s), each counted in both directions"
        ),
    ),
    _Check(
        "min_monotonicity", lambda m, sign: monotonicity(m.decile_returns, sign), True,
        lambda m, sign, bar, trials: (
            f"decile monotonicity {monotonicity(m.decile_returns, sign):.2f} (Spearman correlation of the decile "
            f"returns with the decile, in the factor's direction) is below {bar:g}"
        ),
    ),
    _Check(
        "min_ic", lambda m, sign: sign * m.ic_mean, True,
        lambda m, sign, bar, trials: (
            f"the training IC {sign * m.ic_mean:.4f} in the factor's direction is below min_ic {bar:g}"
        ),
    ),
    _Check(
        "min_partial_ic", lambda m, sign: sign * m.partial_ic_mean, True,
        lambda m, sign, bar, trials: (
            f"the training partial IC {sign * m.partial_ic_mean:.4f} (residualised against the reference set "
            f"and the accepted factors) in the factor's direction is below min_partial_ic {bar:g}"
        ),
    ),
    _Check(
        "min_year_share", _year_share, True,
        lambda m, sign, bar, trials: (
            f"the training IC went in the factor's direction in {m.years_in_direction} of {m.full_years} full years "
            f"({_year_share(m, sign):.2f}, years with at least {FULL_YEAR_DAYS} scored days), below min_year_share "
            f"{bar:g}"
        ),
    ),
)


def _bar(check: _Check, config: AdmissionConfig, trials: int) -> float | None:
    """A check's threshold: the config's, the t threshold at ``trials``; None when the check is off."""
    return t_threshold(config, trials) if check.name == "t_threshold" else getattr(config, check.name)


def _failing(checks, metrics: Metrics, sign: int, trials: int, config: AdmissionConfig):
    """Each of ``checks`` switched on that ``metrics`` fails, in order, with its threshold."""
    for check in checks:
        bar = _bar(check, config, trials)
        if bar is not None and not check.passes(check.value(metrics, sign), bar):
            yield check, bar


def shortfalls(metrics: Metrics, trials: int, config: AdmissionConfig) -> tuple[Shortfall, ...]:
    """Every training admission check switched on that a candidate with ``metrics`` fails, in admission order.

    Unlike ``reasons``, it goes on past the first; the direction and the confirmation are not among them.
    """
    sign = direction(metrics.ic_mean)
    return tuple(
        Shortfall(check.name, check.gap(check.value(metrics, sign), bar))
        for check, bar in _failing((_COVERAGE, *_IN_DIRECTION), metrics, sign, trials, config)
    )


def reasons(
    metrics: Metrics, trials: int, config: AdmissionConfig, confirmation: Metrics | CompileFailure | None = None,
) -> list[str]:
    """Why a candidate with ``metrics`` is refused, as one reason, or nothing when it is accepted.

    With its ``confirmation`` (its evaluation on the confirmation segment), a candidate passing every
    training check is refused unless its partial IC holds there (``confirmed``); a confirmation that
    did not compile does not hold.
    """
    sign = direction(metrics.ic_mean)
    for check, bar in _failing((_COVERAGE,), metrics, sign, trials, config):
        return [check.reason(metrics, sign, bar, trials)]
    if sign == 0:
        return [f"the training IC {metrics.ic_mean:+.4f} gives the factor no direction"]
    for check, bar in _failing(_IN_DIRECTION, metrics, sign, trials, config):
        return [check.reason(metrics, sign, bar, trials)]
    if confirmation is not None and config.confirm_t is not None and not (
        isinstance(confirmation, Metrics) and confirmed(metrics.ic_mean, confirmation.partial_ic_t, config)
    ):
        return [NOT_CONFIRMED]
    return []
