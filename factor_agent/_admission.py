"""The admission policy: a pure function of a candidate's metrics and the trial count.

A candidate states no direction: its direction is the sign of its training
IC (``direction``), and the statistics below are read in it. The checks run
in order and the first that fails is the reason:

1. on an average training day the candidate has a value on at least
   ``min_coverage`` of the members (its other statistics speak only for
   the members it covers);
2. its training IC is a nonzero number, so it has a direction;
3. its correlation with every reference and accepted factor is at most
   ``max_correlation`` in absolute value, and so is its sampled correlation
   with every known factor (on every ``KNOWN_SAMPLE_EVERY``-th scored date
   of the Evaluator);
4. its partial IC, in its direction, has a Newey-West t at or above the
   threshold for the number of evaluated trials so far;
5. its decile returns, in its direction, rise with the decile: their
   Spearman correlation with the decile number is at least
   ``min_monotonicity``.

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

import numpy as np
from scipy import stats

from factor_agent.config import AdmissionConfig
from factor_agent.evaluator import KNOWN_SAMPLE_EVERY, Metrics

#: The dates a known factor's correlation is sampled on, as the reason says.
_SAMPLED_DATES = f"one training date in {KNOWN_SAMPLE_EVERY}"


def t_threshold(config: AdmissionConfig, trials: int) -> float:
    """The partial-IC Newey-West t a candidate needs as the ``trials``-th evaluated trial."""
    return float(stats.norm.isf(stats.norm.sf(config.t_threshold) / (2 * max(trials, 1) ** config.trial_exponent)))


def direction(ic_mean: float) -> int:
    """A factor's direction: +1 when a higher value went with a higher training return, -1 when lower, 0 when neither."""
    return int(np.sign(ic_mean)) if math.isfinite(ic_mean) else 0


def monotonicity(decile_returns, sign: int) -> float:
    """The Spearman correlation of the decile number with the decile returns, signed by ``sign``."""
    returns = np.asarray(decile_returns, dtype=float)
    if not np.isfinite(returns).all() or np.ptp(returns) == 0:
        return float("nan")
    return sign * float(stats.spearmanr(np.arange(len(returns)), returns).statistic)


#: What each admission reason starts with, and the kind of refusal it is (``reason_kind``).
_KINDS = (
    ("coverage ", "coverage"),
    ("the training IC ", "no direction"),
    ("cross-sectional correlation ", "correlation"),
    ("the partial IC's ", "partial IC"),
    ("decile monotonicity ", "monotonicity"),
)


def reason_kind(reason: str) -> str:
    """The kind of an admission reason ``reasons`` wrote: coverage, no direction, correlation, partial IC or monotonicity."""
    return next((kind for start, kind in _KINDS if reason.startswith(start)), "other")


def reasons(metrics: Metrics, trials: int, config: AdmissionConfig) -> list[str]:
    """Why a candidate with ``metrics`` is refused, as one reason, or nothing when it is accepted."""
    sign = direction(metrics.ic_mean)
    if not metrics.coverage >= config.min_coverage:
        return [
            f"coverage {metrics.coverage:.2f} (the mean daily share of the members with a value) "
            f"is below {config.min_coverage:g}"
        ]
    if sign == 0:
        return [f"the training IC {metrics.ic_mean:+.4f} gives the factor no direction"]
    if abs(metrics.nearest_correlation) > config.max_correlation:
        return [
            f"cross-sectional correlation {metrics.nearest_correlation:+.2f} with {metrics.nearest} "
            f"is above {config.max_correlation:g} in absolute value"
        ]
    if abs(metrics.nearest_known_correlation) > config.max_correlation:
        return [
            f"cross-sectional correlation {metrics.nearest_known_correlation:+.2f} with {metrics.nearest_known}, "
            f"sampled on {_SAMPLED_DATES}, is above {config.max_correlation:g} in absolute value"
        ]
    threshold = t_threshold(config, trials)
    partial_t = sign * metrics.partial_ic_t
    if not partial_t >= threshold:
        return [
            f"the partial IC's Newey-West t {partial_t:.2f} (residualised against the reference set and "
            f"the accepted factors, in the factor's direction) is below the threshold {threshold:.2f} "
            f"at {trials} evaluated trial(s), each counted in both directions"
        ]
    rising = monotonicity(metrics.decile_returns, sign)
    if not rising >= config.min_monotonicity:
        return [
            f"decile monotonicity {rising:.2f} (Spearman correlation of the decile returns with the decile, "
            f"in the factor's direction) is below {config.min_monotonicity:g}"
        ]
    return []
