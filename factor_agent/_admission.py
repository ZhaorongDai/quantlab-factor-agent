"""The admission policy: a pure function of a candidate's metrics and the trial count.

The checks run in order and the first that fails is the reason:

1. on an average training day the candidate has a value on at least
   ``min_coverage`` of the members (its other statistics speak only for
   the members it covers);
2. the training IC has the candidate's ``expected_sign`` (a candidate is
   never flipped);
3. its correlation with every reference and accepted factor is at most
   ``max_correlation`` in absolute value;
4. its partial IC, signed by ``expected_sign``, has a Newey-West t at or
   above the threshold for the number of evaluated trials so far;
5. its decile returns, signed by ``expected_sign``, rise with the decile:
   their Spearman correlation with the decile number is at least
   ``min_monotonicity``.

The threshold is a Bonferroni correction anchored at ``t_threshold``
(Harvey, Liu and Zhu 2016): the one-sided p-value of ``t_threshold`` is
divided by the trial count n, so the first trial needs ``t_threshold`` and
the bar rises with every trial (3.0 becomes about 3.64 at 10 trials, 4.20 at
100 and 4.58 at 600).
"""

import math

import numpy as np
from scipy import stats

from factor_agent.candidate import Candidate
from factor_agent.config import AdmissionConfig
from factor_agent.evaluator import Metrics


def t_threshold(config: AdmissionConfig, trials: int) -> float:
    """The partial-IC Newey-West t a candidate needs as the ``trials``-th evaluated trial."""
    return float(stats.norm.isf(stats.norm.sf(config.t_threshold) / max(trials, 1)))


def monotonicity(decile_returns, expected_sign: int) -> float:
    """The Spearman correlation of the decile number with the decile returns, signed by ``expected_sign``."""
    returns = np.asarray(decile_returns, dtype=float)
    if not np.isfinite(returns).all() or np.ptp(returns) == 0:
        return float("nan")
    return expected_sign * float(stats.spearmanr(np.arange(len(returns)), returns).statistic)


def reasons(candidate: Candidate, metrics: Metrics, trials: int, config: AdmissionConfig) -> list[str]:
    """Why ``candidate`` is refused, as one reason, or nothing when it is accepted."""
    sign = candidate.expected_sign
    if not metrics.coverage >= config.min_coverage:
        return [
            f"coverage {metrics.coverage:.2f} (the mean daily share of the members with a value) "
            f"is below {config.min_coverage:g}"
        ]
    if not math.isfinite(metrics.ic_mean) or np.sign(metrics.ic_mean) != sign:
        return [
            f"the training IC {metrics.ic_mean:+.4f} does not have the expected sign {sign:+d}; "
            f"a candidate is never flipped"
        ]
    if abs(metrics.nearest_correlation) > config.max_correlation:
        return [
            f"cross-sectional correlation {metrics.nearest_correlation:+.2f} with {metrics.nearest} "
            f"is above {config.max_correlation:g} in absolute value"
        ]
    threshold = t_threshold(config, trials)
    partial_t = sign * metrics.partial_ic_t
    if not partial_t >= threshold:
        return [
            f"the partial IC's Newey-West t {partial_t:.2f} (residualised against the reference set and "
            f"the accepted factors, signed by the expected sign) is below the threshold {threshold:.2f} "
            f"at {trials} evaluated trial(s)"
        ]
    rising = monotonicity(metrics.decile_returns, sign)
    if not rising >= config.min_monotonicity:
        return [
            f"decile monotonicity {rising:.2f} (Spearman correlation of the decile returns with the decile, "
            f"signed by the expected sign) is below {config.min_monotonicity:g}"
        ]
    return []
