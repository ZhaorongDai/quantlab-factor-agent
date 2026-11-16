"""The admission checks a campaign configures: each with its own threshold, each one it may switch off.

Admission runs its checks in order on an evaluated candidate, through ``_admission.reasons``, and
the first that fails is the reason. The signal checks (the partial IC's t, the size of the IC and of
the partial IC, yearly stability, monotonicity and the confirmation) judge whether the factor predicts
returns; coverage and correlation do not. A campaign keeps at least one signal check.
"""

import pandas as pd
import pytest

from factor_agent._admission import NOT_CONFIRMED, reason_kind, reasons, t_threshold
from factor_agent.config import AdmissionConfig
from factor_agent.evaluator import CompileFailure, DailyIC, Metrics

#: A candidate every default check accepts: a negative factor (lower value, higher return).
GOOD = dict(
    ic_mean=-0.006, ic_t=-8.0, partial_ic_mean=-0.004, partial_ic_t=-8.0, nearest="x", nearest_correlation=0.1,
    nearest_known="y", nearest_known_correlation=-0.2, decile_returns=tuple(range(10, 0, -1)), coverage=0.9,
    n_days=100, start="s", end="e",
)


def metrics(**change) -> Metrics:
    return Metrics(**{**GOOD, **change})


def test_a_factor_whose_ic_is_smaller_than_min_ic_in_its_direction_is_refused_saying_both():
    config = AdmissionConfig(min_ic=0.008)
    [reason] = reasons(metrics(), 1, config)
    assert reason == "the training IC 0.0060 in the factor's direction is below min_ic 0.008"
    assert reason_kind(reason) == "IC size"
    assert reasons(metrics(ic_mean=-0.009), 1, config) == []


def test_a_factor_whose_partial_ic_is_smaller_than_min_partial_ic_in_its_direction_is_refused_saying_both():
    config = AdmissionConfig(min_partial_ic=0.005)
    [reason] = reasons(metrics(), 1, config)
    assert reason == (
        "the training partial IC 0.0040 (residualised against the reference set and the accepted factors) "
        "in the factor's direction is below min_partial_ic 0.005"
    )
    assert reason_kind(reason) == "partial IC size"
    assert reasons(metrics(partial_ic_mean=-0.006), 1, config) == []


def daily(ic_by_year: dict[int, tuple[int, float]]) -> DailyIC:
    """Each year's first ``days`` business days, every one with the year's IC (and partial IC)."""
    days, ics = [], []
    for year, (count, ic) in ic_by_year.items():
        stamps = pd.bdate_range(f"{year}-01-01", periods=count)
        days += [d.strftime("%Y-%m-%d") for d in stamps]
        ics += [ic] * count
    return DailyIC(tuple(days), tuple(ics), tuple(ics))


def test_a_factor_whose_ic_went_its_way_in_too_few_full_years_is_refused_counting_them():
    # A negative factor: its IC went its way (below zero) in 2001 and 2003, not in 2002; 2004 has too few days to count.
    history = daily({2001: (250, -0.01), 2002: (250, 0.01), 2003: (250, -0.01), 2004: (119, 0.05)})
    config = AdmissionConfig(min_year_share=0.75)
    [reason] = reasons(metrics(daily=history), 1, config)
    assert reason == (
        "the training IC went in the factor's direction in 2 of 3 full years (0.67, years with at least "
        "120 scored days), below min_year_share 0.75"
    )
    assert reason_kind(reason) == "yearly stability"
    assert reasons(metrics(daily=history), 1, AdmissionConfig(min_year_share=0.6)) == []


def test_the_confirmation_is_the_last_check_and_holds_only_in_the_factors_direction():
    config = AdmissionConfig(min_ic=0.008)
    held, reversed_ = metrics(partial_ic_t=-2.5), metrics(partial_ic_t=2.5)
    assert reasons(metrics(ic_mean=-0.009), 1, config, confirmation=held) == []
    assert reasons(metrics(ic_mean=-0.009), 1, config, confirmation=reversed_) == [NOT_CONFIRMED]
    assert reasons(metrics(ic_mean=-0.009), 1, config, confirmation=CompileFailure("no")) == [NOT_CONFIRMED]
    # A candidate failing a training check is refused for it, whatever its confirmation.
    [reason] = reasons(metrics(), 1, config, confirmation=reversed_)
    assert reason_kind(reason) == "IC size"


@pytest.mark.parametrize(("switched_off", "failing"), [
    ({"min_coverage": None}, {"coverage": 0.1}),
    ({"max_correlation": None}, {"nearest_correlation": 0.99, "nearest_known_correlation": -0.99}),
    ({"t_threshold": None}, {"partial_ic_t": -0.5}),
    ({"min_monotonicity": None}, {"decile_returns": tuple(range(10))}),
])
def test_a_check_switched_off_refuses_nothing(switched_off, failing):
    assert reasons(metrics(**failing), 1, AdmissionConfig(min_ic=0.001)) != []
    assert reasons(metrics(**failing), 1, AdmissionConfig(**switched_off, min_ic=0.001)) == []


def test_with_the_confirmation_switched_off_the_confirmation_segment_refuses_nothing():
    assert reasons(metrics(), 1, AdmissionConfig(confirm_t=None), confirmation=metrics(partial_ic_t=9.0)) == []


def test_with_the_t_check_switched_off_there_is_no_threshold():
    assert t_threshold(AdmissionConfig(t_threshold=None), 100) is None
