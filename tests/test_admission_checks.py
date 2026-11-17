"""The admission checks a campaign configures: each with its own threshold, each one it may switch off.

Admission runs its checks in order on an evaluated candidate, through ``_admission.reasons``, and
the first that fails is the reason. The signal checks (the partial IC's t, the size of the IC and of
the partial IC, yearly stability, monotonicity and the confirmation) judge whether the factor predicts
returns; coverage and correlation do not. A campaign keeps at least one signal check.
"""

import json

import pandas as pd
import pytest

from factor_agent._admission import NOT_CONFIRMED, Shortfall, reason_kind, reasons, shortfalls, t_threshold
from factor_agent.config import AdmissionConfig
from factor_agent.evaluator import CompileFailure, DailyIC, Metrics, yearly_stability

#: A candidate every default check accepts: a negative factor (lower value, higher return).
GOOD = dict(
    ic_mean=-0.006, ic_t=-8.0, partial_ic_mean=-0.004, partial_ic_t=-8.0, nearest="x", nearest_correlation=0.1,
    nearest_known="y", nearest_known_correlation=-0.2, decile_returns=tuple(range(10, 0, -1)), coverage=0.9,
    n_days=100, start="s", end="e", years_in_direction=10, full_years=10,
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


def test_a_full_year_is_one_with_enough_scored_days_and_it_went_the_factors_way_when_its_mean_ic_did():
    # A negative factor: its IC went its way (below zero) in 2001 and 2003, not in 2002; 2004 has too few days to count.
    history = daily({2001: (250, -0.01), 2002: (250, 0.01), 2003: (250, -0.01), 2004: (119, 0.05)})
    assert yearly_stability(history, -1) == (2, 3)
    assert yearly_stability(daily({2001: (100, -0.01), 2002: (119, -0.01)}), -1) == (0, 0)


def test_a_factor_whose_ic_went_its_way_in_too_few_full_years_is_refused_counting_them():
    config = AdmissionConfig(min_year_share=0.75)
    [reason] = reasons(metrics(years_in_direction=2, full_years=3), 1, config)
    assert reason == (
        "the training IC went in the factor's direction in 2 of 3 full years (0.67, years with at least "
        "120 scored days), below min_year_share 0.75"
    )
    assert reason_kind(reason) == "yearly stability"
    assert reasons(metrics(years_in_direction=2, full_years=3), 1, AdmissionConfig(min_year_share=0.6)) == []


def test_without_a_full_training_year_the_yearly_stability_fails_whatever_its_threshold():
    [reason] = reasons(metrics(years_in_direction=0, full_years=0), 1, AdmissionConfig(min_year_share=0.01))
    assert reason.startswith("the training IC went in the factor's direction in 0 of 0 full years")


def test_a_factors_statistics_read_back_from_the_record_the_ledger_keeps():
    kept = metrics(pool_share=0.2, overlaps=(("x", 0.3),), daily=daily({2001: (250, -0.01)}))
    recorded = kept.to_dict()
    assert (recorded["years_in_direction"], recorded["full_years"], "daily" in recorded) == (10, 10, False)
    assert Metrics.from_dict(json.loads(json.dumps(recorded))) == kept


def test_a_record_written_before_the_yearly_counts_does_not_read_back():
    recorded = metrics().to_dict()
    del recorded["years_in_direction"], recorded["full_years"]
    with pytest.raises(TypeError):
        Metrics.from_dict(recorded)


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


def test_a_shortfall_is_a_failing_check_with_how_far_below_its_threshold_the_factor_is():
    # The factor is negative: its partial IC 0.004 in its direction is a fifth below 0.005.
    config = AdmissionConfig(t_threshold=None, min_partial_ic=0.005)
    assert shortfalls(metrics(), 1, config) == (Shortfall("min_partial_ic", pytest.approx(0.2)),)
    assert shortfalls(metrics(partial_ic_mean=-0.006), 1, config) == ()


def test_every_failing_check_is_a_shortfall_in_admission_order_where_reasons_give_only_the_first():
    config = AdmissionConfig(min_ic=0.008, min_year_share=0.75)
    poor = metrics(coverage=0.4, nearest_correlation=0.77, partial_ic_t=-2.0, years_in_direction=0, full_years=0)
    assert [(s.check, s.gap) for s in shortfalls(poor, 1, config)] == [
        ("min_coverage", pytest.approx(0.2)),
        ("max_correlation", pytest.approx(0.1)),
        ("t_threshold", pytest.approx((t_threshold(config, 1) - 2.0) / t_threshold(config, 1))),
        ("min_ic", pytest.approx(0.25)),
        ("min_year_share", pytest.approx(1.0)),
    ]
    [reason] = reasons(poor, 1, config)
    assert reason_kind(reason) == "coverage"


def test_a_check_switched_off_is_never_a_shortfall_and_the_confirmation_never_is():
    off = AdmissionConfig(min_coverage=None, max_correlation=None, t_threshold=None, min_monotonicity=None)
    poor = metrics(coverage=0.1, nearest_correlation=0.99, partial_ic_t=-0.1, decile_returns=tuple(range(10)))
    assert shortfalls(poor, 1, AdmissionConfig(min_ic=0.001, **{k: getattr(off, k) for k in (
        "min_coverage", "max_correlation", "t_threshold", "min_monotonicity")})) == ()


def test_the_correlation_shortfall_is_the_larger_of_the_pools_and_the_known_factors():
    assert shortfalls(metrics(nearest_correlation=0.1, nearest_known_correlation=-0.84), 1, AdmissionConfig()) == (
        Shortfall("max_correlation", pytest.approx(0.2)),
    )
