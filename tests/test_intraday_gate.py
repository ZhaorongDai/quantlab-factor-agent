"""The Gate on intraday candidates: a two-stage candidate passes, one breaking a stage rule is refused with why.

The vocabulary is the price-volume and minute-bar field files and the
example operator file; candidates go through a real Gate, sandbox included.
"""

import pytest

from factor_agent.candidate import Candidate
from factor_agent.config import GateConfig
from factor_agent.gate import Gate, Passed, Rejected
from factor_agent.zoo import Zoo
from tests.vocabulary_files import intraday


def _check(code: str):
    candidate = Candidate(
        hypothesis="Late trading carries information.", mechanism_type="behavioural_bias",
        horizon_rationale="A week.", description="Test.", code=code,
    )
    return Gate(GateConfig(), intraday()).check(candidate, Zoo())


@pytest.mark.parametrize("code", [
    # An intraday stage ending in SessionMean, a daily window over it.
    "alpha = WindowedAvg(SessionMean(bar_buy_volume - bar_sell_volume) / SessionMean(bar_volume), 5)",
    # Aggregated values combined with daily fields and daily operators.
    "alpha = Rank(SessionMean(bar_dollar_volume)) - Rank(WindowedAvg(close * volume, 20))",
    # Several intraday branches.
    "late = SessionMean(Select(session_bars_left < 20, bar_volume, ConstantOp(0)))\n"
    "alpha = late / SessionMean(bar_volume) + SessionMean(bar_close / last_price)",
])
def test_a_two_stage_candidate_passes(code):
    result = _check(code)
    assert isinstance(result, Passed), getattr(result, "reasons", None)


@pytest.mark.parametrize("code, reason", [
    ("alpha = bar_close / close", "the intraday field bar_close reaches the daily stage without a session aggregation"),
    ("alpha = SessionMean(bar_close / close)", "the daily field close is read inside the intraday stage, under SessionMean"),
    (
        "alpha = SessionMean(ExpMovingAvg(bar_close, 5))",
        "ExpMovingAvg is used in the intraday stage, but may be used in the daily stage only",
    ),
])
def test_a_candidate_breaking_a_stage_rule_is_refused_with_the_reason(code, reason):
    result = _check(code)
    assert isinstance(result, Rejected)
    assert reason in result.reasons


@pytest.mark.parametrize("code, hint", [
    ("alpha = bar_close / close", "with a session aggregation first, such as SessionMean(bar_close)"),
    ("alpha = SessionMean(bar_close / close)", "SessionMean(x) / close, not SessionMean(x / close)"),
    ("alpha = SessionMean(close)", "aggregate a value of the intraday fields, such as SessionMean(bar_close)"),
    ("alpha = SessionStd(SessionMean(bar_close))", "such as WindowedAvg(SessionMean(x), 20)"),
    ("alpha = SessionMean(ExpMovingAvg(bar_close, 5))", "such as ExpMovingAvg(SessionMean(x), ...)"),
])
def test_a_refused_stage_rule_says_how_to_write_it_instead(code, hint):
    result = _check(code)
    assert isinstance(result, Rejected)
    assert any(hint in h for h in result.hints), result.hints


def test_a_linear_regression_is_refused_in_the_intraday_stage():
    result = _check("alpha = SessionMean(WindowedLinearRegressionSlope(bar_close, 10))")
    assert isinstance(result, Rejected)
    assert result.reasons == (
        "WindowedLinearRegressionSlope is used in the intraday stage, but may be used in the daily stage only",
    )


def test_an_exponential_average_is_allowed_in_the_daily_stage():
    result = _check("alpha = ExpMovingAvg(SessionMean(bar_close) / close, 5)")
    assert isinstance(result, Passed), getattr(result, "reasons", None)


@pytest.mark.parametrize("code", [
    # 15 bars and a one-bar lag are intraday values; 15 is no daily window.
    "alpha = SessionMean(WindowedAvg(bar_close, 15) / BackRef(bar_close, 1))",
    # Lags chained in the intraday stage: 10 + 5 bars, an intraday lag.
    "alpha = SessionMean(bar_close / BackRef(BackRef(bar_close, 10), 5))",
    # A loop of the intraday stage over 15 bars.
    "loop = ForeachBackWindow(bar_close, 15)\n"
    "with Builder(loop):\n"
    "    item = IterValue(loop, bar_close)\n"
    "alpha = SessionMean(ReduceMax(item) / bar_close)",
])
def test_a_window_of_the_intraday_set_passes_in_the_intraday_stage(code):
    result = _check(code)
    assert isinstance(result, Passed), getattr(result, "reasons", None)


@pytest.mark.parametrize("code, reason", [
    (
        "alpha = SessionMean(WindowedAvg(bar_close, 20))",
        "WindowedAvg's window 20 is not one of the intraday windows (5, 10, 15, 30, 60, 120 bars)",
    ),
    (
        "alpha = SessionMean(bar_close / BackRef(bar_close, 250))",
        "BackRef's lag 250 is not one of the intraday lags (1, 5, 10, 15, 30, 60 bars)",
    ),
    (
        "alpha = WindowedAvg(SessionMean(bar_close), 15)",
        "WindowedAvg's window 15 is not one of the daily windows (5, 10, 20, 40, 60, 120, 250)",
    ),
    (
        "alpha = SessionMean(bar_close / BackRef(BackRef(bar_close, 60), 60))",
        "the lags chain to BackRef lag 120 in the intraday stage, which is not one of the intraday lags "
        "(1, 5, 10, 15, 30, 60 bars)",
    ),
])
def test_a_window_outside_its_stages_set_is_refused_naming_the_set(code, reason):
    result = _check(code)
    assert isinstance(result, Rejected)
    assert result.reasons == (reason,)
    assert any("in the intraday stage a window is one of 5, 10, 15, 30, 60, 120 bars" in h for h in result.hints), result.hints


#: An intraday candidate, and a rewrite of it: other variable names, commutative operands in another order.
EARLIER = (
    "flow = (bar_buy_volume - bar_sell_volume) * bar_close\n"
    "spread = SessionCorr(bar_volume, WindowedAvg(bar_close, 5))\n"
    "alpha = WindowedAvg(SessionSum(flow) / SessionSum(bar_volume * bar_close), 5) + spread * Rank(close)"
)
REWRITTEN = (
    "imbalance = bar_close * (bar_buy_volume - bar_sell_volume)\n"
    "alpha = Rank(close) * SessionCorr(WindowedAvg(bar_close, 5), bar_volume)"
    " + WindowedAvg(SessionSum(imbalance) / SessionSum(bar_close * bar_volume), 5)"
)


def _candidate(code: str) -> Candidate:
    return Candidate(
        hypothesis="Order flow carries information.", mechanism_type="behavioural_bias",
        horizon_rationale="A week.", description="Test.", code=code,
    )


def test_a_rewritten_intraday_candidate_is_a_duplicate_of_the_earlier_one():
    gate = Gate(GateConfig(), intraday())
    earlier = gate.check(_candidate(EARLIER), Zoo())
    assert isinstance(earlier, Passed), getattr(earlier, "reasons", None)
    rewritten = gate.check(_candidate(REWRITTEN), Zoo())
    assert isinstance(rewritten, Rejected) and any("duplicate" in r for r in rewritten.reasons), rewritten
    assert Gate(GateConfig(), intraday()).check(_candidate(REWRITTEN), Zoo()).fingerprint == earlier.fingerprint


def test_a_rewritten_intraday_candidate_rewrites_the_zoo_factor_it_copies():
    earlier = Gate(GateConfig(), intraday()).check(_candidate(EARLIER), Zoo())
    zoo = Zoo()
    zoo.add("order_flow", "accepted", earlier.graph)
    rewritten = Gate(GateConfig(), intraday()).check(_candidate(REWRITTEN), zoo)
    assert isinstance(rewritten, Rejected)
    assert any(r.startswith("the factor rewrites") for r in rewritten.reasons), rewritten.reasons
