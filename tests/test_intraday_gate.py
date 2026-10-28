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
