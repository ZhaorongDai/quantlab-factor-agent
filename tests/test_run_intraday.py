"""A run of an intraday family as the frontend reads it: each trial and accepted factor says whether it is intraday.

The campaign of ``synthetic_intraday_market`` proposes the late volume
share, a candidate aggregating each session's minute bars, which is
accepted, beside a daily candidate and code the Gate refuses. A trial is
intraday when its graph holds a session aggregation.
"""

import pytest

from factor_agent.web.run_view import RunView
from tests import synthetic_intraday_market as market
from tests.campaign_runner import candidate, run, write_config
from tests.fake_llm import FakeLLM
from tests.vocabulary_files import MINUTE_BARS, PRICE_VOLUME

LATE_SHARE = candidate(
    "late volume share",
    f"late = SessionMean(Select(session_bars_left < {market.LATE_BARS}, bar_volume, ConstantOp(0)))\n"
    "alpha = WindowedAvg(late / SessionMean(bar_volume), 5)",
)
DAILY_PRICE = candidate("daily price", "alpha = WindowedAvg(close / open, 5)")
UNKNOWN = candidate("unknown name", "alpha = SecretOp(SessionMean(bar_volume))")
CANDIDATES = [LATE_SHARE, DAILY_PRICE, UNKNOWN]


@pytest.fixture(scope="module")
def view(tmp_path_factory):
    root = tmp_path_factory.mktemp("run_intraday")
    components = market.build(root / "data")
    config = write_config(
        root, {**components, "exposures": None},
        segments={"training": {"start": market.TRAINING[0], "end": market.TRAINING[1]}},
        data={name: components[name] for name in ("prices", "label", "membership", "intraday")},
        scoring={"neutralize": None}, fields=[str(PRICE_VOLUME), str(MINUTE_BARS)],
        round={"candidates": len(CANDIDATES), "min_per_type": 0, "max_rounds": 1, "retries": 0},
    )
    run(config, FakeLLM(lambda number, directions: CANDIDATES[:len(directions)]), root / "kunquant_cache")
    return RunView.read(root / "run")


def test_a_trial_is_intraday_when_its_graph_holds_a_session_aggregation(view):
    trials = {t.candidate["description"]: t for t in view.trials}
    assert trials["late volume share"].metrics is not None
    assert trials["daily price"].metrics is not None
    assert {name: t.intraday for name, t in trials.items()} == {
        "late volume share": True, "daily price": False, "unknown name": False,
    }


def test_an_accepted_intraday_factor_says_it_is_intraday(view):
    assert [(f.description, f.intraday) for f in view.factors] == [("late volume share", True)]
