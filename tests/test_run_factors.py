"""The accepted factors of a run a real campaign wrote, and the facts that say how far to trust them.

The run has two rounds of four slots and KLEN, Alpha158's candle length, as
its only known factor. Round 1 accepts quiet volume (a day traded on less
volume than its average, so a factor that predicts with direction -1) and
refuses two noise candidates below the t threshold, which their refusals
quote, and code the Gate refuses. Round 2 accepts the planted intraday
signal (direction +1), whose nearest accepted factor is quiet volume, and
refuses a dollar-volume copy of quiet volume for its correlation, quoting
the limit, and two pieces of code at the Gate: no refusal of round 2 quotes
a threshold.
"""

import json
import shutil

import pytest

from factor_agent import _admission
from factor_agent.config import AdmissionConfig
from factor_agent.web.run_view import RunView
from quantlab.core.component import rebuild
from quantlab.factor.config import FactorConfig
from quantlab.factor.predefined.alpha158 import Alpha158Stock
from tests import synthetic_market
from tests.campaign_runner import candidate, run, write_config
from tests.fake_llm import FakeLLM

QUIET_VOLUME = "quiet volume: a day traded on less volume than its twenty-day average"

ROUND_1 = [
    candidate(QUIET_VOLUME, "alpha = WindowedAvg(volume, 20) / volume"),
    candidate("noise", "alpha = BackRef(volume, 10) / WindowedAvg(volume, 60)"),
    candidate("unknown name", "alpha = SecretOp(close)"),
    candidate("more noise", "alpha = BackRef(close, 20) / BackRef(WindowedAvg(close, 10), 5)"),
]
ROUND_2 = [
    candidate("planted", "alpha = close / open"),
    candidate("dollar volume surge", "alpha = close * volume / WindowedAvg(close * volume, 20)"),
    candidate("unknown name again", "alpha = SecretOp(open)"),
    candidate("syntax", "alpha = close /"),
]

KLEN = "KLEN (Alpha158Stock, a known factor)"


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("factors")
    components = synthetic_market.build(root / "data")
    klen = Alpha158Stock(FactorConfig(
        warmup_bars=60, dataset=rebuild(components["prices"]), mode="batch",
        data_columns=("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"), factor_names=("KLEN",),
    ))
    config = write_config(root / "two_rounds", components, known=[klen.get_config()],
                          round={"candidates": 4, "min_per_type": 0, "retries": 0, "max_rounds": 2})
    run(config, FakeLLM(lambda number, directions: ROUND_1 if number == 1 else ROUND_2), root / "kunquant_cache")
    return root


def _factors(runs):
    return {f.description: f for f in RunView.read(runs / "two_rounds" / "run").factors}


def _ledger(runs) -> dict:
    lines = (runs / "two_rounds" / "run" / "ledger.jsonl").read_text().splitlines()
    return {entry["candidate"]["description"]: entry for entry in map(json.loads, lines)}


def test_each_accepted_factor_reads_with_its_round_mechanism_type_direction_and_write_up(runs):
    factors = _factors(runs)
    assert list(factors) == [QUIET_VOLUME, "planted"]
    quiet, planted = factors[QUIET_VOLUME], factors["planted"]
    assert (quiet.round, quiet.factor_direction, planted.round, planted.factor_direction) == (1, -1, 2, 1)
    assert planted.label == "planted (accepted in round 2)"
    assert planted.code == "alpha = close / open" and planted.hypothesis == "Hypothesis behind planted."
    ledger = _ledger(runs)
    assert list(quiet.decile_returns) == ledger[QUIET_VOLUME]["metrics"]["decile_returns"]
    assert quiet.coverage == ledger[QUIET_VOLUME]["metrics"]["coverage"]
    assert planted.mechanism_type == ledger["planted"]["direction"]["mechanism_type"]


def test_every_ic_and_t_is_given_in_the_factors_direction_with_the_raw_value_kept(runs):
    quiet = _factors(runs)[QUIET_VOLUME]
    raw = _ledger(runs)[QUIET_VOLUME]["metrics"]
    assert raw["ic_mean"] < 0 and raw["partial_ic_t"] < 0
    assert (quiet.ic_mean, quiet.ic_t, quiet.partial_ic_mean, quiet.partial_ic_t) == (
        -raw["ic_mean"], -raw["ic_t"], -raw["partial_ic_mean"], -raw["partial_ic_t"])
    assert quiet.metrics["partial_ic_t"] == raw["partial_ic_t"]


def test_a_rounds_threshold_is_the_one_admission_judged_it_by(runs):
    # Round 2 evaluates two candidates after round 1's three, and no refusal of it quotes its threshold.
    assert not any("threshold" in r for e in _ledger(runs).values() if e["round"] == 2 for r in e["reasons"])
    factors = _factors(runs)
    assert factors[QUIET_VOLUME].threshold == pytest.approx(3.51, abs=0.005)
    assert factors["planted"].threshold == pytest.approx(_admission.t_threshold(AdmissionConfig(), 5))


def _without_events(runs, tmp_path) -> RunView:
    """The run as a campaign without an events file leaves it."""
    shutil.copytree(runs / "two_rounds" / "run", tmp_path / "run")
    (tmp_path / "run" / "events.jsonl").unlink()
    return RunView.read(tmp_path / "run")


def test_without_events_a_rounds_threshold_is_the_one_its_refusals_quote_carried_over_to_a_round_that_quotes_none(
    runs, tmp_path,
):
    noise = _ledger(runs)["noise"]["reasons"][0]
    assert "is below the threshold 3.51 at 3 evaluated trial(s)" in noise
    factors = {f.description: f for f in _without_events(runs, tmp_path).factors}
    assert factors[QUIET_VOLUME].threshold == 3.51
    assert factors["planted"].threshold == 3.51


def test_decile_monotonicity_is_the_one_admission_judged_in_the_factors_direction(runs):
    for factor in _factors(runs).values():
        expected = _admission.monotonicity(factor.decile_returns, factor.factor_direction)
        assert factor.monotonicity == pytest.approx(expected) and factor.monotonicity > 0.6


def test_the_closest_factors_are_given_with_their_correlations_and_an_earlier_accepted_one_by_round_and_first_words(runs):
    factors, ledger = _factors(runs), _ledger(runs)
    planted, raw = factors["planted"], ledger["planted"]["metrics"]
    assert planted.nearest.label == f"{QUIET_VOLUME} (accepted in round 1)"
    assert (planted.nearest.name, planted.nearest.accepted_round) == ("round 1: quiet volume: a day traded on…", 1)
    assert planted.nearest.correlation == raw["nearest_correlation"]
    assert (planted.nearest_known.label, planted.nearest_known.name) == (KLEN, KLEN)
    assert planted.nearest_known.accepted_round is None
    assert planted.nearest_known.correlation == raw["nearest_known_correlation"]
    # The first factor accepted has nothing in the pool to be near.
    assert factors[QUIET_VOLUME].nearest is None and factors[QUIET_VOLUME].nearest_known.label == KLEN


def _first_rounds(runs, tmp_path, rounds: int) -> RunView:
    """The run as it read when only its first ``rounds`` rounds were written."""
    shutil.copytree(runs / "two_rounds" / "run", tmp_path / "run")
    state = json.loads((tmp_path / "run" / "state.json").read_text())
    (tmp_path / "run" / "state.json").write_text(json.dumps({**state, "rounds": rounds, "stop": None}))
    return RunView.read(tmp_path / "run")


def test_the_correlation_limit_is_the_one_the_campaign_was_opened_with_before_any_refusal_quotes_it(runs, tmp_path):
    first = _first_rounds(runs, tmp_path, 1)
    assert not any("absolute value" in r for t in first.trials for r in t.reasons)
    assert first.correlation_limit == 0.7


def test_without_events_the_correlation_limit_is_the_one_a_refusal_quoted_and_unknown_while_none_does(runs, tmp_path):
    assert "is above 0.7 in absolute value" in _ledger(runs)["dollar volume surge"]["reasons"][0]
    assert _without_events(runs, tmp_path / "all").correlation_limit == 0.7
    first = _first_rounds(runs, tmp_path / "first", 1)
    (tmp_path / "first" / "run" / "events.jsonl").unlink()
    first = RunView.read(tmp_path / "first" / "run")
    assert first.correlation_limit is None
    assert [f.threshold for f in first.factors] == [3.51]


def test_a_run_with_no_accepted_factor_has_none(runs, tmp_path):
    assert _first_rounds(runs, tmp_path, 0).factors == ()
