"""The diagnosis of a run a real campaign wrote, and the collisions it is read from.

Three runs on the synthetic market, the proposer scripted to provoke the
diagnosis's rules (every Gate refusal is final: no retries):

- *troubled*, KLEN (Alpha158's candle length) its only known factor, two
  rounds of twelve slots: eight rewrites of KLEN, six replies that are not
  candidates, five pieces of code whose graph does not build, four
  candidates that cover too few stocks and one noise candidate.
- *bar*, one round of six slots: four noise candidates below the t
  threshold and two that cover too few stocks; nothing is accepted.
- *repeats*, two rounds of six slots: round 1 accepts quiet volume and
  refuses two repeats of it as duplicates of the run's own candidate and
  three noise candidates below the t threshold; round 2 rewrites quiet
  volume five times and proposes a dollar-volume copy of it, refused for its
  correlation.
"""

import pytest
from quantlab.core.component import rebuild
from quantlab.factor.config import FactorConfig
from quantlab.factor.predefined.alpha158 import Alpha158Stock

from factor_agent.web.run_view import Collision, RunView
from tests import synthetic_market
from tests.campaign_runner import candidate, run, write_config
from tests.fake_llm import FakeLLM

KLEN = "KLEN (Alpha158Stock, a known factor)"
QUIET_VOLUME = "quiet volume: a day traded on less volume than its twenty-day average"
QUIET = f"{QUIET_VOLUME} (accepted in round 1)"


def _klen(components) -> dict:
    return Alpha158Stock(FactorConfig(
        warmup_bars=60, dataset=rebuild(components["prices"]), mode="batch",
        data_columns=("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"), factor_names=("KLEN",),
    )).get_config()


def _sparse(n: int) -> dict:
    # Log of a number that is negative on most days: a value on a few members only.
    return candidate(f"sparse {n}", f"alpha = Log(Log(close / open) * 100 - {n})")


def _noise(n: int) -> dict:
    lag, window = [(5, 10), (10, 20), (20, 40), (40, 60), (60, 5)][n]
    return candidate(f"noise {n}", f"alpha = BackRef(high, {lag}) / BackRef(WindowedAvg(high, {window}), 5)")


def _klen_rewrite(n: int) -> dict:
    return candidate(f"candle length {n}", "alpha = Rank((high - low) / open)")


def _build_failure(n: int) -> dict:
    return candidate(f"divide by zero {n}", "alpha = Div(close, 0)")


#: Not a candidate: a reply item the parser cannot read as one.
UNREADABLE = "a candidate the model forgot to write"


def _troubled(number: int, directions: list) -> list:
    builds = 3 if number == 1 else 2
    items = ([_klen_rewrite(i) for i in range(4)] + [UNREADABLE] * 3 + [_build_failure(i) for i in range(builds)]
             + [_sparse(2 * number - 1), _sparse(2 * number)])
    return items + [_noise(0)] * (12 - len(items))  # one noise candidate in round 2


def _bar(number: int, directions: list) -> list:
    return [_noise(i) for i in range(4)] + [_sparse(1), _sparse(2)]


def _repeats(number: int, directions: list) -> list:
    if number == 1:
        return [candidate(QUIET_VOLUME, "alpha = WindowedAvg(volume, 20) / volume"),
                candidate("quiet volume again", "alpha = Rank(WindowedAvg(volume, 20) / volume)"),
                candidate("quiet volume doubled", "alpha = WindowedAvg(volume, 20) / volume * 2"),
                *[_noise(i) for i in range(3)]]
    return [candidate(f"quiet volume rewritten {i}", "alpha = Rank(WindowedAvg(volume, 20) / volume)") for i in range(5)] + [
        candidate("dollar volume surge", "alpha = close * volume / WindowedAvg(close * volume, 20)")]


@pytest.fixture(scope="module")
def market(tmp_path_factory):
    root = tmp_path_factory.mktemp("diagnosis")
    return {"root": root, "components": synthetic_market.build(root / "data"), "cache": root / "kunquant_cache"}


def _run(market, name: str, propose, rounds: int, slots: int, **sections) -> RunView:
    config = write_config(market["root"] / name, market["components"], **sections,
                          round={"candidates": slots, "min_per_type": 0, "retries": 0, "max_rounds": rounds})
    run(config, FakeLLM(propose), market["cache"])
    return RunView.read(market["root"] / name / "run")


@pytest.fixture(scope="module")
def troubled(market):
    return _run(market, "troubled", _troubled, 2, 12, known=[_klen(market["components"])])


@pytest.fixture(scope="module")
def bar(market):
    return _run(market, "bar", _bar, 1, 6)


@pytest.fixture(scope="module")
def repeats(market):
    return _run(market, "repeats", _repeats, 2, 6)


def test_the_diagnosis_holds_at_most_three_findings_the_one_touching_the_most_trials_first(troubled):
    assert troubled.summary.trials == 24
    assert [(f.rule, f.trials) for f in troubled.diagnosis] == [
        ("known_collisions", 8), ("unreadable_replies", 6), ("graph_build_failures", 5)]


def test_a_factor_of_the_users_collided_with_more_than_five_times_is_named_with_the_count(troubled):
    finding = troubled.diagnosis[0]
    assert KLEN in finding.seen and "8" in finding.seen
    assert "research directions" in finding.do and "known factors" in finding.do



def test_unreadable_replies_and_graph_build_failures_are_found_with_what_to_do(troubled):
    _, unreadable, builds = troubled.diagnosis
    assert "6 of 24 trials" in unreadable.seen and "model" in unreadable.do and "output limit" in unreadable.do
    assert "5 of 24 trials" in builds.seen and "operator file" in builds.do


def test_a_code_collision_names_its_object_and_kind(troubled):
    rewrite = troubled.trials[0]
    assert rewrite.reasons[0].startswith("the factor rewrites KLEN")
    assert rewrite.collisions == (Collision(label=KLEN, name=KLEN, kind="code", accepted_round=None, known=True),)
    others = [t for t in troubled.trials if not t.candidate.get("description", "").startswith("candle length")]
    assert len(others) == 16 and all(t.collisions == () for t in others)


def test_nothing_accepted_mostly_below_the_t_threshold_is_the_statistical_bar_and_coverage_refusals_are_found(bar):
    assert [(f.rule, f.trials) for f in bar.diagnosis] == [("below_threshold", 4), ("coverage_refusals", 2)]
    below, coverage = bar.diagnosis
    assert "4 of 6 evaluated trials" in below.seen and "not an error" in below.do
    assert "2 of 6 trials" in coverage.seen and "fields" in coverage.do and "dates" in coverage.do


def test_collisions_with_factors_the_run_accepted_are_one_finding_reported_as_the_novelty_checks_at_work(repeats):
    assert [(f.rule, f.trials) for f in repeats.diagnosis] == [("accepted_collisions", 6)]
    assert "novelty checks" in repeats.diagnosis[0].do
    by_description = {t.candidate["description"]: t for t in repeats.trials}
    quiet = Collision(label=QUIET, name="round 1: quiet volume: a day traded on…", kind="code", accepted_round=1,
                      known=False)
    assert by_description["quiet volume rewritten 0"].collisions == (quiet,)
    assert by_description["dollar volume surge"].collisions == (Collision(**{**quiet.__dict__, "kind": "ranking"}),)


def test_a_repeat_of_the_runs_own_earlier_candidate_is_not_a_collision(repeats):
    duplicates = [t for t in repeats.trials if t.reasons and t.reasons[0].startswith("the graph is a duplicate")]
    assert len(duplicates) == 2
    assert all(t.collisions == () for t in duplicates)
