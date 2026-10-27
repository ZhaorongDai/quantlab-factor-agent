"""A run's trials as its trials page shows them: the evaluated ones placed for the admission map, the others grouped.

The run has two rounds of four slots and no retries. Round 1 accepts quiet
volume (a factor that predicts with direction -1), refuses noise below the
t threshold, refuses code the Gate does not know and an item the parser
cannot read. Round 2 accepts the planted intraday signal (direction +1)
and refuses the same unknown code again, a syntax error, and a candidate the
reviewer still finds inconsistent after its rewrite.
"""

import json

import pytest

from factor_agent import _admission
from factor_agent.config import AdmissionConfig
from factor_agent.web.run_view import RunView
from tests import synthetic_market
from tests.campaign_runner import candidate, run, write_config
from tests.fake_llm import FakeLLM

QUIET_VOLUME = "quiet volume: a day traded on less volume than its twenty-day average"

ROUND_1 = [
    candidate(QUIET_VOLUME, "alpha = WindowedAvg(volume, 20) / volume"),
    candidate("noise", "alpha = BackRef(volume, 10) / WindowedAvg(volume, 60)"),
    candidate("unknown name", "alpha = SecretOp(close)"),
    "not a candidate",
]
ROUND_2 = [
    candidate("planted", "alpha = close / open"),
    candidate("unknown name again", "alpha = SecretOp(close)"),
    candidate("syntax", "alpha = close /"),
    candidate("inconsistent thing", "alpha = Rank(BackRef(Min(open, close) / low, 10))"),
]


@pytest.fixture(scope="module")
def view(tmp_path_factory):
    root = tmp_path_factory.mktemp("trials")
    components = synthetic_market.build(root / "data")
    config = write_config(root / "two_rounds", components,
                          round={"candidates": 4, "min_per_type": 0, "retries": 0, "max_rounds": 2})
    run(config, FakeLLM(lambda number, directions: ROUND_1 if number == 1 else ROUND_2), root / "kunquant_cache")
    lines = (root / "two_rounds" / "run" / "ledger.jsonl").read_text().splitlines()
    return RunView.read(root / "two_rounds" / "run"), [json.loads(line) for line in lines]


def _description(entry: dict) -> str:
    return entry["candidate"].get("description", entry["candidate"].get("raw"))


def test_every_evaluated_trial_is_placed_by_its_status_and_round(view):
    run_view, ledger = view
    placed = [(p.round, p.slot, p.status) for p in run_view.evaluated]
    assert placed == [(1, 0, "accepted"), (1, 1, "rejected"), (2, 0, "accepted")]


def test_a_placed_trial_has_its_partial_ic_t_in_its_direction_and_its_largest_correlation(view):
    run_view, ledger = view
    quiet, noise, planted = run_view.evaluated
    raw = ledger[0]["metrics"]
    assert raw["ic_mean"] < 0 and quiet.factor_direction == -1
    assert (quiet.ic_mean, quiet.ic_t, quiet.partial_ic_mean, quiet.partial_ic_t) == (
        -raw["ic_mean"], -raw["ic_t"], -raw["partial_ic_mean"], -raw["partial_ic_t"])
    assert quiet.partial_ic_t > 0 and planted.factor_direction == 1
    # The first trial evaluated had nothing in the pool to be near, and the run has no known factors.
    assert quiet.largest_correlation is None
    assert planted.largest_correlation == abs(ledger[4]["metrics"]["nearest_correlation"])


def test_a_placed_trial_carries_its_rounds_threshold_and_the_map_draws_the_latest_one(view):
    run_view, ledger = view
    assert "is below the threshold" in ledger[1]["reasons"][0]
    quiet, noise, planted = run_view.evaluated
    assert noise.partial_ic_t < quiet.threshold == noise.threshold
    assert f"threshold {noise.threshold:.2f}" in ledger[1]["reasons"][0]
    # Round 2 evaluates one candidate after round 1's two: admission judged it as the third trial.
    assert planted.threshold == run_view.threshold == pytest.approx(_admission.t_threshold(AdmissionConfig(), 3))


def test_trials_that_never_reached_the_evaluator_are_grouped_by_stage_and_reason_with_their_rounds(view):
    run_view, ledger = view
    groups = [(g.stage, g.rounds, [(t.round, t.slot) for t in g.trials]) for g in run_view.never_evaluated]
    assert groups == [
        ("parse", (1,), [(1, 3)]),
        ("gate", (1, 2), [(1, 2), (2, 1)]),
        ("gate", (2,), [(2, 2)]),
        ("review", (2,), [(2, 3)]),
    ]
    unknown = run_view.never_evaluated[1]
    assert unknown.reason == ledger[2]["reasons"][0] and "SecretOp" in unknown.reason


def test_a_run_with_no_written_round_has_nothing_to_place(tmp_path):
    (tmp_path / "snapshot.json").write_text("{}")
    empty = RunView.read(tmp_path)
    assert (empty.evaluated, empty.never_evaluated, empty.threshold) == ((), (), None)
