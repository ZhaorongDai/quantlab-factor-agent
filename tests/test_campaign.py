"""Campaign.run(): one round end to end, with a scripted LLM as the only fake.

The proposer's first reply holds every kind of candidate; the next two
answer the Gate's fix requests: the syntax error is fixed on the first
retry, every other refused candidate comes back unchanged and fails.

The Gate builds every graph in a real subprocess and the Evaluator compiles
the round with KunQuant and reads the synthetic store of
``tests/synthetic_market.py`` through quantlab.
"""

import json
from pathlib import Path

import pytest
import yaml

from factor_agent.campaign import Campaign
from factor_agent.config import CampaignConfig
from quantlab.core.component import rebuild
from quantlab.factor.config import FactorConfig, PolarsFactorConfig
from quantlab.factor.kunquant_compile import CACHE_ENV
from quantlab.factor.predefined.alpha158 import Alpha158Stock
from tests import synthetic_market


def _candidate(description: str, code: str, **overrides) -> dict:
    return {
        "hypothesis": f"Hypothesis behind {description}.",
        "mechanism_type": "behavioural_bias",
        "expected_sign": 1,
        "horizon_rationale": "Resolves within a week.",
        "reference": "Synthetic test reference",
        "description": description,
        "code": code,
        **overrides,
    }


CANDIDATES = [
    _candidate("planted", "alpha = close / open"),
    _candidate("noise", "alpha = BackRef(volume, 10) / WindowedAvg(volume, 60)"),
    _candidate("syntax error", "alpha = close /"),
    _candidate("unknown name", "alpha = SecretOp(close)"),
    _candidate("import", "import os\nalpha = close"),
    _candidate("two outputs", "alpha = (close / open, volume)"),
    _candidate("endless loop", "while True:\n    pass\nalpha = close"),
    _candidate("bad schema", "alpha = close", mechanism_type="astrology"),
    _candidate("duplicate", "x = close\ny = open\nalpha = x / y"),
]

FIXED = _candidate("syntax error", "alpha = Rank(BackRef(Min(open, close) / low, 10))")

#: Candidates refused before evaluation in the first reply, in order; the
#: proposer is asked to fix exactly these.
REFUSED = ["syntax error", "unknown name", "import", "two outputs", "endless loop", "bad schema", "duplicate"]
STILL_FAILING = [c for c in CANDIDATES if c["description"] in REFUSED[1:]]


def _reply(items: list[dict]) -> str:
    return "Here are the candidates.\n```json\n" + json.dumps(items, indent=2) + "\n```\n"


@pytest.fixture(scope="module")
def outcome(tmp_path_factory):
    root = tmp_path_factory.mktemp("campaign")
    components = synthetic_market.build(root / "data")
    replies = root / "replies.json"
    replies.write_text(json.dumps([_reply(CANDIDATES), _reply([FIXED, *STILL_FAILING]), _reply(STILL_FAILING)]))
    config_path = root / "campaign.yaml"
    config_path.write_text(yaml.safe_dump(json.loads(json.dumps({
        "run_dir": str(root / "run"),
        "segments": {"training": {"start": synthetic_market.TRAINING[0], "end": synthetic_market.TRAINING[1]}},
        "data": components,
        "evaluator": {"warmup_bars": 60, "njobs": 4},
        "gate": {"timeout_seconds": 3},
        "admission": {"t_threshold": 3.0},
        "round": {"candidates": len(CANDIDATES)},
        "llm": {"proposer": {"backend": "scripted", "model": "scripted", "replies": str(replies)}},
    }))))
    cache = root / "kunquant_cache"
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(CACHE_ENV, str(cache))
        result = Campaign(CampaignConfig.from_file(config_path)).run()
    ledger = [json.loads(line) for line in (root / "run" / "ledger.jsonl").read_text().splitlines()]
    return {"result": result, "ledger": ledger, "cache": cache}


def _by_description(records) -> dict:
    return {record.candidate["description"]: record for record in records}


def test_the_planted_signal_is_accepted_and_the_rest_rejected_or_failed(outcome):
    result = outcome["result"]
    assert [r.candidate["description"] for r in result.accepted] == ["planted"]
    assert {r.candidate["description"] for r in result.rejected} == {"noise", "syntax error"}
    assert {r.candidate["description"] for r in result.failed} == set(REFUSED[1:])
    planted = result.accepted[0]
    assert planted.metrics["ic_mean"] > 0.1
    assert planted.metrics["ic_t"] >= 3.0


def test_the_noise_candidate_is_rejected_on_its_metrics(outcome):
    noise = _by_description(outcome["result"].rejected)["noise"]
    assert noise.stage == "admission"
    assert abs(noise.metrics["ic_t"]) < 3.0
    assert any("t" in reason for reason in noise.reasons)


@pytest.mark.parametrize(
    ("description", "reason"),
    [
        ("unknown name", "SecretOp"),
        ("import", "import"),
        ("two outputs", "exactly one output"),
        ("endless loop", "timed out"),
        ("duplicate", "duplicate"),
    ],
)
def test_a_candidate_the_gate_refuses_three_times_is_a_failed_trial(outcome, description, reason):
    record = _by_description(outcome["result"].failed)[description]
    assert record.stage == "gate"
    assert record.metrics is None
    assert any(reason in r for r in record.reasons), record.reasons
    assert len(record.attempts) == 3
    assert all(any(reason in r for r in attempt["reasons"]) for attempt in record.attempts)


def test_a_reply_item_that_breaks_the_schema_fails_when_parsed(outcome):
    record = _by_description(outcome["result"].failed)["bad schema"]
    assert record.stage == "parse"
    assert len(record.attempts) == 3
    assert any("astrology" in r for r in record.reasons)


def test_a_candidate_fixed_on_its_first_retry_is_evaluated(outcome):
    record = _by_description(outcome["result"].rejected)["syntax error"]
    assert record.candidate["code"] == FIXED["code"]
    assert record.metrics is not None
    assert [a["stage"] for a in record.attempts] == ["gate"]
    assert any("syntax error" in r for r in record.attempts[0]["reasons"])
    assert record.attempts[0]["item"]["code"] == "alpha = close /"


def test_failed_trials_do_not_count_as_evaluated(outcome):
    result = outcome["result"]
    assert result.evaluated_trials == 3
    assert all(r.metrics is None for r in result.failed)


def test_only_the_training_segment_is_evaluated(outcome):
    result = outcome["result"]
    evaluated = [r for r in result.accepted + result.rejected if r.metrics is not None]
    assert {r.candidate["description"] for r in evaluated} == {"planted", "noise", "syntax error"}
    for record in evaluated:
        assert record.metrics["start"] >= synthetic_market.TRAINING[0]
        assert record.metrics["end"] <= synthetic_market.TRAINING[1]


def test_every_candidate_is_in_the_ledger_with_its_metrics(outcome):
    ledger = {entry["candidate"]["description"]: entry for entry in outcome["ledger"]}
    assert set(ledger) == {c["description"] for c in CANDIDATES}
    assert ledger["planted"]["status"] == "accepted"
    assert ledger["planted"]["metrics"]["ic_t"] >= 3.0
    assert ledger["planted"]["fingerprint"]
    assert ledger["planted"]["stats"]["dimension"] == "pure number"
    assert ledger["noise"]["status"] == "rejected"
    assert ledger["import"]["status"] == "failed"
    assert ledger["noise"]["metrics"]["n_days"] > 400
    assert all(entry["round"] == 1 for entry in outcome["ledger"])


def test_the_round_is_compiled_once_with_its_neutralisation(outcome):
    # The round graph and its neutralisation; the neutralisation graph may come
    # from a library compiled earlier in the same process, leaving one entry.
    entries = [p for p in Path(outcome["cache"]).iterdir() if not p.name.startswith(".")]
    assert 1 <= len(entries) <= 2


def _run(root: Path, replies: list[list[dict]], reference=(), **sections) -> dict:
    """Run a campaign on a fresh synthetic store; ``replies`` are the proposer's, one list of candidates per reply."""
    components = synthetic_market.build(root / "data")
    prices = rebuild(components["prices"])
    path = root / "replies.json"
    path.write_text(json.dumps([_reply(items) for items in replies]))
    config_path = root / "campaign.yaml"
    config_path.write_text(yaml.safe_dump(json.loads(json.dumps({
        "run_dir": str(root / "run"),
        "segments": {"training": {"start": synthetic_market.TRAINING[0], "end": synthetic_market.TRAINING[1]}},
        "data": components,
        "reference": [factor(prices).get_config() for factor in reference],
        "evaluator": {"warmup_bars": 60, "njobs": 4},
        "llm": {"proposer": {"backend": "scripted", "model": "scripted", "replies": str(path)}},
        **sections,
    }))))
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(CACHE_ENV, str(root / "kunquant_cache"))
        result = Campaign(CampaignConfig.from_file(config_path)).run()
    return {"result": result, "run_dir": root / "run"}


def _alpha158(prices, names=None) -> Alpha158Stock:
    return Alpha158Stock(FactorConfig(
        warmup_bars=60, dataset=prices, mode="batch",
        data_columns=("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"), factor_names=names,
    ))


#: Alpha158's candlestick features: the reference of the multi-round campaign, which carries no volume.
KBAR = ("KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2")


# Two rounds against a zoo of Alpha158's candlestick features and a Polars
# factor: round 1 accepts the volume signal and refuses a rewrite of KMID;
# round 2 proposes the volume signal again under a rank, which only the zoo
# can refuse.
ROUND_ONE = [
    _candidate("volume", "alpha = volume / WindowedAvg(volume, 20)"),
    _candidate("kmid", "alpha = (close - open) / open"),
]
ROUND_TWO = [_candidate("volume again", "alpha = Rank(volume / WindowedAvg(volume, 20))")]


@pytest.fixture(scope="module")
def two_rounds(tmp_path_factory):
    return _run(
        tmp_path_factory.mktemp("two_rounds"), [ROUND_ONE, ROUND_TWO],
        reference=[lambda prices: _alpha158(prices, KBAR),
                   lambda prices: synthetic_market.AdjustedMomentum(PolarsFactorConfig(warmup_bars=20, dataset=prices))],
        round={"candidates": 2, "retries": 0, "max_rounds": 2},
    )


def test_a_rewrite_of_a_reference_factor_fails_naming_it(two_rounds):
    record = _by_description(two_rounds["result"].failed)["kmid"]
    assert record.stage == "gate"
    assert any("KMID (Alpha158Stock)" in r and "Div(Sub(close, open), open)" in r for r in record.reasons)


def test_a_factor_accepted_in_an_earlier_round_is_in_the_zoo(two_rounds):
    result = two_rounds["result"]
    assert [(r.round, r.candidate["description"]) for r in result.accepted] == [(1, "volume")]
    record = _by_description(result.failed)["volume again"]
    assert record.round == 2
    assert any("volume (accepted in round 1)" in r for r in record.reasons), record.reasons


def test_reference_factors_without_a_graph_are_reported_as_not_covered(two_rounds):
    assert two_rounds["result"].not_covered == ("momentum_20 (AdjustedMomentum)",)
    zoo = json.loads((two_rounds["run_dir"] / "zoo.json").read_text())
    assert zoo["not_covered"] == ["momentum_20 (AdjustedMomentum)"]
    assert "KMID (Alpha158Stock)" in zoo["covered"]


# One round against KMID, the planted intraday signal up to a constant. Each
# candidate meets one admission check; the volume signal passes them all.
ADMISSION = [
    _candidate("wrong sign", "alpha = high / Max(open, close)", expected_sign=-1),
    _candidate("noisy copy", "alpha = (close - open) / (high - low)"),
    _candidate("averaged copy", "alpha = WindowedAvg(close / open, 5)"),
    _candidate("incremental", "alpha = volume / WindowedAvg(volume, 20)"),
    _candidate("incremental again", "alpha = volume / WindowedAvg(volume, 10)"),
    _candidate("one decile", "alpha = Min(open, close) / low"),
]


@pytest.fixture(scope="module")
def admission(tmp_path_factory):
    outcome = _run(
        tmp_path_factory.mktemp("admission"), [ADMISSION],
        reference=[lambda prices: _alpha158(prices, ("KMID",))],
        round={"candidates": len(ADMISSION), "retries": 0},
    )
    result = outcome["result"]
    return {**outcome, "records": _by_description(result.accepted + result.rejected)}


def test_a_signal_with_the_wrong_stated_sign_is_rejected_not_flipped(admission):
    record = admission["records"]["wrong sign"]
    assert record.status == "rejected"
    assert record.stage == "admission"
    assert record.metrics["ic_mean"] > 0
    assert record.metrics["ic_t"] > 3
    assert record.candidate["expected_sign"] == -1
    assert record.reasons == [
        f"the training IC {record.metrics['ic_mean']:+.4f} does not have the expected sign -1; "
        "a candidate is never flipped"
    ]


def test_a_noisy_copy_of_a_reference_factor_is_rejected_for_correlation(admission):
    record = admission["records"]["noisy copy"]
    assert record.status == "rejected"
    assert record.metrics["nearest"] == "KMID (Alpha158Stock)"
    assert len(record.reasons) == 1
    assert "correlation" in record.reasons[0] and "KMID (Alpha158Stock)" in record.reasons[0]


def test_a_candidate_whose_ic_the_reference_explains_is_rejected_on_partial_ic(admission):
    record = admission["records"]["averaged copy"]
    assert record.status == "rejected"
    assert record.metrics["ic_t"] > 5
    assert len(record.reasons) == 1 and "partial IC" in record.reasons[0]


def test_a_candidate_with_incremental_information_is_accepted_on_partial_ic(admission):
    record = admission["records"]["incremental"]
    assert record.status == "accepted"
    assert record.reasons == []
    assert abs(record.metrics["nearest_correlation"]) < 0.1
    assert record.metrics["partial_ic_t"] > 10


def test_a_candidate_accepted_earlier_in_the_round_is_compared_with_the_next(admission):
    record = admission["records"]["incremental again"]
    assert record.status == "rejected"
    assert record.metrics["nearest"] == "incremental (accepted in round 1)"


def test_a_candidate_whose_ic_comes_from_one_decile_fails_monotonicity(admission):
    record = admission["records"]["one decile"]
    assert record.status == "rejected"
    assert record.metrics["partial_ic_t"] > 5
    assert len(record.reasons) == 1 and "monotonicity" in record.reasons[0]


def test_admission_metrics_are_in_the_ledger(admission):
    ledger = [json.loads(line) for line in (admission["run_dir"] / "ledger.jsonl").read_text().splitlines()]
    metrics = {entry["candidate"]["description"]: entry["metrics"] for entry in ledger}
    assert set(metrics) == {c["description"] for c in ADMISSION}
    assert len(metrics["incremental"]["decile_returns"]) == 10
    assert metrics["noisy copy"]["nearest_correlation"] > 0.7


# The weak upper-wick signal (Newey-West t a little above 3): proposed alone
# it is the only evaluated trial; proposed with nine noise candidates it is
# judged at ten, when the threshold has risen above its t, wherever it stands
# in the proposal order.
WEAK = _candidate("weak", "alpha = high / Max(open, close)")
NOISE = [
    _candidate(f"noise {i}", code)
    for i, code in enumerate(
        [f"alpha = BackRef(high / Max(open, close), {lag})" for lag in (5, 10, 20, 40)]
        + [f"alpha = BackRef(volume, {lag}) / WindowedAvg(volume, 60)" for lag in (5, 10, 20, 40)]
        + ["alpha = BackRef(volume, 5) / WindowedAvg(volume, 120)"]
    )
]


@pytest.fixture(scope="module")
def weak_alone(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("weak_alone"), [[WEAK]], round={"candidates": 1, "retries": 0})


@pytest.fixture(scope="module")
def weak_among_noise(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("weak_among_noise"), [[WEAK, *NOISE]], round={"candidates": 10, "retries": 0})


def test_the_same_metrics_pass_at_a_low_trial_count_and_fail_at_a_high_one(weak_alone, weak_among_noise):
    alone = _by_description(weak_alone["result"].accepted)["weak"]
    among = _by_description(weak_among_noise["result"].rejected)["weak"]
    assert json.dumps(alone.metrics) == json.dumps(among.metrics)  # NaN-safe: no pool, no nearest factor
    assert 3.0 < alone.metrics["partial_ic_t"] < 3.6
    assert len(among.reasons) == 1
    assert "at 10 evaluated trial(s)" in among.reasons[0]
    assert weak_among_noise["result"].evaluated_trials == 10
