"""Campaign.run(): one round end to end, with a scripted LLM as the only fake.

The proposer fills the round's directions with every kind of candidate, in
order; the fix requests are answered by fixing the syntax error on the
first retry and sending every other refused candidate back unchanged, so it
fails. The planner and the reviewer of ``tests/fake_llm`` plan to the quota
and pass every candidate.

The Gate builds every graph in a real subprocess and the Evaluator compiles
the round with KunQuant and reads the synthetic store of
``tests/synthetic_market.py`` through quantlab.
"""

import json
import math
from pathlib import Path

import pytest

from factor_agent.gate.vocabulary import VocabularyError
from quantlab.core.component import rebuild
from quantlab.factor.config import FactorConfig, NeutralizedConfig, PolarsFactorConfig
from quantlab.factor.predefined.alpha158 import Alpha158Stock
from quantlab.factor.predefined.neutralized import NeutralizedFactor
from tests import synthetic_market
from tests.campaign_runner import candidate as _candidate
from tests.campaign_runner import run, write_config
from tests.fake_llm import FakeLLM, Interrupted, round_of
from tests.vocabulary_files import PRICE_VOLUME

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


def _fix(number, failures):
    return [FIXED if f["refused_version"]["description"] == "syntax error" else f["refused_version"] for f in failures]


@pytest.fixture(scope="module")
def outcome(tmp_path_factory):
    root = tmp_path_factory.mktemp("campaign")
    config_path = write_config(
        root, synthetic_market.build(root / "data"),
        gate={"timeout_seconds": 10}, admission={"t_threshold": 3.0},
        round={"candidates": len(CANDIDATES), "min_per_type": 1, "max_rounds": 1},
    )
    fake = FakeLLM(lambda number, directions: CANDIDATES, fix=_fix)
    result = run(config_path, fake, root / "kunquant_cache")
    ledger = [json.loads(line) for line in (root / "run" / "ledger.jsonl").read_text().splitlines()]
    return {"result": result, "ledger": ledger, "cache": root / "kunquant_cache"}


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


def _run(root: Path, rounds: list[list[dict]], reference=(), fix=None, known=(), crash=None, **sections) -> dict:
    """Run a campaign on a fresh synthetic store; ``rounds`` holds the proposer's candidates of each round.

    Each item of ``reference`` (and of ``known``) makes a reference (known) factor from the prices
    and the exposures; ``fix`` answers the fix requests (``FakeLLM``'s). With ``crash``, the run is
    first interrupted at the request ``crash(request)`` is true for, then resumed with a fresh fake.
    """
    components = synthetic_market.build(root / "data")
    prices, exposures = rebuild(components["prices"]), rebuild(components["exposures"])
    sections["round"] = {"min_per_type": 0, "max_rounds": len(rounds), **sections.get("round", {})}
    made = [factor(prices, exposures).get_config() for factor in reference]
    if known:
        sections["known"] = [factor(prices, exposures).get_config() for factor in known]
    config_path = write_config(root, components, made, **sections)
    if crash is not None:
        with pytest.raises(Interrupted):
            run(config_path, FakeLLM(lambda number, directions: rounds[number - 1], fix=fix, interrupt=crash), root / "kunquant_cache")
    fake = FakeLLM(lambda number, directions: rounds[number - 1], fix=fix)
    result = run(config_path, fake, root / "kunquant_cache")
    return {"result": result, "run_dir": root / "run", "fake": fake}


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
ROUND_TWO = [
    _candidate("volume again", "alpha = Rank(volume / WindowedAvg(volume, 20))"),
    _candidate("noise", "alpha = BackRef(volume, 10) / WindowedAvg(volume, 60)"),
]


@pytest.fixture(scope="module")
def two_rounds(tmp_path_factory):
    return _run(
        tmp_path_factory.mktemp("two_rounds"), [ROUND_ONE, ROUND_TWO],
        reference=[lambda prices, _: _alpha158(prices, KBAR),
                   lambda prices, _: synthetic_market.AdjustedMomentum(PolarsFactorConfig(warmup_bars=20, dataset=prices))],
        round={"candidates": 2, "retries": 0},
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


# One round with Alpha158's candlestick features as the reference set and
# every Alpha158 feature, the candlestick features again and a Polars factor
# as known factors: a rewrite of a known feature fails at the Gate naming it
# as known, and an output of the reference set is taken there only.
AGAINST_KNOWN = [
    _candidate("std20", "alpha = WindowedStddev(close, 20) / close"),
    _candidate("volume", "alpha = volume / WindowedAvg(volume, 20)"),
]


@pytest.fixture(scope="module")
def against_known(tmp_path_factory):
    return _run(
        tmp_path_factory.mktemp("against_known"), [AGAINST_KNOWN],
        reference=[lambda prices, _: _alpha158(prices, KBAR)],
        known=[lambda prices, _: _alpha158(prices),
               lambda prices, _: _alpha158(prices, KBAR),
               lambda prices, _: synthetic_market.AdjustedMomentum(PolarsFactorConfig(warmup_bars=20, dataset=prices))],
        round={"candidates": len(AGAINST_KNOWN), "retries": 0},
    )


def test_a_rewrite_of_a_known_factor_fails_at_the_gate_naming_it_as_known(against_known):
    record = _by_description(against_known["result"].failed)["std20"]
    assert record.stage == "gate"
    assert any("STD20 (Alpha158Stock, a known factor)" in r for r in record.reasons), record.reasons


def test_known_factors_are_in_the_record_of_what_the_graph_checks_cover(against_known):
    zoo = json.loads((against_known["run_dir"] / "zoo.json").read_text())
    assert zoo["not_covered"] == ["momentum_20 (AdjustedMomentum, a known factor)"]
    assert against_known["result"].not_covered == ("momentum_20 (AdjustedMomentum, a known factor)",)
    assert "KMID (Alpha158Stock)" in zoo["covered"] and "STD20 (Alpha158Stock, a known factor)" in zoo["covered"]
    # An output of the reference set is taken once, as a reference factor, even from a known factor
    # whose outputs are all in the reference set.
    assert not any(label.endswith("a known factor)") and label.split()[0] in KBAR for label in zoo["covered"])
    assert len(zoo["covered"]) == len(set(zoo["covered"])) == 169


def test_the_families_section_lists_the_known_classes_after_the_reference_families_marked_as_known(against_known):
    log = against_known["run_dir"] / "llm_log"
    for role in ("planner", "proposer"):
        prompt = json.loads(sorted(log.glob(f"*-{role}.json"))[0].read_text())["request"]["prompt"]
        reference = prompt.index("- Alpha158Stock: ")
        known = prompt.index("- Alpha158Stock (known, not in the reference set, refused when rewritten): ")
        momentum = prompt.index("- AdjustedMomentum (known, not in the reference set, refused when rewritten): ")
        assert reference < known < momentum, prompt
        assert f"({len(KBAR)} outputs: KMID," in prompt[reference:known] and "(160 outputs: " in prompt[known:momentum]


# Two rounds with KMID as the only known factor and no reference set: a noisy
# copy of KMID passes the Gate (it is no rewrite) but is refused at admission
# on its sampled correlation; the volume signal is accepted, its nearest
# known factor recorded; round 2 is told both.
KNOWN_ROUND_ONE = [
    _candidate("noisy kmid", "alpha = (close - open) / (high - low)"),
    _candidate("volume", "alpha = volume / WindowedAvg(volume, 20)"),
]
KNOWN_ROUND_TWO = [
    _candidate("noise", "alpha = BackRef(volume, 10) / WindowedAvg(volume, 60)"),
    _candidate("more noise", "alpha = BackRef(close, 20) / BackRef(WindowedAvg(close, 10), 5)"),
]
KMID_KNOWN = "KMID (Alpha158Stock, a known factor)"


def _against_kmid_known(root: Path, crash=None) -> dict:
    return _run(
        root, [KNOWN_ROUND_ONE, KNOWN_ROUND_TWO], known=[lambda prices, _: _alpha158(prices, ("KMID",))],
        crash=crash, round={"candidates": 2, "retries": 0},
    )


@pytest.fixture(scope="module")
def against_kmid_known(tmp_path_factory):
    return _against_kmid_known(tmp_path_factory.mktemp("kmid_known"))


def test_a_candidate_correlated_with_a_known_factor_is_rejected_at_admission_naming_it(against_kmid_known):
    record = _by_description(against_kmid_known["result"].rejected)["noisy kmid"]
    assert record.stage == "admission"
    assert record.metrics["nearest"] == ""
    correlation = record.metrics["nearest_known_correlation"]
    assert correlation > 0.7
    assert record.reasons == [
        f"cross-sectional correlation {correlation:+.2f} with {KMID_KNOWN}, sampled on one training date in 5, "
        f"is above 0.7 in absolute value"
    ]


def test_the_ledger_records_the_nearest_known_factor_of_every_evaluated_candidate(against_kmid_known):
    ledger = [json.loads(line) for line in (against_kmid_known["run_dir"] / "ledger.jsonl").read_text().splitlines()]
    metrics = {entry["candidate"]["description"]: entry["metrics"] for entry in ledger}
    assert set(metrics) == {"noisy kmid", "volume", "noise", "more noise"}
    assert all(m["nearest_known"] == KMID_KNOWN for m in metrics.values())
    assert abs(metrics["volume"]["nearest_known_correlation"]) < 0.2
    assert [r.candidate["description"] for r in against_kmid_known["result"].accepted] == ["volume"]


def test_the_last_rounds_feedback_gives_the_nearest_known_factor(against_kmid_known):
    record = _by_description(against_kmid_known["result"].rejected)["noisy kmid"]
    prompt = [r.prompt for r in against_kmid_known["fake"].of_role("proposer") if round_of(r.prompt) == 2][0]
    feedback = prompt.split("noisy kmid", 1)[1]
    assert (
        f"no existing factor to compare with; nearest known factor {KMID_KNOWN}, "
        f"sampled correlation {record.metrics['nearest_known_correlation']:+.2f}"
    ) in feedback


def test_a_resumed_campaign_with_known_factors_writes_the_same_ledger(against_kmid_known, tmp_path):
    resumed = _against_kmid_known(tmp_path, crash=lambda r: r.role == "reviewer" and round_of(r.prompt) == 2)
    assert {round_of(r.prompt) for r in resumed["fake"].requests} == {2}
    assert (resumed["run_dir"] / "ledger.jsonl").read_text() == (
        against_kmid_known["run_dir"] / "ledger.jsonl"
    ).read_text()


# One round against KMID neutralised, as the shared ``*_neutral`` stores hold
# their factors: the planner, the Gate and the Evaluator know it as the
# Alpha158 factor it neutralises.
NEUTRAL_ROUND = [
    _candidate("kmid", "alpha = (close - open) / open"),
    _candidate("noisy copy", "alpha = (close - open) / (high - low)"),
]


@pytest.fixture(scope="module")
def against_neutral_kmid(tmp_path_factory):
    def neutral_kmid(prices, exposures):
        return NeutralizedFactor(NeutralizedConfig(factor=_alpha158(prices, ("KMID",)), dataset=exposures, njobs=4))

    return _run(
        tmp_path_factory.mktemp("neutral_kmid"), [NEUTRAL_ROUND], reference=[neutral_kmid],
        round={"candidates": len(NEUTRAL_ROUND), "retries": 0},
    )


def test_a_neutralised_reference_factor_is_known_by_the_factor_it_neutralises(against_neutral_kmid):
    result, run_dir = against_neutral_kmid["result"], against_neutral_kmid["run_dir"]
    planner = json.loads(sorted((run_dir / "llm_log").glob("*-planner.json"))[0].read_text())["request"]["prompt"]
    assert "- Alpha158Stock: " in planner and "(1 outputs: KMID)" in planner
    records = _by_description(result.records)
    assert any("KMID (Alpha158Stock)" in r for r in records["kmid"].reasons), records["kmid"].reasons
    assert records["noisy copy"].metrics["nearest"] == "KMID (Alpha158Stock)"


# One round against KMID, the planted intraday signal up to a constant. Each
# candidate meets one admission check; the volume signal passes them all.
ADMISSION = [
    _candidate("noisy copy", "alpha = (close - open) / (high - low)"),
    _candidate("averaged copy", "alpha = WindowedAvg(close / open, 5)"),
    _candidate("incremental", "alpha = volume / WindowedAvg(volume, 20)"),
    _candidate("incremental again", "alpha = volume / WindowedAvg(volume, 10)"),
    _candidate("one decile", "alpha = Min(open, close) / low"),
    # close / open = exp(0.01 z), so this is Log(z - 1): a value on about one member in six.
    _candidate("sparse", "alpha = Log(Log(close / open) * 100 - 1)"),
]


@pytest.fixture(scope="module")
def admission(tmp_path_factory):
    outcome = _run(
        tmp_path_factory.mktemp("admission"), [ADMISSION],
        reference=[lambda prices, _: _alpha158(prices, ("KMID",))],
        round={"candidates": len(ADMISSION), "retries": 0},
    )
    result = outcome["result"]
    return {**outcome, "records": _by_description(result.accepted + result.rejected)}


#: The planted signal read backwards: a higher value goes with a lower return.
BACKWARDS = [_candidate("backwards", "alpha = open / close")]
#: A second round, so the first round's acceptance shows in the prompts.
AFTER = [_candidate("noise", "alpha = BackRef(volume, 10) / WindowedAvg(volume, 60)")]


@pytest.fixture(scope="module")
def backwards(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("backwards"), [BACKWARDS, AFTER], round={"candidates": 1, "retries": 0})


def test_a_signal_read_backwards_is_accepted_with_the_direction_its_training_ic_shows(backwards):
    [record] = [r for r in backwards["result"].accepted if r.candidate["description"] == "backwards"]
    assert record.metrics["ic_mean"] < -0.1
    assert "expected_sign" not in record.candidate


def test_the_prompts_name_an_accepted_factors_direction(backwards):
    prompt = backwards["fake"].of_role("proposer")[-1].prompt
    best = prompt.split("Current best mechanisms", 1)[1]
    assert "factor direction -1 (a higher value went with a lower return)" in best
    assert "expected sign" not in prompt.lower() and "expected_sign" not in prompt


def test_a_fix_hint_goes_to_the_fix_request_but_not_into_the_recorded_reasons(tmp_path):
    item = _candidate("integer input", "alpha = Sub(close / open, 1)")
    outcome = _run(tmp_path, [[item]], round={"candidates": 1, "retries": 1})
    fix_request = outcome["fake"].of_role("proposer")[1].prompt
    assert "write the number as ConstantOp(1)" in fix_request
    [record] = outcome["result"].failed
    assert any("Bad inputs, given" in r for r in record.reasons), record.reasons
    assert not any("ConstantOp(1)" in r for r in record.reasons), record.reasons


def _wrapped(number, failures):
    """A fix reply that echoes each refused entry with the fixed candidate nested under "candidate"."""
    return [
        {"number": i + 1, "direction": f["direction"], "refused_because": f["refused_because"],
         "candidate": {**FIXED, "mechanism_type": f["direction"]["mechanism_type"]}}
        for i, f in enumerate(failures)
    ]


def test_a_fixed_candidate_nested_under_candidate_in_the_reply_is_read(tmp_path):
    outcome = _run(tmp_path, [[_candidate("syntax error", "alpha = close /")]], fix=_wrapped,
                   round={"candidates": 1, "retries": 1})
    [record] = outcome["result"].records
    assert record.stage != "parse", record.reasons
    assert record.candidate["code"] == FIXED["code"]


def test_a_reply_item_of_another_shape_still_fails_to_parse(tmp_path):
    def nested_twice(number, failures):
        return [{"fixed": {"candidate": FIXED}} for _ in failures]

    outcome = _run(tmp_path, [[_candidate("syntax error", "alpha = close /")]], fix=nested_twice,
                   round={"candidates": 1, "retries": 1})
    [record] = outcome["result"].failed
    assert record.stage == "parse"
    assert "candidate fields: missing" in record.reasons[0]


def test_one_trial_is_judged_as_two_tests_one_per_direction(weak_alone, tmp_path):
    strict = _run(tmp_path, [[WEAK]], admission={"t_threshold": 3.3}, round={"candidates": 1, "retries": 0})
    [record] = strict["result"].rejected
    assert 3.3 < record.metrics["partial_ic_t"] < 3.49
    assert record.reasons == [
        f"the partial IC's Newey-West t {record.metrics['partial_ic_t']:.2f} (residualised against the reference "
        "set and the accepted factors, in the factor's direction) is below the threshold 3.49 at 1 evaluated "
        "trial(s), each counted in both directions"
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


def test_a_candidate_with_a_value_on_few_members_is_rejected_for_coverage(admission):
    record = admission["records"]["sparse"]
    assert record.status == "rejected"
    assert 0.1 < record.metrics["coverage"] < 0.25
    inner = "SubConst(MulConst(Log(Div(close, open)), 100), 1)"
    assert record.reasons == [
        f"coverage {record.metrics['coverage']:.2f} (the mean daily share of the members with a value) "
        f"is below 0.5; it falls at Log({inner}), to {record.metrics['coverage']:.2f} from 1.00 for {inner}"
    ]


def test_a_candidate_with_full_coverage_is_not_rejected_for_it(admission):
    record = admission["records"]["incremental"]
    assert record.metrics["coverage"] == pytest.approx(1.0)
    assert not any("coverage" in reason for reason in record.reasons)


def test_a_candidate_filling_a_sparse_value_is_rejected_for_the_coverage_of_its_data(tmp_path):
    filled = _candidate("filled", "alpha = SetInfOrNanToValue(Log(Log(close / open) * 100 - 1), 0)")
    outcome = _run(tmp_path, [[filled]], round={"candidates": 1, "retries": 0})
    [record] = outcome["result"].accepted + outcome["result"].rejected
    assert record.status == "rejected"
    assert 0.1 < record.metrics["coverage"] < 0.25
    [reason] = record.reasons
    assert reason.startswith("coverage ") and "is below 0.5; it falls at Log(SubConst(" in reason


def test_the_coverage_threshold_is_read_from_the_config(tmp_path):
    sparse = [c for c in ADMISSION if c["description"] == "sparse"]
    outcome = _run(tmp_path, [sparse], admission={"min_coverage": 0.1}, round={"candidates": 1, "retries": 0})
    [record] = outcome["result"].accepted + outcome["result"].rejected
    assert record.metrics["coverage"] < 0.5
    assert not any("coverage" in reason for reason in record.reasons)


def test_admission_metrics_are_in_the_ledger(admission):
    ledger = [json.loads(line) for line in (admission["run_dir"] / "ledger.jsonl").read_text().splitlines()]
    metrics = {entry["candidate"]["description"]: entry["metrics"] for entry in ledger}
    assert set(metrics) == {c["description"] for c in ADMISSION}
    assert len(metrics["incremental"]["decile_returns"]) == 10
    assert metrics["noisy copy"]["nearest_correlation"] > 0.7
    assert all(m["nearest_known"] == "" and math.isnan(m["nearest_known_correlation"]) for m in metrics.values())


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


def test_a_field_column_missing_from_the_price_dataset_fails_at_start_before_any_llm_call(tmp_path):
    fields = tmp_path / "fields.yaml"
    fields.write_text(PRICE_VOLUME.read_text().replace("column: adjVolume", "column: adjVolumeTypo"))
    config_path = write_config(tmp_path, synthetic_market.build(tmp_path / "data"), fields=[str(fields)])
    fake = FakeLLM(lambda number, directions: [])
    with pytest.raises(VocabularyError, match="adjVolumeTypo"):
        run(config_path, fake, tmp_path / "kunquant_cache")
    assert fake.requests == []


def test_each_admission_reason_is_counted_under_its_kind():
    from factor_agent._admission import reason_kind, reasons
    from factor_agent.config import AdmissionConfig
    from factor_agent.evaluator import Metrics

    good = dict(
        ic_mean=0.05, ic_t=8.0, partial_ic_mean=0.04, partial_ic_t=8.0, nearest="x", nearest_correlation=0.1,
        nearest_known="y", nearest_known_correlation=-0.2, decile_returns=tuple(range(10)), coverage=0.9, n_days=100, start="s", end="e",
    )
    cases = {
        "coverage": {"coverage": 0.1},
        "no direction": {"ic_mean": 0.0},
        "correlation": {"nearest_correlation": 0.99},
        "correlation with a known factor": {"nearest_known_correlation": -0.99},
        "partial IC": {"partial_ic_t": 0.5},
        "monotonicity": {"decile_returns": tuple(range(10, 0, -1))},
    }
    config = AdmissionConfig()
    assert reasons(Metrics(**good), 1, config) == []
    for kind, change in cases.items():
        [reason] = reasons(Metrics(**{**good, **change}), 1, config)
        assert reason_kind(reason) == kind.removesuffix(" with a known factor"), reason
