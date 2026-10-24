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
from quantlab.factor.kunquant_compile import CACHE_ENV
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
    _candidate("noise", "alpha = WindowedAvg(volume, 5)"),
    _candidate("syntax error", "alpha = close /"),
    _candidate("unknown name", "alpha = SecretOp(close)"),
    _candidate("import", "import os\nalpha = close"),
    _candidate("two outputs", "alpha = (close / open, volume)"),
    _candidate("endless loop", "while True:\n    pass\nalpha = close"),
    _candidate("bad schema", "alpha = close", mechanism_type="astrology"),
    _candidate("duplicate", "x = close\ny = open\nalpha = x / y"),
]

FIXED = _candidate("syntax error", "alpha = Rank(WindowedAvg(high / low, 5))")

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


def test_the_round_is_compiled_once(outcome):
    entries = [p for p in Path(outcome["cache"]).iterdir() if not p.name.startswith(".")]
    assert len(entries) == 1
