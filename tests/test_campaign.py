"""Campaign.run(): one round end to end, with a scripted LLM as the only fake.

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
]


@pytest.fixture(scope="module")
def outcome(tmp_path_factory):
    root = tmp_path_factory.mktemp("campaign")
    components = synthetic_market.build(root / "data")
    replies = root / "replies.json"
    reply = "Here are the candidates.\n```json\n" + json.dumps(CANDIDATES, indent=2) + "\n```\n"
    replies.write_text(json.dumps([reply]))
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


def test_the_planted_signal_is_accepted_and_the_rest_rejected(outcome):
    result = outcome["result"]
    assert [r.candidate["description"] for r in result.accepted] == ["planted"]
    assert sorted(r.candidate["description"] for r in result.rejected) == sorted(
        c["description"] for c in CANDIDATES[1:]
    )
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
        ("syntax error", "syntax error"),
        ("unknown name", "SecretOp"),
        ("import", "import"),
        ("two outputs", "exactly one output"),
        ("endless loop", "timed out"),
    ],
)
def test_a_gate_rejection_never_reaches_the_evaluator(outcome, description, reason):
    record = _by_description(outcome["result"].rejected)[description]
    assert record.stage == "gate"
    assert record.metrics is None
    assert any(reason in r for r in record.reasons), record.reasons


def test_a_reply_item_that_breaks_the_schema_is_rejected_when_parsed(outcome):
    record = _by_description(outcome["result"].rejected)["bad schema"]
    assert record.stage == "parse"
    assert any("astrology" in r for r in record.reasons)


def test_only_the_training_segment_is_evaluated(outcome):
    result = outcome["result"]
    evaluated = [r for r in result.accepted + result.rejected if r.metrics is not None]
    assert {r.candidate["description"] for r in evaluated} == {"planted", "noise"}
    for record in evaluated:
        assert record.metrics["start"] >= synthetic_market.TRAINING[0]
        assert record.metrics["end"] <= synthetic_market.TRAINING[1]


def test_every_candidate_is_in_the_ledger_with_its_metrics(outcome):
    ledger = {entry["candidate"]["description"]: entry for entry in outcome["ledger"]}
    assert set(ledger) == {c["description"] for c in CANDIDATES}
    assert ledger["planted"]["status"] == "accepted"
    assert ledger["planted"]["metrics"]["ic_t"] >= 3.0
    assert ledger["planted"]["fingerprint"]
    assert ledger["noise"]["status"] == "rejected"
    assert ledger["noise"]["metrics"]["n_days"] > 400
    assert all(entry["round"] == 1 for entry in outcome["ledger"])


def test_the_round_is_compiled_once(outcome):
    entries = [p for p in Path(outcome["cache"]).iterdir() if not p.name.startswith(".")]
    assert len(entries) == 1
