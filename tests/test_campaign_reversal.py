"""A refuted hypothesis: its training IC has the opposite sign to the one it stated.

The trial is rejected and never flipped, but the loop can follow it up: the
planner sees it among the refuted hypotheses and may plan a ``reversal`` of
it, a mechanism that predicts the observed direction. Every role is
answered by ``tests/fake_llm.FakeLLM`` on the synthetic store, where
``close / open`` predicts a higher return.
"""

import json

import pytest

from tests import synthetic_market
from tests.campaign_runner import candidate, run, write_config
from tests.fake_llm import FakeLLM, new_directions

#: Round 1's refuted hypothesis: the planted intraday signal, stated with the wrong sign.
REFUTED = candidate("intraday move read backwards", "alpha = close / open", expected_sign=-1)
#: Its label among the refuted hypotheses.
LABEL = "intraday move read backwards (refuted in round 1)"
#: Noise for the other slots, one code per round and slot.
NOISE_CODES = [f"alpha = BackRef(volume, {lag}) / WindowedAvg(volume, 60)" for lag in (5, 10, 20, 40, 60, 120, 250)]


def _plan(number, quota, prompt):
    directions = new_directions(number, quota, prompt)
    if number == 2 and LABEL in prompt:
        directions[0] = {
            "mechanism_type": directions[0]["mechanism_type"], "action": "reversal", "parent": LABEL,
            "direction": "strong intraday moves continue rather than reverse",
        }
    return directions


def _propose(number, directions):
    items = []
    for slot, direction in enumerate(directions):
        if number == 1 and slot == 0:
            items.append(REFUTED)
        elif direction["action"] == "reversal":
            items.append(candidate("intraday continuation", "alpha = Log(close / open)"))
        else:
            items.append(candidate(f"noise {number} {slot}", NOISE_CODES[(number - 1) * 2 + slot % 2]))
    return items


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    root = tmp_path_factory.mktemp("reversal")
    components = synthetic_market.build(root / "data")
    path = write_config(
        root, components,
        round={"candidates": 5, "min_per_type": 1, "retries": 0, "max_rounds": 2, "memory_every": 1},
    )
    fake = FakeLLM(_propose, plan=_plan)
    result = run(path, fake, root / "kunquant_cache")
    return {"fake": fake, "result": result, "run_dir": root / "run"}


def _records(campaign):
    return [json.loads(line) for line in (campaign["run_dir"] / "ledger.jsonl").read_text().splitlines()]


def test_a_refuted_hypothesis_is_rejected_with_the_reason_saying_it_may_be_followed_up(campaign):
    [refuted] = [r for r in _records(campaign) if r["candidate"]["description"] == REFUTED["description"]]
    assert refuted["status"] == "rejected"
    assert any(
        "opposite sign to the hypothesis" in reason and "a reversal of it" in reason
        for reason in refuted["reasons"]
    ), refuted["reasons"]


def test_the_planner_sees_the_refuted_hypothesis_as_a_reversal_parent(campaign):
    prompt = campaign["fake"].of_role("planner")[1].prompt
    section = prompt.split("Refuted hypotheses", 1)[1]
    assert LABEL in section
    assert '"reversal"' in prompt


def test_the_history_keeps_the_stated_sign_and_the_opposite_sign_category(campaign):
    prompt = campaign["fake"].of_role("planner")[1].prompt
    history = prompt.split("All history", 1)[1].split("\n## ", 1)[0]
    [line] = [line for line in history.splitlines() if REFUTED["description"] in line]
    assert "expected sign -1" in line and "opposite sign" in line, line


def test_a_reversal_of_a_refuted_hypothesis_is_planned_and_evaluated(campaign):
    [reversal] = [r for r in _records(campaign) if r["direction"]["action"] == "reversal"]
    assert reversal["direction"]["parent"] == LABEL
    assert reversal["metrics"] is not None


def test_the_distiller_may_record_which_way_a_mechanism_worked(campaign):
    prompt = campaign["fake"].of_role("distiller")[0].prompt
    assert "Directional findings" in prompt
    assert "opposite sign" in prompt
