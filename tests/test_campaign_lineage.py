"""A direction builds on an earlier trial named by its id, and a run without feedback reads none.

Every trial is named ``T<round>-<slot>`` in the prompts. The planner may name
any evaluated trial, accepted or refused, as the parent of a refinement or a
mutation, and the proposer is given that trial's code, statistics and reasons.
With ``context.feedback`` off, no planner or proposer prompt holds an earlier
trial and no memory is distilled: the control arm of a test of whether the
agent learns from its trials. With ``context.lineage`` the planner's
accepted factors and the proposer's parent block carry each trial's Lineage:
how the trials built on it did.
"""

import json

import pytest
import yaml

from factor_agent.config import ContextConfig
from factor_agent.snapshot import SnapshotMismatch
from tests.campaign_runner import candidate, run, write_config
from tests.fake_llm import FakeLLM, has_tag, new_directions, round_of, tagged
from tests.test_campaign_loop import market, noise  # noqa: F401  (the module's synthetic store)

REFINED = "alpha = BackRef(close, 3) / BackRef(WindowedAvg(close, 30), 3)"


def _refine_first_slot(parent: str):
    """A plan whose first direction refines ``parent`` from round 2 on, the rest new."""
    def plan(number, quota, prompt):
        directions = new_directions(number, quota, prompt)
        if number > 1:
            directions[0] = {**directions[0], "action": "refinement", "parent": parent,
                             "direction": "measure the same mechanism over a longer window"}
        return directions
    return plan


def _propose(number, directions):
    return [
        candidate("a refinement", REFINED) if number > 1 and slot == 0 else noise(number, slot)
        for slot, _ in enumerate(directions)
    ]


def _campaign(tmp_path, market, fake, context=None, **round_settings):  # noqa: F811
    sections = {"round": {"candidates": 10, "min_per_type": 1, "retries": 0, "max_rounds": 2, **round_settings}}
    if context is not None:
        sections["context"] = context
    return run(write_config(tmp_path, market["components"], **sections), fake, market["cache"])


@pytest.fixture(scope="module")
def refining(tmp_path_factory, market):  # noqa: F811
    fake = FakeLLM(_propose, plan=_refine_first_slot("T1-02"))
    result = _campaign(tmp_path_factory.mktemp("refining"), market, fake)
    return {"result": result, "fake": fake}


def test_the_planner_sees_every_earlier_trial_under_its_id(refining):
    planner = refining["fake"].of_role("planner")[1].prompt
    history = tagged_text(planner, "recent_history")
    assert "- T1-01, round 1," in history and "- T1-10, round 1," in history
    assert "\n2. T1-02 [" in tagged_text(planner, "last_round")


def test_a_direction_refining_a_refused_trial_keeps_its_action_parent_and_text(refining):
    records = refining["result"].records
    assert records[1].status != "accepted" and records[1].evaluated
    refined = [r for r in records if r.round == 2][0]
    assert refined.direction == {
        "mechanism_type": refined.direction["mechanism_type"], "action": "refinement", "parent": "T1-02",
        "direction": "measure the same mechanism over a longer window",
    }


def test_the_proposer_is_given_the_parent_trials_code_statistics_and_reasons(refining):
    [prompt] = [r.prompt for r in refining["fake"].of_role("proposer") if "Round: 2" in r.prompt]
    first = tagged(prompt, "directions")[0]
    parent = first["parent_trial"]
    assert parent.startswith("T1-02, round 1, ")
    assert noise(1, 1)["code"] in parent
    assert "Newey-West t" in parent and "reasons: " in parent
    assert "parent_trial" not in tagged(prompt, "directions")[1]


def test_a_direction_naming_no_evaluated_trial_keeps_its_text_as_a_new_direction(tmp_path, market):  # noqa: F811
    fake = FakeLLM(_propose, plan=_refine_first_slot("the factor I liked"))
    records = _campaign(tmp_path, market, fake).records
    refined = [r for r in records if r.round == 2][0]
    assert refined.direction["action"] == "new" and refined.direction["parent"] == ""
    assert refined.direction["direction"] == "measure the same mechanism over a longer window"


@pytest.fixture(scope="module")
def without_feedback(tmp_path_factory, market):  # noqa: F811
    fake = FakeLLM(_propose, plan=_refine_first_slot("T1-02"))
    result = _campaign(
        tmp_path_factory.mktemp("without_feedback"), market, fake,
        context={"feedback": False}, max_rounds=3, memory_every=1,
    )
    return {"result": result, "fake": fake}


def test_without_feedback_no_planner_or_proposer_prompt_holds_an_earlier_trial(without_feedback):
    fake = without_feedback["fake"]
    prompts = [r.prompt for role in ("planner", "proposer") for r in fake.of_role(role)]
    assert len(prompts) > 3
    for prompt in prompts:
        assert not any(has_tag(prompt, tag) for tag in ("memory", "recent_history", "last_round", "best", "earlier_summary"))
        assert "T1-" not in prompt and noise(1, 1)["description"] not in prompt


def test_without_feedback_no_memory_is_distilled_and_every_direction_is_new(without_feedback):
    assert without_feedback["fake"].of_role("distiller") == []
    assert {r.direction["action"] for r in without_feedback["result"].records} == {"new"}


def test_without_feedback_the_planner_is_told_every_direction_is_new(without_feedback):
    planner = without_feedback["fake"].of_role("planner")[1].prompt
    assert "Every research direction is new" in planner and "measured another\n  way" not in planner


def tagged_text(prompt: str, tag: str) -> str:
    return prompt.split(f"<{tag}>\n", 1)[1].split(f"\n</{tag}>", 1)[0]


#: Round 2's first candidate: the volume shock of round 1 (accepted) with a second term, so the pool
#: explains part of it and names the accepted trial.
OVERLAPPING = "alpha = volume / WindowedAvg(volume, 20) + close / open"


def _overlap_plan(number, quota, prompt):
    directions = new_directions(number, quota, prompt)
    if number == 2:
        directions[0] = {**directions[0], "action": "refinement", "parent": "T1-01",
                         "direction": "the volume shock with the day's move"}
    return directions


def _overlap_propose(number, directions):
    if number == 1:
        return [candidate("volume shock", "alpha = volume / WindowedAvg(volume, 20)")] + [
            noise(number, slot) for slot in range(1, len(directions))]
    if number == 2:
        return [candidate("volume shock with the move", OVERLAPPING)] + [
            noise(number, slot) for slot in range(1, len(directions))]
    return [noise(number, slot) for slot in range(len(directions))]


@pytest.fixture(scope="module")
def overlapping(tmp_path_factory, market):  # noqa: F811
    fake = FakeLLM(_overlap_propose, plan=_overlap_plan)
    result = _campaign(tmp_path_factory.mktemp("overlapping"), market, fake, max_rounds=3)
    return {"result": result, "fake": fake}


def test_the_feedback_says_how_much_of_a_candidate_the_pool_explains_and_names_the_trials_by_id(overlapping):
    records = overlapping["result"].records
    assert records[0].status == "accepted"
    child = [r for r in records if r.round == 2][0]
    assert child.metrics["pool_share"] > 0.05
    assert child.metrics["overlaps"][0][0] == "volume shock (accepted in round 1)"
    planner = overlapping["fake"].of_role("planner")[2].prompt
    last = tagged_text(planner, "last_round")
    first = last.split("\n2. ")[0]
    assert "of its daily cross-section is explained by the reference set and the accepted factors" in first
    assert "most correlated: T1-01 +" in first


def test_a_childs_history_line_shows_its_parents_residual_t(overlapping):
    planner = overlapping["fake"].of_role("planner")[2].prompt
    [line] = [l for l in tagged_text(planner, "recent_history").splitlines() if l.startswith("- T2-01,")]
    assert "refinement of T1-01 (whose residual t was " in line
    assert "training IC" in line and "(t " in line


def test_the_feedback_shows_the_order_of_the_deciles_returns_and_no_return(overlapping):
    records = overlapping["result"].records
    child = [r for r in records if r.round == 2][0]
    deciles = child.metrics["decile_returns"]
    order = sorted(range(10), key=lambda i: deciles[i])
    ranks = [order.index(i) + 1 for i in range(10)]
    planner = overlapping["fake"].of_role("planner")[2].prompt
    first = tagged_text(planner, "last_round").split("\n2. ")[0]
    assert "the deciles' returns ranked from 1 (lowest) to 10, lowest factor values first: " + " ".join(map(str, ranks)) in first
    assert f"{deciles[0]:.4f}" not in planner


def _lineage_plan(number, quota, prompt):
    """Every round after the first refines T1-01 (the volume shock, accepted) in its first slot."""
    directions = new_directions(number, quota, prompt)
    if number > 1:
        directions[0] = {**directions[0], "action": "refinement", "parent": "T1-01",
                         "direction": "the volume shock with the day's move"}
    return directions


@pytest.fixture(scope="module")
def with_lineage(tmp_path_factory, market):  # noqa: F811
    fake = FakeLLM(_overlap_propose, plan=_lineage_plan)
    root = tmp_path_factory.mktemp("with_lineage")
    result = _campaign(root, market, fake, context={"lineage": True}, max_rounds=3)
    return {"result": result, "fake": fake, "root": root}


def test_with_lineage_the_planners_accepted_factors_carry_each_ones_lineage(with_lineage):
    records = with_lineage["result"].records
    assert records[0].status == "accepted"
    child = [r for r in records if r.round == 2][0]
    assert child.evaluated and child.status != "accepted"
    second, third = (r.prompt for r in with_lineage["fake"].of_role("planner")[1:3])
    assert "  lineage: no trial has built on it yet" in tagged_text(second, "best")
    t = child.metrics["partial_ic_t"] * (1 if child.metrics["ic_mean"] > 0 else -1)
    own = records[0].metrics["partial_ic_t"] * (1 if records[0].metrics["ic_mean"] > 0 else -1)
    assert (
        "  lineage: 1 trial built on it (1 refinement, 0 mutations), 0 stopped before evaluation, none accepted; "
        f"best evaluated child T2-01, residual t in its direction {t:+.2f} against this trial's {own:+.2f}"
    ) in tagged_text(third, "best")


def test_with_lineage_the_proposers_parent_block_carries_the_parents_lineage(with_lineage):
    proposers = {round_of(r.prompt): r.prompt for r in with_lineage["fake"].of_role("proposer")}
    assert tagged(proposers[2], "directions")[0]["parent_trial"].endswith("\n   lineage: no trial has built on it yet")
    assert "\n   lineage: 1 trial built on it (1 refinement, 0 mutations)" in tagged(proposers[3], "directions")[0]["parent_trial"]
    snapshot = json.loads((with_lineage["root"] / "run" / "snapshot.json").read_text())
    assert snapshot["settings"]["context"]["lineage"] is True


def test_without_lineage_no_prompt_carries_a_lineage_and_the_snapshot_does_not_record_it(overlapping):
    fake = overlapping["fake"]
    assert not any("lineage:" in r.prompt for role in ("planner", "proposer") for r in fake.of_role(role))
    snapshot = json.loads((overlapping["result"].ledger_path.parent / "snapshot.json").read_text())
    assert "lineage" not in snapshot["settings"]["context"]


def test_a_resume_may_not_switch_lineage_on_and_a_run_without_it_resumes(tmp_path, market):  # noqa: F811
    path = write_config(
        tmp_path, market["components"], round={"candidates": 10, "min_per_type": 1, "retries": 0, "max_rounds": 1},
    )
    run(path, FakeLLM(_overlap_propose, plan=_lineage_plan), market["cache"])
    config = yaml.safe_load(path.read_text())
    config["round"]["max_rounds"] = 2
    path.write_text(yaml.safe_dump(config))
    assert run(path, FakeLLM(_overlap_propose, plan=_lineage_plan), market["cache"]).rounds == 2
    config["round"]["max_rounds"], config["context"] = 3, {"lineage": True}
    path.write_text(yaml.safe_dump(config))
    fake = FakeLLM(_overlap_propose, plan=_lineage_plan)
    with pytest.raises(SnapshotMismatch, match=r"context\.lineage"):
        run(path, fake, market["cache"])
    assert fake.requests == []


def test_lineage_needs_the_earlier_trials():
    with pytest.raises(ValueError, match="lineage needs feedback"):
        ContextConfig(feedback=False, lineage=True)
    with pytest.raises(TypeError, match="lineage"):
        ContextConfig(lineage="yes")
