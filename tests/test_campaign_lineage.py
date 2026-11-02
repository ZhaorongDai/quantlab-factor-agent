"""A direction builds on an earlier trial named by its id, and a run without feedback reads none.

Every trial is named ``T<round>-<slot>`` in the prompts. The planner may name
any evaluated trial, accepted or refused, as the parent of a refinement or a
mutation, and the proposer is given that trial's code, statistics and reasons.
With ``context.feedback`` off, no planner or proposer prompt holds an earlier
trial and no memory is distilled: the control arm of a test of whether the
agent learns from its trials.
"""

import pytest

from tests.campaign_runner import candidate, run, write_config
from tests.fake_llm import FakeLLM, has_tag, new_directions, tagged
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
