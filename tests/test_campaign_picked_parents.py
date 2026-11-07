"""Picked parents in a round: with ``round.picked_parents`` the code chooses some of the slots' parents.

After the quota, the campaign builds the Trial graph of the evaluated
trials (``_trial_graph``) and gives each, in pick order, a slot of its mechanism
type while that type has one unclaimed, up to ``round.picked_parents``. The
planner is told the picked slots and writes their action and idea; a picked
slot it leaves out is planned as a refinement of its parent, with an issue.
Every role is answered by ``tests/fake_llm.FakeLLM``; the Gate and the
Evaluator are the real ones on the synthetic store.
"""

import json

import pytest
import yaml

from factor_agent import _trial_graph, events
from factor_agent._feedback import named
from factor_agent.campaign import Campaign
from factor_agent.config import CampaignConfig
from factor_agent.prompts import PLAN, PromptError
from factor_agent.snapshot import SnapshotMismatch
from tests.campaign_runner import run, write_config
from tests.fake_llm import FakeLLM, has_tag, new_directions, round_of, tagged
from tests.test_campaign_loop import market, noise  # noqa: F401  (the module's synthetic store)

#: One slot of each of the five default mechanism types a round.
ROUND = {"candidates": 5, "min_per_type": 1, "retries": 0, "max_rounds": 2}
THEMES = {"theme_correlation": 0.5}


def _propose(number, directions):
    return [noise(number, slot) for slot in range(len(directions))]


def _follow_picks(number, quota, prompt):
    """The planner: a mutation of each picked slot's parent in its type, new directions elsewhere."""
    directions = new_directions(number, quota, prompt)
    if not has_tag(prompt, "picked_slots"):
        return directions
    for slot in tagged(prompt, "picked_slots"):
        [i] = [i for i, d in enumerate(directions) if d["mechanism_type"] == slot["mechanism_type"]]
        directions[i] = {**directions[i], "action": "mutation", "parent": slot["parent"],
                         "direction": f"another channel {slot['parent']} points to"}
    return directions


def _config(root, market, round_settings=None, **sections):  # noqa: F811
    return write_config(root, market["components"], round={**ROUND, **(round_settings or {})}, **sections)


def _plans(run_dir):
    return [e for e in events.read(run_dir / events.FILE) if e.type == "plan"]


@pytest.fixture(scope="module")
def picked(tmp_path_factory, market):  # noqa: F811
    fake = FakeLLM(_propose, plan=_follow_picks)
    root = tmp_path_factory.mktemp("picked")
    result = run(_config(root, market, {"picked_parents": 2}, context=THEMES), fake, market["cache"])
    return {"result": result, "fake": fake, "run_dir": root / "run"}


def test_the_planner_is_told_two_picked_slots_each_of_its_parents_type(picked):
    records = picked["result"].records
    first, second = (r.prompt for r in picked["fake"].of_role("planner"))
    assert not has_tag(first, "picked_slots")  # nothing was evaluated before round 1
    slots = tagged(second, "picked_slots")
    assert len(slots) == 2 and len({s["mechanism_type"] for s in slots}) == 2  # each type has one slot
    type_of = {f"T1-{slot:02d}": r.direction["mechanism_type"] for slot, r in enumerate(records[:5], 1)}
    for s in slots:
        assert s["mechanism_type"] == type_of[s["parent"]]
    assert "as a refinement or a mutation of its parent" in second


def test_the_plan_event_records_the_picked_parents_with_their_utilities_and_fronts(picked):
    first, second = _plans(picked["run_dir"])
    assert first.picked == []
    slots = tagged(picked["fake"].of_role("planner")[1].prompt, "picked_slots")
    assert [p["trial"] for p in second.picked] == [s["parent"] for s in slots]
    for p in second.picked:
        assert set(p) == {"trial", "mechanism_type", "quality", "fertility", "novelty", "front"}
        assert p["front"] >= 1
    assert second.issues == []


def test_a_planned_direction_naming_a_picked_parent_fills_its_slot(picked):
    slots = tagged(picked["fake"].of_role("planner")[1].prompt, "picked_slots")
    second = [r for r in picked["result"].records if r.round == 2]
    for s in slots:
        [record] = [r for r in second if r.direction["mechanism_type"] == s["mechanism_type"]]
        assert record.direction["action"] == "mutation" and record.direction["parent"] == s["parent"]
    snapshot = json.loads((picked["run_dir"] / "snapshot.json").read_text())
    assert snapshot["settings"]["round"] == {"picked_parents": 2}


def _claimed(graph, quota, records, k):
    """The picks the claiming rule makes from ``graph``: in pick order, one per slot of the parent's type, up to ``k``."""
    type_of = {trial: r.direction["mechanism_type"] for trial, r in named(records)}
    left, picks = dict(quota), []
    for pick in graph.pick_order():
        if left.get(type_of[pick.trial], 0) > 0 and len(picks) < k:
            left[type_of[pick.trial]] -= 1
            picks.append(pick.trial)
    return picks


def test_each_round_writes_its_trial_graph_whose_pick_order_gives_the_plans_picks(picked):
    run_dir, records = picked["run_dir"], picked["result"].records
    assert sorted(p.name for p in (run_dir / "trial_graph").iterdir()) == ["round_01.json", "round_02.json"]
    first, second = _plans(run_dir)
    assert _trial_graph.read(run_dir, 1).nodes == ()  # nothing was evaluated before round 1
    graph = _trial_graph.read(run_dir, 2)
    assert [n.trial for n in graph.nodes] == [f"T1-{slot:02d}" for slot in range(1, 6)]
    assert _claimed(graph, second.quota, records, 2) == [p["trial"] for p in second.picked]
    assert _trial_graph.read(run_dir, 3) is None


def test_a_round_planned_again_after_a_resume_writes_its_trial_graph_anew(tmp_path, market):  # noqa: F811
    path = _config(tmp_path, market, {"picked_parents": 2}, context=THEMES)
    exhausted = FakeLLM(_propose, plan=_follow_picks, quota=lambda r: r.role == "reviewer" and round_of(r.prompt) == 2)
    run(path, exhausted, market["cache"])
    kept = tmp_path / "run" / "trial_graph" / "round_02.json"
    assert _trial_graph.read(tmp_path / "run", 2).nodes  # written after the plan, before the round was dropped
    kept.write_text("{}")
    result = run(path, FakeLLM(_propose, plan=_follow_picks), market["cache"])
    second = _plans(tmp_path / "run")[-1]
    assert second.round == 2
    assert _claimed(_trial_graph.read(tmp_path / "run", 2), second.quota, result.records, 2) == [
        p["trial"] for p in second.picked
    ]


def test_a_picked_slot_the_planner_leaves_out_is_planned_as_a_refinement_of_its_parent(tmp_path, market):  # noqa: F811
    fake = FakeLLM(_propose)  # every direction new
    run(_config(tmp_path, market, {"picked_parents": 1}, context=THEMES), fake, market["cache"])
    [p] = _plans(tmp_path / "run")[1].picked
    plan = _plans(tmp_path / "run")[1]
    [direction] = [d for d in plan.directions if d["mechanism_type"] == p["mechanism_type"]]
    assert direction["action"] == "refinement" and direction["parent"] == p["trial"]
    assert plan.issues == [  # the type's one slot is the picked one, so its new direction is dropped
        f"planner: the picked {p['mechanism_type']} slot of {p['trial']} is not planned; "
        f"it is planned as a refinement of {p['trial']}",
        f"planner: 1 {p['mechanism_type']} directions besides the picked ones for 0 slots; the first are kept",
    ]


def test_without_picked_parents_the_planner_and_the_snapshot_are_as_before(tmp_path, market):  # noqa: F811
    fake = FakeLLM(_propose)
    run(_config(tmp_path, market, context=THEMES), fake, market["cache"])
    assert not any(has_tag(r.prompt, "picked_slots") for r in fake.of_role("planner"))
    assert all(plan.picked == [] for plan in _plans(tmp_path / "run"))
    assert not (tmp_path / "run" / "trial_graph").exists()
    snapshot = json.loads((tmp_path / "run" / "snapshot.json").read_text())
    assert "round" not in snapshot["settings"]


def test_picked_parents_need_themes(tmp_path, market):  # noqa: F811
    with pytest.raises(ValueError, match="picked_parents needs context.theme_correlation"):
        CampaignConfig.from_file(_config(tmp_path, market, {"picked_parents": 2}))
    with pytest.raises(ValueError, match="picked_parents"):
        CampaignConfig.from_file(_config(tmp_path, market, {"picked_parents": -1}, context=THEMES))


def _without_placeholder(tmp_path, market):  # noqa: F811
    template = tmp_path / "plan.txt"
    template.write_text(PLAN.replace("{picked}", ""))
    return _config(tmp_path, market, {"picked_parents": 2}, context=THEMES, prompts={"plan": str(template)})


def test_a_replaced_plan_template_without_the_picked_slots_is_refused_at_start(tmp_path, market):  # noqa: F811
    fake = FakeLLM(_propose)
    with pytest.raises(PromptError, match=r"lacks the placeholder \{picked\}"):
        run(_without_placeholder(tmp_path, market), fake, market["cache"])
    assert fake.requests == [] and not (tmp_path / "run").exists()


def test_a_check_reports_a_replaced_plan_template_without_the_picked_slots(tmp_path, market):  # noqa: F811
    report = Campaign(CampaignConfig.from_file(_without_placeholder(tmp_path, market))).check()
    [problem] = [p for p in report.problems if not p.startswith("llm.")]  # the scripted roles' replies file is unmade
    assert problem.startswith("templates: ") and "lacks the placeholder {picked}" in problem


def test_a_replaced_plan_template_without_the_placeholder_is_taken_when_picked_parents_are_off(tmp_path, market):  # noqa: F811
    template = tmp_path / "plan.txt"
    template.write_text(PLAN.replace("{picked}", ""))
    path = _config(tmp_path, market, {"max_rounds": 1}, prompts={"plan": str(template)})
    assert [p for p in Campaign(CampaignConfig.from_file(path)).check().problems if not p.startswith("llm.")] == []


def test_a_resume_may_not_change_picked_parents(tmp_path, market):  # noqa: F811
    path = _config(tmp_path, market, {"picked_parents": 2, "max_rounds": 1}, context=THEMES)
    run(path, FakeLLM(_propose, plan=_follow_picks), market["cache"])
    config = yaml.safe_load(path.read_text())
    config["round"].update(max_rounds=2, picked_parents=3)
    path.write_text(yaml.safe_dump(config))
    fake = FakeLLM(_propose, plan=_follow_picks)
    with pytest.raises(SnapshotMismatch, match=r"round\.picked_parents"):
        run(path, fake, market["cache"])
    assert fake.requests == []
