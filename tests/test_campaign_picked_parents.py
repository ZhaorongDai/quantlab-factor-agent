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

from factor_agent import events
from factor_agent.campaign import Campaign
from factor_agent.config import ROLES, CampaignConfig
from factor_agent.prompts import PLAN, PromptError
from factor_agent.snapshot import SnapshotMismatch
from tests.campaign_runner import run, write_config
from tests.fake_llm import FakeLLM, has_tag, new_directions, tagged
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


def test_the_snapshot_records_every_trial_graph_value_and_not_the_default_fertile_children(picked):
    settings = json.loads((picked["run_dir"] / "snapshot.json").read_text())["settings"]
    assert settings["trial_graph"] == {  # the paper's
        "gamma": 0.75, "iterations": 100, "tolerance": 1e-8,
        "weights": {"generation": 1 / 3, "structure": 1 / 3, "correlation": 1 / 3},
        "min_jaccard": 0.2, "structure_neighbours": 30, "min_correlation": 0.1, "correlation_neighbours": 30,
        "nearest_accepted": 10, "max_expansions": 5,
    }
    assert "fertile_children" not in settings["context"]


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
    snapshot = json.loads((tmp_path / "run" / "snapshot.json").read_text())
    assert "round" not in snapshot["settings"] and "trial_graph" not in snapshot["settings"]
    assert "fertile_children" not in snapshot["settings"]["context"]


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


@pytest.mark.parametrize("section, value, says", [
    ("trial_graph", {"gamma": 0.5}, r"trial_graph\.gamma was 0\.75, now 0\.5"),
    ("context", {"fertile_children": 2}, r"context\.fertile_children was not recorded, now 2"),
])
def test_a_resume_may_not_change_the_trial_graph_or_the_fertile_children(tmp_path, market, section, value, says):  # noqa: F811
    path = _config(tmp_path, market, {"picked_parents": 2, "max_rounds": 1}, context=THEMES)
    run(path, FakeLLM(_propose, plan=_follow_picks), market["cache"])
    config = yaml.safe_load(path.read_text())
    config["round"]["max_rounds"] = 2
    config[section] = {**config.get(section, {}), **value}
    path.write_text(yaml.safe_dump(config))
    fake = FakeLLM(_propose, plan=_follow_picks)
    with pytest.raises(SnapshotMismatch, match=says):
        run(path, fake, market["cache"])
    assert fake.requests == []


def test_a_replay_whose_config_omits_the_trial_graph_picks_as_the_recorded_run(tmp_path, market):  # noqa: F811
    changed = {"gamma": 0.5, "max_expansions": 2}
    recorded = _config(tmp_path / "recorded", market, {"picked_parents": 2},
                       context={**THEMES, "fertile_children": 1}, trial_graph=changed)
    run(recorded, FakeLLM(_propose, plan=_follow_picks), market["cache"])
    log = tmp_path / "recorded" / "run" / "llm_log"
    replay = _config(tmp_path / "replayed", market, {"picked_parents": 2}, context=THEMES,
                     llm={role: {"backend": "replay", "model": f"model-{role}", "log": str(log)} for role in ROLES})
    run(replay, None, market["cache"])
    assert [p.picked for p in _plans(tmp_path / "replayed" / "run")] == [p.picked for p in _plans(tmp_path / "recorded" / "run")]
    settings = json.loads((tmp_path / "replayed" / "run" / "snapshot.json").read_text())["settings"]
    assert settings["trial_graph"]["gamma"] == 0.5 and settings["trial_graph"]["max_expansions"] == 2
    assert settings["context"]["fertile_children"] == 1
