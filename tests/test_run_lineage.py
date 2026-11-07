"""A run's Lineage as the web panel reads it: each trial's Lineage, the rounds it was picked in, and any round's plan.

The run has three rounds of five slots, one per default mechanism type, and
two Picked parents a round. The planner fills each picked slot with a
mutation of its parent and, in round 3, builds the first slot of a type
without a picked one on a round-1 trial of its own choosing. The proposer
writes noise, but code the Gate does not know for the first mutation of
round 2, so one child stops before evaluation.
"""

import json
import shutil

import pytest

from factor_agent import events
from factor_agent.web.run_view import RunView, read_round
from tests.campaign_runner import candidate, run, write_config
from tests.fake_llm import FakeLLM, has_tag, new_directions, tagged
from tests.test_campaign_loop import market, noise  # noqa: F401  (the module's synthetic store)

ROUND = {"candidates": 5, "min_per_type": 1, "retries": 0, "max_rounds": 3}


def _plan(number, quota, prompt):
    directions = new_directions(number, quota, prompt)
    picked = tagged(prompt, "picked_slots") if has_tag(prompt, "picked_slots") else []
    for slot in picked:
        [i] = [i for i, d in enumerate(directions) if d["mechanism_type"] == slot["mechanism_type"]]
        directions[i] = {**directions[i], "action": "mutation", "parent": slot["parent"], "direction": "another channel"}
    if number == 3:
        free = next(i for i, d in enumerate(directions) if d["action"] == "new")
        parent = next(f"T1-{s:02d}" for s in range(1, 6) if f"T1-{s:02d}" not in {p["parent"] for p in picked})
        directions[free] = {**directions[free], "action": "refinement", "parent": parent, "direction": "the planner's own"}
    return directions


def _propose(number, directions):
    first = next((i for i, d in enumerate(directions) if d["action"] == "mutation"), None)
    return [candidate("unknown", "alpha = SecretOp(close)") if number == 2 and slot == first else noise(number, slot)
            for slot in range(len(directions))]


@pytest.fixture(scope="module")
def picked(tmp_path_factory, market):  # noqa: F811
    root = tmp_path_factory.mktemp("lineage")
    config = write_config(root, market["components"], round={**ROUND, "picked_parents": 2},
                          context={"theme_correlation": 0.5})
    run(config, FakeLLM(_propose, plan=_plan), market["cache"])
    plans = {e.round: e for e in events.read(root / "run" / events.FILE) if e.type == "plan"}
    return root / "run", RunView.read(root / "run"), plans


def _by_id(view):
    return {f"T{t.round}-{t.slot + 1:02d}": t for t in view.trials}


def test_a_trial_carries_the_rounds_it_was_a_picked_parent_in(picked):
    _, view, plans = picked
    expected = {}
    for number, plan in plans.items():
        for p in plan.picked:
            expected.setdefault(p["trial"], []).append(number)
    assert sum(map(len, expected.values())) == 4  # two in round 2, two in round 3
    assert {trial: list(t.picked_in) for trial, t in _by_id(view).items() if t.picked_in} == expected


def test_a_trial_says_whether_the_code_picked_its_parent(picked):
    _, view, plans = picked
    trials = _by_id(view)
    for number in (2, 3):
        parents = sorted(p["trial"] for p in plans[number].picked)
        assert sorted(t.direction["parent"] for t in trials.values() if t.round == number and t.parent_picked) == parents
    [own] = [t for t in trials.values() if t.direction["action"] == "refinement"]
    assert own.round == 3 and not own.parent_picked
    assert not any(t.parent_picked for t in trials.values() if t.round == 1)


def test_a_parents_lineage_counts_its_children_and_the_one_that_stopped_before_evaluation(picked):
    _, view, _ = picked
    trials = _by_id(view)
    [stopped] = [t for t in trials.values() if t.status == "failed"]
    assert stopped.round == 2 and stopped.direction["action"] == "mutation"
    parent = trials[stopped.direction["parent"]]
    children = [t for t in trials.values() if t.direction["parent"] == stopped.direction["parent"]]
    lineage = parent.lineage
    assert lineage.attempts == len(children) >= 1 and lineage.unevaluated == 1
    assert lineage.mutations == sum(c.direction["action"] == "mutation" for c in children)
    assert lineage.own_t == pytest.approx(next(p for p in view.evaluated
                                               if (p.round, p.slot) == (parent.round, parent.slot)).partial_ic_t)
    assert stopped.lineage.attempts == 0 and stopped.lineage.own_t is None and stopped.lineage.best_child is None


def test_a_round_reads_its_slots_quota_and_issues_from_its_plan(picked):
    run_dir, view, plans = picked
    plan = plans[3]
    found = read_round(run_dir, 3)
    assert found.quota == plan.quota and list(found.issues) == plan.issues and found.graph is None
    assert [(s.mechanism_type, s.action, s.parent) for s in found.slots] == [
        (d["mechanism_type"], d["action"], d["parent"]) for d in plan.directions]
    assert sorted(s.parent for s in found.slots if s.picked) == sorted(p["trial"] for p in plan.picked)
    statuses = [t.status for t in view.trials if t.round == 3]
    assert [s.outcome for s in found.slots] == statuses
    assert read_round(run_dir, 4) is None and read_round(run_dir, 0) is None


def test_the_round_in_progress_reads_as_planned_with_its_slots_pending(picked, tmp_path):
    run_dir, view, plans = picked
    copy = tmp_path / "run"
    shutil.copytree(run_dir, copy)
    # The run as it read while round 3 was proposed: its state counts two rounds, its events end at round 3's plan.
    state = json.loads((copy / "state.json").read_text())
    (copy / "state.json").write_text(json.dumps({**state, "rounds": 2}))
    lines = (copy / events.FILE).read_text().splitlines()
    end = next(i for i, line in enumerate(lines) if json.loads(line)["type"] == "plan" and json.loads(line)["round"] == 3)
    (copy / events.FILE).write_text("\n".join(lines[:end + 1]) + "\n")
    found = read_round(copy, 3)
    assert [s.outcome for s in found.slots] == [None] * 5
    assert sum(s.picked for s in found.slots) == 2
    running = RunView.read(copy)
    assert running.running.round == 3
    assert [p.parent_picked for p in running.running.pending] == [s.picked for s in found.slots]
    assert sorted(t.round for t in running.trials if 3 in t.picked_in) == sorted(
        int(p["trial"][1]) for p in plans[3].picked)


def test_without_events_a_run_still_has_its_lineage_and_its_rounds_slots_but_no_picks(picked, tmp_path):
    run_dir, view, _ = picked
    copy = tmp_path / "run"
    shutil.copytree(run_dir, copy)
    (copy / events.FILE).unlink()
    old = RunView.read(copy)
    assert [t.lineage.attempts for t in old.trials] == [t.lineage.attempts for t in view.trials]
    assert not any(t.picked_in or t.parent_picked for t in old.trials)
    found = read_round(copy, 2)
    assert (found.quota, found.issues) == ({}, ())
    assert [s.action for s in found.slots] == [t.direction["action"] for t in view.trials if t.round == 2]
    assert not any(s.picked for s in found.slots)


def test_a_run_without_picked_parents_has_no_picks(tmp_path, market):  # noqa: F811
    config = write_config(tmp_path, market["components"], round={**ROUND, "max_rounds": 2})
    run(config, FakeLLM(_propose, plan=_plan), market["cache"])
    view = RunView.read(tmp_path / "run")
    assert not any(t.picked_in or t.parent_picked for t in view.trials)
    assert all(t.lineage is not None for t in view.trials)
    assert not any(s.picked for s in read_round(tmp_path / "run", 2).slots)


def _with(run_dir, copy, fertile_children=None):
    """A copy of the run whose children clear the threshold (0.67 for any trial count), with the snapshot's count."""
    shutil.copytree(run_dir, copy)
    lines = (copy / events.FILE).read_text().splitlines()
    start = {**json.loads(lines[0]), "t_threshold": 0.0, "trial_exponent": 1e-9}
    (copy / events.FILE).write_text("\n".join([json.dumps(start), *lines[1:]]) + "\n")
    if fertile_children is not None:
        manifest = json.loads((copy / "snapshot.json").read_text())
        manifest["settings"]["context"]["fertile_children"] = fertile_children
        (copy / "snapshot.json").write_text(json.dumps(manifest))
    return _by_id(RunView.read(copy))


def test_the_fertility_averages_as_many_children_as_the_runs_snapshot_says(picked, tmp_path):
    run_dir, _, _ = picked
    three, one = _with(run_dir, tmp_path / "three"), _with(run_dir, tmp_path / "one", fertile_children=1)
    # T1-02 has one evaluated child: the mean of three pads it with two zeros, the mean of one does not.
    assert three["T1-02"].lineage.attempts == 1 and three["T1-02"].lineage.fertility > 0
    assert one["T1-02"].lineage.fertility == pytest.approx(3 * three["T1-02"].lineage.fertility)
    # T1-05 has two: the best alone is more than their mean with a zero.
    assert one["T1-05"].lineage.fertility > three["T1-05"].lineage.fertility > 0
    assert [t.lineage.attempts for t in one.values()] == [t.lineage.attempts for t in three.values()]
