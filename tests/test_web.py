"""The frontend's HTTP routes over a run a real campaign wrote, and its console script.

The JSON of the run list, of a run's accepted factors, of its Acceptance
reports, of its health, of its round in progress and of one of its LLM
calls is compared with stored snapshots (``tests/snapshots/web_runs.json``,
``tests/snapshots/web_run_factors.json``, ``tests/snapshots/web_run_acceptance.json``,
``tests/snapshots/web_run_health.json``, ``tests/snapshots/web_run_running.json``,
``tests/snapshots/web_run_call.json``), of a round's plan
(``tests/snapshots/web_run_round.json``) and of a run with Picked parents
(``tests/snapshots/web_run_picked.json``: a round's plan with its Trial
graph, and the trials' Lineage and picks), with the values that change from
run to run or machine to machine (the code version, the time, statistics)
replaced. After a deliberate change of the
JSON's shape, rewrite the snapshot with ``FACTOR_AGENT_UPDATE_SNAPSHOTS=1``.
"""

import json
import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from factor_agent import _trial_graph, events  # noqa: E402
from factor_agent.evaluator import Evaluator  # noqa: E402
from factor_agent.web.app import create_app, main  # noqa: E402
from quantlab.core.component import rebuild  # noqa: E402
from quantlab.factor.config import FactorConfig  # noqa: E402
from quantlab.factor.predefined.alpha158 import Alpha158Stock  # noqa: E402
from tests import synthetic_market  # noqa: E402
from tests.campaign_runner import candidate, crash_at, run, write_config  # noqa: E402
from tests.fake_llm import FakeLLM, Interrupted, has_tag, new_directions, tagged  # noqa: E402
from tests.test_campaign_loop import noise  # noqa: E402

SNAPSHOT = Path(__file__).parent / "snapshots" / "web_runs.json"
FACTORS = Path(__file__).parent / "snapshots" / "web_run_factors.json"
ACCEPTANCE = Path(__file__).parent / "snapshots" / "web_run_acceptance.json"
TRIALS = Path(__file__).parent / "snapshots" / "web_run_trials.json"
HEALTH = Path(__file__).parent / "snapshots" / "web_run_health.json"
CALL = Path(__file__).parent / "snapshots" / "web_run_call.json"
RUNNING = Path(__file__).parent / "snapshots" / "web_run_running.json"
FILES = Path(__file__).parent / "snapshots" / "web_run_files.json"
IC = Path(__file__).parent / "snapshots" / "web_run_ic.json"
ROUND_PLAN = Path(__file__).parent / "snapshots" / "web_run_round.json"
PICKED = Path(__file__).parent / "snapshots" / "web_run_picked.json"

#: Each client's folder of runs, synthetic market components and KunQuant cache.
CLIENT: dict = {}

#: The example run's round settings.
SETTINGS = {"candidates": 2, "min_per_type": 0, "retries": 0, "max_rounds": 1}

ROUND = [
    candidate("planted", "alpha = close / open"),
    candidate("unknown name", "alpha = SecretOp(close)"),
]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("web")
    components = synthetic_market.build(root / "data")
    klen = Alpha158Stock(FactorConfig(
        warmup_bars=60, dataset=rebuild(components["prices"]), mode="batch",
        data_columns=("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"), factor_names=("KLEN",),
    ))
    config = write_config(root / "runs" / "example", components, known=[klen.get_config()],
                          round=SETTINGS)
    run(config, FakeLLM(lambda number, directions: ROUND), root / "kunquant_cache")
    client = TestClient(create_app(root / "runs"))
    CLIENT[client] = {"runs": root / "runs", "components": components, "cache": root / "kunquant_cache"}
    return client


def _stable(runs: list[dict]) -> list[dict]:
    return [{**r, "code_version": "<code version>", "updated": "<time>"} for r in runs]


def test_the_run_list_route_answers_with_the_snapshot_json(client):
    response = client.get("/api/runs")
    assert response.status_code == 200
    runs = _stable(response.json())
    if os.environ.get("FACTOR_AGENT_UPDATE_SNAPSHOTS"):
        SNAPSHOT.parent.mkdir(exist_ok=True)
        SNAPSHOT.write_text(json.dumps(runs, indent=2) + "\n")
    assert runs == json.loads(SNAPSHOT.read_text())


def test_a_run_is_read_by_its_name_and_an_unknown_name_is_not_found(client):
    run_json = client.get("/api/run", params={"name": "example/run"}).json()
    assert run_json["summary"]["name"] == "example/run"
    assert [t["status"] for t in run_json["trials"]] == ["accepted", "failed"]
    assert client.get("/api/run", params={"name": "elsewhere"}).status_code == 404
    assert client.get("/api/run", params={"name": "../.."}).status_code == 404


def _shape(value):
    """``value`` with every number that is not a count replaced, as the contract does not fix them."""
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_shape(item) for item in value]
    return "<number>" if isinstance(value, float) else value


def test_a_runs_accepted_factors_answer_with_the_snapshot_json(client):
    run_json = client.get("/api/run", params={"name": "example/run"}).json()
    factors = _shape({key: run_json[key] for key in ("factors", "correlation_limit")})
    if os.environ.get("FACTOR_AGENT_UPDATE_SNAPSHOTS"):
        FACTORS.write_text(json.dumps(factors, indent=2) + "\n")
    assert factors == json.loads(FACTORS.read_text())


def test_a_runs_health_answers_with_the_snapshot_json(client):
    health = client.get("/api/run", params={"name": "example/run"}).json()["health"]
    assert health["last_call"] is not None
    health = _shape({**health, "last_call": "<time>"})
    if os.environ.get("FACTOR_AGENT_UPDATE_SNAPSHOTS"):
        HEALTH.write_text(json.dumps(health, indent=2) + "\n")
    assert health == json.loads(HEALTH.read_text())


def test_a_runs_version_changes_when_the_run_writes_and_not_otherwise(client, tmp_path):
    def version():
        response = client.get("/api/run/version", params={"name": "example/run"})
        assert response.status_code == 200
        return response.json()

    before = version()
    assert version() == before
    state = CLIENT[client]["runs"] / "example" / "run" / "state.json"
    os.utime(state, ns=(state.stat().st_atime_ns, state.stat().st_mtime_ns + 1_000_000))
    assert version() != before
    assert client.get("/api/run/version", params={"name": "elsewhere"}).status_code == 404


def _judged(source: Path, root: Path) -> TestClient:
    """A client over a copy of ``source`` holding a validation report with a contribution table and a forced hold-out."""
    from factor_agent.acceptance import AcceptanceReport, Contribution, ModelResult

    def results(rank_ic: float) -> tuple:
        return tuple(ModelResult(seed=seed, rank_ic=rank_ic + seed / 100, ic=0.01, monotonicity=0.5, long_short=0.001,
                                 top_k_return=0.002, top_k_excess=0.001, top_k_annualised=0.1,
                                 decile_returns=(0.0,) * 10, n_days=200, run="") for seed in (0, 1))

    planted = "planted (accepted in round 1)"
    common = {"train_window": ("2010-01-04", "2011-12-20"), "seeds": (0, 1), "top_k": 20,
              "accepted": {planted: "accepted_0"}, "code_version": {"factor_agent": "abc"}}
    validation = AcceptanceReport(segment="validation", test_window=("2012-01-03", "2012-12-20"),
                                  baseline=results(0.02), with_accepted=results(0.03), contributions=(
                                      Contribution(planted, "drop_one", without=results(0.02), with_factor=results(0.03)),
                                      Contribution(planted, "add_one", without=results(0.02), with_factor=results(0.03))),
                                  **common)
    opens = ({"opened_at": "2026-10-01T00:00:00+00:00", "code_version": {"factor_agent": "abc"}, "forced": False},
             {"opened_at": "2026-10-02T00:00:00+00:00", "code_version": {"factor_agent": "abc"}, "forced": True})
    holdout = AcceptanceReport(segment="holdout", test_window=("2013-01-03", "2013-12-20"), baseline=results(0.02),
                               with_accepted=results(0.01), forced=True, holdout_opens=opens, **common)
    run_dir = root / "example" / "run"
    shutil.copytree(source, run_dir)
    (run_dir / "acceptance").mkdir()
    for minute, report in enumerate((validation, holdout)):
        path = run_dir / "acceptance" / f"{report.segment}.json"
        path.write_text(json.dumps(report.to_dict()))
        # The reports are listed in the order written; a coarse file clock gives two quick writes one mtime.
        os.utime(path, (1_800_000_000 + 60 * minute,) * 2)
    (run_dir / "holdout.lock").write_text(json.dumps({"segment": {}, "opens": opens}))
    return TestClient(create_app(root))


def test_a_runs_acceptance_reports_answer_with_the_snapshot_json(client, tmp_path):
    judged = _judged(CLIENT[client]["runs"] / "example" / "run", tmp_path)
    acceptance = judged.get("/api/run", params={"name": "example/run"}).json()["acceptance"]
    for report in (*acceptance["reports"], acceptance["validation"], acceptance["holdout"]):
        report["written"] = "<time>"
    acceptance = _shape(acceptance)
    if os.environ.get("FACTOR_AGENT_UPDATE_SNAPSHOTS"):
        ACCEPTANCE.write_text(json.dumps(acceptance, indent=2) + "\n")
    assert acceptance == json.loads(ACCEPTANCE.read_text())


def test_a_run_whose_distiller_has_written_nothing_answers_with_no_memory_versions(client):
    assert client.get("/api/run", params={"name": "example/run"}).json()["memory"] == []


def test_a_runs_trials_page_answers_with_the_snapshot_json(client):
    run_json = client.get("/api/run", params={"name": "example/run"}).json()
    trials = _shape({key: run_json[key] for key in ("trials", "evaluated", "threshold", "never_evaluated")})
    if os.environ.get("FACTOR_AGENT_UPDATE_SNAPSHOTS"):
        TRIALS.write_text(json.dumps(trials, indent=2) + "\n")
    assert trials == json.loads(TRIALS.read_text())


def test_the_page_is_served_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200 and '<div id="root">' in response.text


def test_the_console_script_has_no_option_to_bind_another_interface(capsys):
    with pytest.raises(SystemExit):
        main(["runs", "--host", "0.0.0.0"])
    assert "unrecognized arguments: --host" in capsys.readouterr().err


def _times(value):
    """``value`` with every time replaced, as the contract does not fix them."""
    if isinstance(value, dict):
        return {key: "<time>" if key in ("time", "sent", "returned", "started", "evaluation", "opened_at") and item else _times(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_times(item) for item in value]
    return value


def test_an_llm_call_is_read_by_its_number_and_answers_with_the_snapshot_json(client):
    call = client.get("/api/run/call", params={"name": "example/run", "number": 2}).json()
    assert call["role"] == "proposer" and "<directions>" in call["prompt"]
    call = _shape(_times({**call, "prompt": "<prompt>", "reply": "<reply>", "model": "<model>"}))
    if os.environ.get("FACTOR_AGENT_UPDATE_SNAPSHOTS"):
        CALL.write_text(json.dumps(call, indent=2) + "\n")
    assert call == json.loads(CALL.read_text())
    assert client.get("/api/run/call", params={"name": "example/run", "number": 99}).status_code == 404
    assert client.get("/api/run/call", params={"name": "elsewhere", "number": 1}).status_code == 404


def test_a_runs_config_and_snapshot_files_answer_with_the_snapshot_json(client):
    files = client.get("/api/run/files", params={"name": "example/run"}).json()
    assert files["configs"][0]["opening"] == 1 and "run_dir:" in files["configs"][0]["text"]
    files = _times({
        "configs": [{**c, "text": "<config>"} for c in files["configs"]],
        "snapshot": [{**f, "text": "<file>"} for f in files["snapshot"]],
    })
    if os.environ.get("FACTOR_AGENT_UPDATE_SNAPSHOTS"):
        FILES.write_text(json.dumps(files, indent=2) + "\n")
    assert files == json.loads(FILES.read_text())
    assert client.get("/api/run/files", params={"name": "elsewhere"}).status_code == 404


def test_a_trials_daily_ic_answers_with_the_snapshot_json(client):
    evaluated = next(t for t in client.get("/api/run", params={"name": "example/run"}).json()["trials"] if t["metrics"])
    params = {"name": "example/run", "round": evaluated["round"], "slot": evaluated["slot"]}
    series = client.get("/api/run/ic", params=params).json()
    assert len(series["days"]) == len(series["ic"]) == len(series["partial_ic"]) > 2
    series = _shape({key: values[:2] for key, values in series.items()})
    if os.environ.get("FACTOR_AGENT_UPDATE_SNAPSHOTS"):
        IC.write_text(json.dumps(series, indent=2) + "\n")
    assert series == json.loads(IC.read_text())
    assert client.get("/api/run/ic", params={**params, "round": 99}).status_code == 404


def test_the_round_in_progress_answers_with_the_snapshot_json(client, tmp_path):
    # The example run as it read while its round was evaluated: it crashed in the Evaluator.
    config = write_config(tmp_path / "example", CLIENT[client]["components"], round=SETTINGS)
    with crash_at(Evaluator, "evaluate"), pytest.raises(Interrupted):
        run(config, FakeLLM(lambda number, directions: ROUND), CLIENT[client]["cache"])
    running = TestClient(create_app(tmp_path)).get("/api/run", params={"name": "example/run"}).json()["running"]
    assert [p["stage"] for p in running["pending"]] == ["consistent", "proposed"]
    running = _shape(_times({**running, "calls": [{**c, "model": "<model>"} for c in running["calls"]]}))
    if os.environ.get("FACTOR_AGENT_UPDATE_SNAPSHOTS"):
        RUNNING.write_text(json.dumps(running, indent=2) + "\n")
    assert running == json.loads(RUNNING.read_text())
    assert client.get("/api/run", params={"name": "example/run"}).json()["running"] is None


def test_a_run_answers_with_its_diagnosis_and_each_trials_collisions(client):
    run_json = client.get("/api/run", params={"name": "example/run"}).json()
    assert run_json["diagnosis"] == []
    assert [t["collisions"] for t in run_json["trials"]] == [[], []]


def test_a_rounds_plan_answers_with_the_snapshot_json(client):
    plan = client.get("/api/run/round", params={"name": "example/run", "round": 1}).json()
    assert [s["trial"] for s in plan["slots"]] == ["T1-01", "T1-02"]
    if os.environ.get("FACTOR_AGENT_UPDATE_SNAPSHOTS"):
        ROUND_PLAN.write_text(json.dumps(plan, indent=2) + "\n")
    assert plan == json.loads(ROUND_PLAN.read_text())
    assert client.get("/api/run/round", params={"name": "example/run", "round": 2}).status_code == 404
    assert client.get("/api/run/round", params={"name": "elsewhere", "round": 1}).status_code == 404


def _follow_picks(number, quota, prompt):
    """The planner: a mutation of each picked slot's parent in its type, new directions elsewhere."""
    directions = new_directions(number, quota, prompt)
    for slot in tagged(prompt, "picked_slots") if has_tag(prompt, "picked_slots") else []:
        i = next(i for i, d in enumerate(directions) if d["mechanism_type"] == slot["mechanism_type"] and not d["parent"])
        directions[i] = {**directions[i], "action": "mutation", "parent": slot["parent"], "direction": "another channel"}
    return directions


def test_a_run_with_picked_parents_answers_with_the_snapshot_json(client, tmp_path):
    config = write_config(tmp_path / "picked", CLIENT[client]["components"], context={"theme_correlation": 0.5},
                          round={"candidates": 5, "min_per_type": 1, "retries": 0, "max_rounds": 2, "picked_parents": 2})
    run(config, FakeLLM(lambda number, directions: [noise(number, slot) for slot in range(len(directions))],
                        plan=_follow_picks), CLIENT[client]["cache"])
    picked = TestClient(create_app(tmp_path))
    trials = picked.get("/api/run", params={"name": "picked/run"}).json()["trials"]
    plan = picked.get("/api/run/round", params={"name": "picked/run", "round": 2}).json()
    assert sum(s["picked"] for s in plan["slots"]) == 2
    assert [s["trial"] for s in plan["slots"]] == [f"T2-{slot:02d}" for slot in range(1, 6)]
    found = _shape({"round": {**plan, "slots": [{**s, "outcome": "<outcome>"} for s in plan["slots"]]},
                    "trials": [{key: t[key] for key in ("round", "slot", "lineage", "picked_in", "parent_picked")}
                               for t in trials]})
    if os.environ.get("FACTOR_AGENT_UPDATE_SNAPSHOTS"):
        PICKED.write_text(json.dumps(found, indent=2) + "\n")
    assert found == json.loads(PICKED.read_text())


def test_a_rounds_trial_graph_ranks_as_the_graph_file_does_and_is_absent_without_the_file(client, tmp_path):
    config = write_config(tmp_path / "picked", CLIENT[client]["components"], context={"theme_correlation": 0.5},
                          round={"candidates": 5, "min_per_type": 1, "retries": 0, "max_rounds": 2, "picked_parents": 2})
    run(config, FakeLLM(lambda number, directions: [noise(number, slot) for slot in range(len(directions))],
                        plan=_follow_picks), CLIENT[client]["cache"])
    picked = TestClient(create_app(tmp_path))
    params = {"name": "picked/run", "round": 2}
    found = picked.get("/api/run/round", params=params).json()["graph"]
    graph = _trial_graph.read(tmp_path / "picked" / "run", 2)
    assert found["gamma"] == graph.gamma
    plan = next(e for e in events.read(tmp_path / "picked" / "run" / events.FILE) if e.type == "plan" and e.round == 2)
    assert found["picked"] == [p["trial"] for p in plan.picked]
    nodes = {n["trial"]: n for n in found["nodes"]}
    for propagated in (True, False):
        ranked = sorted((n for n in found["nodes"] if n["rank_propagated" if propagated else "rank_own"] is not None),
                        key=lambda n: n["rank_propagated" if propagated else "rank_own"])
        assert [n["trial"] for n in ranked] == [p.trial for p in graph.pick_order(propagated)]
    for n in found["nodes"]:
        for u in ("quality", "fertility", "novelty"):
            assert n["propagated"][u] == pytest.approx((1 - found["gamma"]) * n["own"][u] + found["gamma"] * n["neighbourhood"][u])
        assert [(h["trial"], h["weight"], tuple(h["relations"])) for h in n["heaviest"]] == [
            (h.trial, h.weight, h.relations) for h in graph.heaviest(n["trial"], 5)]
        assert len({h["trial"] for h in n["heaviest"]}) == len(n["heaviest"])
    assert set(nodes) == {n.trial for n in graph.nodes}
    (tmp_path / "picked" / "run" / "trial_graph" / "round_02.json").unlink()
    assert picked.get("/api/run/round", params=params).json()["graph"] is None
