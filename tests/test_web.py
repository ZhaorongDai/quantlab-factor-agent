"""The frontend's HTTP routes over a run a real campaign wrote, and its console script.

The JSON of the run list, of a run's accepted factors, of its Acceptance
reports and of its health is compared with stored snapshots
(``tests/snapshots/web_runs.json``, ``tests/snapshots/web_run_factors.json``,
``tests/snapshots/web_run_acceptance.json``, ``tests/snapshots/web_run_health.json``), with the values that change from
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

from factor_agent.web.app import create_app, main  # noqa: E402
from quantlab.core.component import rebuild  # noqa: E402
from quantlab.factor.config import FactorConfig  # noqa: E402
from quantlab.factor.predefined.alpha158 import Alpha158Stock  # noqa: E402
from tests import synthetic_market  # noqa: E402
from tests.campaign_runner import candidate, run, write_config  # noqa: E402
from tests.fake_llm import FakeLLM  # noqa: E402

SNAPSHOT = Path(__file__).parent / "snapshots" / "web_runs.json"
FACTORS = Path(__file__).parent / "snapshots" / "web_run_factors.json"
ACCEPTANCE = Path(__file__).parent / "snapshots" / "web_run_acceptance.json"
TRIALS = Path(__file__).parent / "snapshots" / "web_run_trials.json"
HEALTH = Path(__file__).parent / "snapshots" / "web_run_health.json"

#: Each client's folder of runs.
RUNS: dict = {}

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
                          round={"candidates": 2, "min_per_type": 0, "retries": 0, "max_rounds": 1})
    run(config, FakeLLM(lambda number, directions: ROUND), root / "kunquant_cache")
    client = TestClient(create_app(root / "runs"))
    RUNS[client] = root / "runs"
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
    state = RUNS[client] / "example" / "run" / "state.json"
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
    for report in (validation, holdout):
        (run_dir / "acceptance" / f"{report.segment}.json").write_text(json.dumps(report.to_dict()))
    (run_dir / "holdout.lock").write_text(json.dumps({"segment": {}, "opens": opens}))
    return TestClient(create_app(root))


def test_a_runs_acceptance_reports_answer_with_the_snapshot_json(client, tmp_path):
    judged = _judged(RUNS[client] / "example" / "run", tmp_path)
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
    trials = _shape({key: run_json[key] for key in ("evaluated", "threshold", "never_evaluated")})
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


def test_a_run_answers_with_its_diagnosis_and_each_trials_collisions(client):
    run_json = client.get("/api/run", params={"name": "example/run"}).json()
    assert run_json["diagnosis"] == []
    assert [t["collisions"] for t in run_json["trials"]] == [[], []]
