"""The frontend's HTTP routes over a run a real campaign wrote, and its console script.

The JSON of the run list and of a run's accepted factors is compared with
stored snapshots (``tests/snapshots/web_runs.json``,
``tests/snapshots/web_run_factors.json``), with the values that change from
run to run or machine to machine (the code version, the time, statistics)
replaced. After a deliberate change of the
JSON's shape, rewrite the snapshot with ``FACTOR_AGENT_UPDATE_SNAPSHOTS=1``.
"""

import json
import os
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
TRIALS = Path(__file__).parent / "snapshots" / "web_run_trials.json"

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
    return TestClient(create_app(root / "runs"))


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
