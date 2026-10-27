"""The frontend's HTTP routes over a run a real campaign wrote, and its console script.

The JSON of the run list is compared with a stored snapshot
(``tests/snapshots/web_runs.json``), with the values that change from run to
run (the code version, the time) replaced. After a deliberate change of the
JSON's shape, rewrite the snapshot with ``FACTOR_AGENT_UPDATE_SNAPSHOTS=1``.
"""

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from factor_agent.web.app import create_app, main  # noqa: E402
from tests import synthetic_market  # noqa: E402
from tests.campaign_runner import candidate, run, write_config  # noqa: E402
from tests.fake_llm import FakeLLM  # noqa: E402

SNAPSHOT = Path(__file__).parent / "snapshots" / "web_runs.json"

ROUND = [
    candidate("planted", "alpha = close / open"),
    candidate("unknown name", "alpha = SecretOp(close)"),
]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("web")
    components = synthetic_market.build(root / "data")
    config = write_config(root / "runs" / "example", components,
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


def test_a_run_whose_distiller_has_written_nothing_answers_with_no_memory_versions(client):
    assert client.get("/api/run", params={"name": "example/run"}).json()["memory"] == []


def test_the_page_is_served_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200 and '<div id="root">' in response.text


def test_the_console_script_has_no_option_to_bind_another_interface(capsys):
    with pytest.raises(SystemExit):
        main(["runs", "--host", "0.0.0.0"])
    assert "unrecognized arguments: --host" in capsys.readouterr().err
