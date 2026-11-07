"""Campaign.check(): what a run would refuse before its first LLM call, found without writing to the run directory.

The configs are the synthetic market's, each campaign reading its own
copies of the repository's field and operator files so that a test can
change them; the roles' scripted backends read a replies file that exists.
A resumed run is first run for one round with ``tests/fake_llm.FakeLLM``.
"""

import http.server
import json
import threading
from pathlib import Path

import pytest
import yaml

from factor_agent.campaign import Campaign
from factor_agent.cli import main
from factor_agent.config import ROLES, CampaignConfig
from tests import synthetic_market
from tests.campaign_runner import candidate, run, write_config
from tests.fake_llm import FakeLLM
from tests.vocabulary_files import EXAMPLE_OPERATORS, PRICE_VOLUME

SETTINGS = {"candidates": 5, "min_per_type": 1, "retries": 0, "max_rounds": 1}


@pytest.fixture(scope="module")
def market(tmp_path_factory):
    root = tmp_path_factory.mktemp("market")
    return {"components": synthetic_market.build(root / "data"), "cache": root / "kunquant_cache"}


def _config(root: Path, market, **sections) -> Path:
    """A good config under ``root``: its own vocabulary files and scripted roles whose replies file exists."""
    (root / "vocabulary").mkdir(parents=True, exist_ok=True)
    fields, operators = root / "vocabulary" / PRICE_VOLUME.name, root / "vocabulary" / EXAMPLE_OPERATORS.name
    fields.write_text(PRICE_VOLUME.read_text())
    operators.write_text(EXAMPLE_OPERATORS.read_text())
    replies = root / "replies.json"
    replies.write_text("[]")
    llm = {role: {"backend": "scripted", "model": f"model-{role}", "replies": str(replies)} for role in ROLES}
    sections = {"llm": llm, "round": SETTINGS, "fields": [str(fields)], "operators": str(operators), **sections}
    return write_config(root, market["components"], **sections)


def _check(path: Path):
    return Campaign(CampaignConfig.from_file(path)).check()


def test_a_good_config_has_no_problem_and_the_text_lists_what_was_checked(tmp_path, market):
    report = _check(_config(tmp_path, market))
    assert report.problems == []
    assert report.skipped == []
    text = report.text()
    for checked in ("data.prices", "data.label", "data.membership", "scoring", *(f"llm.{r}" for r in ROLES),
                    "reference and known factors", "run snapshot", "vocabulary", "templates"):
        assert checked in text
    assert "would be refused" not in text


def test_a_missing_price_store_is_a_problem_and_the_vocabulary_and_templates_are_skipped(tmp_path, market):
    missing = tmp_path / "nowhere" / "prices.zarr"
    path = _config(tmp_path, market)
    config = yaml.safe_load(path.read_text())
    config["data"]["prices"]["zarr_file_path"] = str(missing)
    path.write_text(yaml.safe_dump(config))
    report = _check(path)
    [problem] = report.problems
    assert problem.startswith("data.prices: ") and str(missing) in problem
    assert dict(report.skipped) == {
        "vocabulary": "it needs data.prices, which failed",
        "templates": "they need vocabulary, which failed or did not run",
    }
    text = report.text()
    assert "skipped  vocabulary: it needs data.prices, which failed" in text and str(missing) in text
    assert text.endswith("the run would be refused")


def test_an_unset_key_variable_is_a_problem_of_its_role(tmp_path, market, monkeypatch):
    monkeypatch.delenv("FACTOR_AGENT_TEST_KEY", raising=False)
    path = _config(tmp_path, market)
    config = yaml.safe_load(path.read_text())
    config["llm"]["reviewer"] = {
        "backend": "openai_compatible", "model": "a-model", "base_url": "http://localhost:1/v1",
        "api_key_env": "FACTOR_AGENT_TEST_KEY",
    }
    path.write_text(yaml.safe_dump(config))
    report = _check(path)
    [problem] = report.problems
    assert problem.startswith("llm.reviewer: ") and "FACTOR_AGENT_TEST_KEY" in problem
    assert report.skipped == []


def test_several_problems_are_reported_together(tmp_path, market, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    path = _config(tmp_path, market)
    config = yaml.safe_load(path.read_text())
    config["data"]["label"]["label"]["dataset"]["zarr_file_path"] = str(tmp_path / "nowhere" / "label.zarr")
    config["llm"]["planner"] = {"backend": "anthropic", "model": "a-model"}
    config["reference"] = [{"store": str(tmp_path / "nowhere" / "factors")}]
    path.write_text(yaml.safe_dump(config))
    report = _check(path)
    assert [problem.split(":")[0] for problem in report.problems] == [
        "data.label", "llm.planner", "reference and known factors",
    ]
    assert dict(report.skipped) == {"templates": "they need data.label, which failed or did not run"}


def _propose(number, directions):
    planted = [candidate("intraday move", "alpha = close / open")]
    noise = [
        candidate(f"noise {slot}", f"alpha = BackRef(volume, {(5, 10, 20, 40, 60)[slot]}) / WindowedAvg(volume, 20)")
        for slot in range(len(directions))
    ]
    return (planted + noise)[:len(directions)]


@pytest.fixture
def resumed(tmp_path, market) -> Path:
    """The config of a run that ran its one round and is resumed with a raised round limit."""
    path = _config(tmp_path, market)
    run(path, FakeLLM(_propose), market["cache"])
    config = yaml.safe_load(path.read_text())
    config["round"]["max_rounds"] = 2
    path.write_text(yaml.safe_dump(config))
    return path


def _contents(folder: Path) -> dict[str, bytes]:
    return {str(file.relative_to(folder)): file.read_bytes() for file in sorted(folder.rglob("*")) if file.is_file()}


def test_a_check_leaves_a_fresh_run_directory_unmade(tmp_path, market):
    report = _check(_config(tmp_path, market))
    assert report.ok
    assert not (tmp_path / "run").exists()


def test_a_check_of_a_resume_leaves_the_run_directory_byte_identical(resumed, tmp_path):
    before = _contents(tmp_path / "run")
    report = _check(resumed)
    assert report.problems == [] and report.warnings == []
    assert _contents(tmp_path / "run") == before


def test_a_resume_with_a_changed_field_file_is_a_problem_of_the_run_snapshot(resumed, tmp_path):
    fields = tmp_path / "vocabulary" / PRICE_VOLUME.name
    fields.write_text(fields.read_text() + "\n# an edit after the run started\n")
    before = _contents(tmp_path / "run")
    [problem] = _check(resumed).problems
    assert problem.startswith("run snapshot: ") and f"fields/{fields.name} changed" in problem
    assert _contents(tmp_path / "run") == before


def test_a_resume_on_other_code_is_a_warning_not_a_problem(resumed, tmp_path):
    manifest = tmp_path / "run" / "snapshot.json"
    recorded = json.loads(manifest.read_text())
    recorded["versions"][-1]["factor_agent"] = "0123abc"
    manifest.write_text(json.dumps(recorded))
    report = _check(resumed)
    assert report.ok
    [warning] = report.warnings
    assert warning.startswith("run snapshot: ") and "factor_agent 0123abc ->" in warning
    assert "warning  run snapshot" in report.text()


def test_a_missing_exposure_store_is_a_problem_of_the_scoring(tmp_path, market):
    missing = tmp_path / "nowhere" / "exposures.zarr"
    path = _config(tmp_path, market)
    config = yaml.safe_load(path.read_text())
    config["scoring"]["neutralize"]["exposures"]["zarr_file_path"] = str(missing)
    path.write_text(yaml.safe_dump(config))
    report = _check(path)
    assert report.problems == [f"scoring: the store {missing} it reads does not exist"]
    assert dict(report.skipped) == {"templates": "they need scoring, which failed or did not run"}


class _Provider:
    """A local chat-completions server: it answers the model ``known`` and refuses any other with HTTP 404."""

    def __init__(self, known: str):
        self.models: list[str] = []
        provider = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                model = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["model"]
                provider.models.append(model)
                status, body = (200, {
                    "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": model,
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "OK"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
                }) if model == known else (404, {"error": {"message": f"The model {model} does not exist"}})
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"


@pytest.fixture
def provider():
    server = _Provider(known="a-model")
    yield server
    server.server.shutdown()


def _on_provider(path: Path, provider: _Provider, models: dict[str, str]) -> Path:
    """``path``'s config with each role of ``models`` on ``provider`` with that model."""
    config = yaml.safe_load(path.read_text())
    for role, model in models.items():
        config["llm"][role] = {"backend": "openai_compatible", "model": model, "base_url": provider.url}
    path.write_text(yaml.safe_dump(config))
    return path


def test_a_ping_sends_one_request_per_distinct_backend_and_reports_a_failing_one(tmp_path, market, provider):
    path = _on_provider(_config(tmp_path, market), provider, {
        "planner": "a-model", "proposer": "a-model", "reviewer": "a-wrong-model",
    })
    report = Campaign(CampaignConfig.from_file(path)).check(ping=True)
    assert sorted(provider.models) == ["a-model", "a-wrong-model"]
    [problem] = report.problems
    assert problem.startswith("ping reviewer: ") and "a-wrong-model does not exist" in problem
    assert dict(report.skipped) == {"ping distiller": "the scripted backend is not pinged"}
    text = report.text()
    assert "ok       ping planner, proposer: answered in " in text
    assert not (tmp_path / "run").exists()


def test_without_ping_no_request_is_sent(tmp_path, market, provider):
    path = _on_provider(_config(tmp_path, market), provider, {"planner": "a-wrong-model"})
    report = Campaign(CampaignConfig.from_file(path)).check()
    assert report.ok and provider.models == []
    assert not any(check.name.startswith("ping") for check in report.checks)


def test_a_role_whose_backend_cannot_be_built_is_not_pinged(tmp_path, market, monkeypatch):
    monkeypatch.delenv("FACTOR_AGENT_TEST_KEY", raising=False)
    path = _config(tmp_path, market)
    config = yaml.safe_load(path.read_text())
    config["llm"]["reviewer"] = {
        "backend": "openai_compatible", "model": "a-model", "base_url": "http://localhost:1/v1",
        "api_key_env": "FACTOR_AGENT_TEST_KEY",
    }
    path.write_text(yaml.safe_dump(config))
    report = Campaign(CampaignConfig.from_file(path)).check(ping=True)
    [problem] = report.problems
    assert problem.startswith("llm.reviewer: ")
    assert dict(report.skipped)["ping reviewer"] == "it needs llm.reviewer, which failed"


def test_the_command_exits_0_on_a_good_config_and_prints_the_report(tmp_path, market, capsys):
    path = _config(tmp_path, market)
    assert main(["check", str(path)]) == 0
    out = capsys.readouterr().out
    assert "ok       vocabulary" in out and "0 problem(s)" in out


def test_the_command_exits_1_on_a_broken_config(tmp_path, market, capsys):
    path = _config(tmp_path, market)
    config = yaml.safe_load(path.read_text())
    config["data"]["prices"]["zarr_file_path"] = str(tmp_path / "nowhere" / "prices.zarr")
    path.write_text(yaml.safe_dump(config))
    assert main(["check", str(path)]) == 1
    assert capsys.readouterr().out.rstrip().endswith("the run would be refused")


def test_the_command_pings_with_ping(tmp_path, market, provider, capsys):
    path = _on_provider(_config(tmp_path, market), provider, {role: "a-wrong-model" for role in ROLES})
    assert main(["check", str(path), "--ping"]) == 1
    assert provider.models == ["a-wrong-model"]
    assert "ping planner, proposer, reviewer, distiller" in capsys.readouterr().out


@pytest.mark.parametrize("source, says", [
    ("run_dir: [unclosed\n", "while parsing"),
    ("run_dir: run\nbudget: {}\n", "unknown section(s) ['budget']"),
])
def test_a_config_that_cannot_be_read_is_the_first_problem_not_a_traceback(tmp_path, capsys, source, says):
    path = tmp_path / "campaign.yaml"
    path.write_text(source)
    assert main(["check", str(path)]) == 1
    out = capsys.readouterr().out
    assert out.startswith("problem  config\n") and says in out and "Traceback" not in out


def test_a_missing_config_file_is_a_problem(tmp_path, capsys):
    assert main(["check", str(tmp_path / "nowhere.yaml")]) == 1
    assert "problem  config" in capsys.readouterr().out


@pytest.mark.parametrize("damage", [
    lambda text: text[: len(text) // 2],  # a write cut short
    lambda text: json.dumps({key: value for key, value in json.loads(text).items() if key != "versions"}),
])
def test_a_damaged_snapshot_is_a_problem_of_the_run_snapshot_not_a_traceback(resumed, tmp_path, damage):
    manifest = tmp_path / "run" / "snapshot.json"
    manifest.write_text(damage(manifest.read_text()))
    report = _check(resumed)
    [problem] = report.problems
    assert problem.startswith(f"run snapshot: {manifest} cannot be read (") and problem.endswith("the run cannot resume")


def test_an_invalid_trial_graph_value_is_a_problem_of_the_config(tmp_path, market, capsys):
    path = _config(tmp_path, market, trial_graph={"gamma": 1.0})
    assert main(["check", str(path)]) == 1
    out = capsys.readouterr().out
    assert out.startswith("problem  config\n") and "trial_graph: gamma is 1.0" in out
