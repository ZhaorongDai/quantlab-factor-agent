"""The RunSnapshot through Campaign.run(): what a run records it searched with, and what a resume may change.

Each campaign reads its own copies of the repository's price-volume field
file and example operator file, so a test can change them after the run.
Every role is answered by ``tests/fake_llm.FakeLLM``; the Gate and the
Evaluator are the real ones on the synthetic store.
"""

import json
from pathlib import Path

import pytest
import yaml

from factor_agent.config import ROLES
from factor_agent.snapshot import CodeVersionChanged, SnapshotMismatch
from tests import synthetic_market
from tests.campaign_runner import candidate, run, write_config
from tests.fake_llm import FakeLLM
from tests.vocabulary_files import EXAMPLE_OPERATORS, PRICE_VOLUME

SETTINGS = {"candidates": 5, "min_per_type": 1, "retries": 0, "max_rounds": 1}


def _propose(number, directions):
    planted = [candidate("intraday move", "alpha = close / open")]
    noise = [
        candidate(f"noise {slot}", f"alpha = BackRef(volume, {(5, 10, 20, 40, 60)[slot]}) / WindowedAvg(volume, 20)")
        for slot in range(len(directions))
    ]
    return (planted + noise)[:len(directions)]


@pytest.fixture(scope="module")
def market(tmp_path_factory):
    root = tmp_path_factory.mktemp("market")
    return {"components": synthetic_market.build(root / "data"), "cache": root / "kunquant_cache"}


def _files(root: Path) -> dict:
    """Copies of the repository's field and operator files under ``root``, as the config's sections."""
    (root / "vocabulary").mkdir(parents=True, exist_ok=True)
    fields, operators = root / "vocabulary" / PRICE_VOLUME.name, root / "vocabulary" / EXAMPLE_OPERATORS.name
    fields.write_text(PRICE_VOLUME.read_text())
    operators.write_text(EXAMPLE_OPERATORS.read_text())
    return {"fields": [str(fields)], "operators": str(operators)}


def _campaign(root: Path, market, **sections) -> tuple[Path, dict]:
    files = _files(root)
    path = write_config(root, market["components"], round=SETTINGS, **files, **sections)
    run(path, FakeLLM(_propose), market["cache"])
    return path, files


def test_a_first_start_copies_every_file_and_records_the_code_versions(tmp_path, market):
    path, files = _campaign(tmp_path, market)
    run_dir = tmp_path / "run"
    fields, operators = Path(files["fields"][0]), Path(files["operators"])
    assert (run_dir / "snapshot" / "fields" / fields.name).read_text() == fields.read_text()
    assert (run_dir / "snapshot" / "operators" / operators.name).read_text() == operators.read_text()
    state = json.loads((run_dir / "state.json").read_text())
    [versions] = state["versions"]
    assert versions["factor_agent"] and versions["quantlab"]


@pytest.mark.parametrize("changed", ["fields", "operators"])
def test_a_resume_after_a_field_or_operator_file_changed_fails_naming_the_file(tmp_path, market, changed):
    path, files = _campaign(tmp_path, market)
    file = Path(files[changed][0] if changed == "fields" else files[changed])
    file.write_text(file.read_text() + "\n# an edit after the run started\n")
    fake = FakeLLM(_propose)
    with pytest.raises(SnapshotMismatch, match=rf"{changed}/{file.name}.*new run directory"):
        run(path, fake, market["cache"])
    assert fake.requests == []


def test_a_resume_with_another_field_file_fails_naming_it(tmp_path, market):
    path, files = _campaign(tmp_path, market)
    extra = tmp_path / "vocabulary" / "extra.yaml"
    extra.write_text(Path(files["fields"][0]).read_text())
    config = yaml.safe_load(path.read_text())
    config["fields"].append(str(extra))
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(SnapshotMismatch, match="fields/extra.yaml"):
        run(path, FakeLLM(_propose), market["cache"])


def test_a_resume_after_a_commit_change_warns_continues_and_lists_every_commit(tmp_path, market):
    # The run is started on an older commit of both checkouts: its state says so.
    path, _ = _campaign(tmp_path, market)
    config = yaml.safe_load(path.read_text())
    config["round"]["max_rounds"] = 2
    path.write_text(yaml.safe_dump(config))
    state_path = tmp_path / "run" / "state.json"
    manifest = tmp_path / "run" / "snapshot.json"
    recorded = json.loads(manifest.read_text())
    current = dict(recorded["versions"][0])
    recorded["versions"][0] = {**current, "factor_agent": "0123abc", "quantlab": "4567def"}
    manifest.write_text(json.dumps(recorded))

    with pytest.warns(CodeVersionChanged, match=r"factor_agent 0123abc -> .*quantlab 4567def ->"):
        result = run(path, FakeLLM(_propose), market["cache"])
    assert result.rounds == 2
    versions = json.loads(state_path.read_text())["versions"]
    assert [v["factor_agent"] for v in versions] == ["0123abc", current["factor_agent"]]
    assert [v["quantlab"] for v in versions] == ["4567def", current["quantlab"]]


def test_a_resume_on_the_same_files_and_commits_neither_fails_nor_warns(tmp_path, market, recwarn):
    path, _ = _campaign(tmp_path, market)
    result = run(path, FakeLLM(_propose), market["cache"])
    assert result.rounds == 1
    assert not [w for w in recwarn if issubclass(w.category, CodeVersionChanged)]
    assert len(json.loads((tmp_path / "run" / "state.json").read_text())["versions"]) == 2


def test_a_replay_reproduces_the_ledger_after_the_repository_files_changed(tmp_path, market):
    recorded = tmp_path / "recorded"
    _, files = _campaign(recorded, market)
    # The files the run read are rewritten so that nothing can be built from them.
    for file in (*files["fields"], files["operators"]):
        Path(file).write_text("not: [a vocabulary\n")
    log = recorded / "run" / "llm_log"
    replayed = tmp_path / "replayed"
    path = write_config(
        replayed, market["components"], round=SETTINGS, **files,
        llm={role: {"backend": "replay", "model": f"model-{role}", "log": str(log)} for role in ROLES},
    )
    result = run(path, None, market["cache"])
    assert result.accepted
    assert (replayed / "run" / "ledger.jsonl").read_text() == (recorded / "run" / "ledger.jsonl").read_text()
    snapshot = recorded / "run" / "snapshot"
    for copy in snapshot.rglob("*.yaml"):
        assert (replayed / "run" / "snapshot" / copy.relative_to(snapshot)).read_text() == copy.read_text()
