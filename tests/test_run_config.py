"""The config a run was started and resumed with, and its snapshot's files, as the frontend reads them.

``run`` ran one round of a config, then was resumed with the same config
raised to two rounds.
"""

import shutil

import pytest
import yaml

from factor_agent.web.run_view import read_files
from tests import synthetic_market
from tests.campaign_runner import candidate, run, write_config
from tests.fake_llm import FakeLLM

ROUND = {"candidates": 4, "min_per_type": 0, "max_rounds": 1}


def propose(number, directions):
    return [candidate(f"planted {number}.{slot}", "alpha = close / open") for slot in range(4)]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("config")
    components = synthetic_market.build(root / "data")
    cache = root / "kunquant_cache"
    path = write_config(root / "run", components, round=ROUND)
    run(path, FakeLLM(propose), cache)
    first = path.read_text()
    raised = yaml.safe_load(first)
    raised["round"]["max_rounds"] = 2
    path.write_text(yaml.safe_dump(raised))
    run(path, FakeLLM(propose), cache)
    return root, first, path.read_text()


def test_each_opening_of_a_run_keeps_the_config_it_was_given(runs):
    root, first, second = runs
    copies = sorted((root / "run" / "run" / "config").iterdir())
    assert [p.name for p in copies] == ["open-1.yaml", "open-2.yaml"]
    assert [p.read_text() for p in copies] == [first, second]


def test_the_frontend_reads_the_configs_by_opening_the_latest_first_and_the_snapshots_files(runs):
    root, first, second = runs
    files = read_files(root / "run" / "run")
    assert [(c.opening, c.text) for c in files.configs] == [(2, second), (1, first)]
    assert all(c.opened_at is not None for c in files.configs)
    names = [f.name for f in files.snapshot]
    assert names and all(n.startswith(("fields/", "operators/", "prompts/")) for n in names)
    field_file = next(f for f in files.snapshot if f.name.startswith("fields/"))
    assert field_file.text == (root / "run" / "run" / "snapshot" / field_file.name).read_text()


def test_a_run_written_before_configs_were_kept_shows_only_its_snapshot(runs, tmp_path):
    root, _, _ = runs
    old = tmp_path / "old"
    shutil.copytree(root / "run" / "run", old)
    shutil.rmtree(old / "config")
    files = read_files(old)
    assert files.configs == () and files.snapshot


def test_each_config_after_the_first_says_what_changed_from_the_opening_before(runs):
    root, _, _ = runs
    latest, first = read_files(root / "run" / "run").configs
    changed = [(line.change, line.line.strip()) for line in latest.changes if line.change != "kept"]
    assert changed == [("removed", "max_rounds: 1"), ("added", "max_rounds: 2")]
    assert first.changes is None
