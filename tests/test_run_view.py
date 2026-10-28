"""Reading run directories that real campaigns wrote: ``list_runs`` and ``RunView.read``.

Every run lives under one folder, as a user keeps them. ``finished`` ran two
rounds of four slots to its round limit: round 1 holds a planted signal
(accepted), noise (below the t threshold), code the Gate refuses and a
candidate the reviewer finds inconsistent; round 2 holds noise. ``fresh``
crashed before its first round was written. ``resumed`` is ``finished``
resumed with a higher round limit, crashed in its fourth round: its state
file still holds the stop of the first open. ``ahead`` is ``finished`` as a
crash leaves it between the ledger's append of round 2 and the state's
rewrite: the ledger is a round ahead of the state. Every run asks the
distiller before each round after the first, so ``finished`` holds one
version of the memory and ``resumed`` three.
"""

import json
import shutil

import pytest

from factor_agent.web.run_view import RunView, list_runs
from tests import synthetic_market
from tests.campaign_runner import candidate, run, write_config
from tests.fake_llm import FakeLLM, Interrupted, round_of

ROUND_1 = [
    candidate("planted", "alpha = close / open"),
    candidate("noise", "alpha = BackRef(volume, 10) / WindowedAvg(volume, 60)"),
    candidate("unknown name", "alpha = SecretOp(close)"),
    candidate("inconsistent thing", "alpha = Rank(BackRef(Min(open, close) / low, 10))"),
]


def later_round(number: int, directions: list[dict]) -> list[dict]:
    """Noise in every slot: a lagged field over its lagged moving average, new in each round and slot."""
    fields = ("open", "high", "low", "close", "volume")
    return [
        candidate(f"noise {number}.{slot}", f"alpha = BackRef({fields[slot]}, {(5, 10, 20, 40)[number - 2]}) / "
                                            f"BackRef(WindowedAvg({fields[slot]}, 20), 5)")
        for slot in range(len(directions))
    ]


def propose(number, directions):
    return ROUND_1 if number == 1 else later_round(number, directions)


ROUND = {"candidates": 4, "min_per_type": 0, "retries": 1, "memory_every": 1}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("work")
    components = synthetic_market.build(root / "data")
    cache = root / "kunquant_cache"
    folder = root / "runs"

    finished = write_config(folder / "finished", components, round={**ROUND, "max_rounds": 2})
    run(finished, FakeLLM(propose), cache)

    fresh = write_config(folder / "fresh", components, round={**ROUND, "max_rounds": 2})
    with pytest.raises(Interrupted):
        run(fresh, FakeLLM(propose, interrupt=lambda r: r.role == "reviewer"), cache)

    shutil.copytree(folder / "finished" / "run", folder / "resumed" / "run")
    resumed = write_config(folder / "resumed", components, round={**ROUND, "max_rounds": 4})
    crash = FakeLLM(propose, interrupt=lambda r: r.role == "reviewer" and round_of(r.prompt) == 4)
    with pytest.raises(Interrupted):
        run(resumed, crash, cache)

    shutil.copytree(folder / "finished" / "run", folder / "ahead" / "run")
    state_path = folder / "ahead" / "run" / "state.json"
    state = json.loads(state_path.read_text())
    state_path.write_text(json.dumps({**state, "rounds": 1, "log": state["log"][:1], "stop": None}, indent=2) + "\n")
    return folder


def _by_name(folder):
    return {summary.name: summary for summary in list_runs(folder)}


def test_listing_finds_every_run_under_the_folder_even_one_whose_first_round_is_not_written(runs):
    listed = _by_name(runs)
    assert set(listed) == {"finished/run", "fresh/run", "resumed/run", "ahead/run"}
    fresh = listed["fresh/run"]
    assert (fresh.rounds, fresh.trials, fresh.accepted, fresh.stop) == (0, 0, 0, None)


def test_the_listing_summarises_each_run_as_reading_the_run_does(runs):
    for summary in list_runs(runs):
        assert summary == RunView.read(runs / summary.name, name=summary.name).summary
        assert summary.unreadable is None


def test_a_run_whose_files_cannot_be_read_is_listed_as_unreadable_without_breaking_the_listing(runs, tmp_path):
    for name in ("finished", "broken"):
        shutil.copytree(runs / "finished" / "run", tmp_path / name)
    (tmp_path / "broken" / "state.json").write_text('{"rounds": 2, "sto')
    listed = _by_name(tmp_path)
    assert listed["finished"].unreadable is None and listed["finished"].trials == 8
    broken = listed["broken"]
    assert "state.json" in broken.unreadable
    assert (broken.rounds, broken.trials, broken.accepted, broken.stop) == (0, 0, 0, None)


def test_the_listing_shows_the_latest_activity_first(runs):
    times = [summary.updated for summary in list_runs(runs)]
    assert times == sorted(times, reverse=True)


def test_a_finished_run_reads_as_its_rounds_trials_accepted_factors_and_stop(runs):
    view = RunView.read(runs / "finished" / "run")
    summary = view.summary
    snapshot = json.loads((runs / "finished" / "run" / "snapshot.json").read_text())
    assert summary.code_version == snapshot["versions"][-1]["factor_agent"]
    assert (summary.rounds, summary.trials, summary.accepted) == (2, 8, 1)
    assert summary.stop.by == "max_rounds"
    # A datetime keeps microseconds only, so the file's time may round a microsecond either way.
    assert summary.updated.timestamp() >= (runs / "finished" / "run" / "state.json").stat().st_mtime - 1e-6
    assert [(t.round, t.slot, t.status, t.stage) for t in view.trials if t.round == 1] == [
        (1, 0, "accepted", "admission"), (1, 1, "rejected", "admission"),
        (1, 2, "failed", "gate"), (1, 3, "rejected", "review"),
    ]
    assert view.trials[0].candidate["description"] == "planted"


def test_nan_in_the_ledger_reads_as_none(runs):
    # The run has no known factors: every evaluated trial's nearest known correlation is NaN in the ledger.
    planted = RunView.read(runs / "finished" / "run").trials[0]
    assert "NaN" in (runs / "finished" / "run" / "ledger.jsonl").read_text().splitlines()[0]
    assert planted.metrics["nearest_known"] == "" and planted.metrics["nearest_known_correlation"] is None


def test_ledger_rounds_beyond_the_state_file_are_not_read(runs):
    view = RunView.read(runs / "ahead" / "run")
    assert (view.summary.rounds, view.summary.trials, view.summary.accepted) == (1, 4, 1)
    assert {t.round for t in view.trials} == {1}
    assert view.summary.stop is None


def test_a_resumed_run_that_has_not_stopped_again_has_no_stop_although_its_state_keeps_the_old_one(runs):
    state = json.loads((runs / "resumed" / "run" / "state.json").read_text())
    assert state["rounds"] == 3 and state["stop"]["by"] == "max_rounds"
    summary = RunView.read(runs / "resumed" / "run").summary
    assert (summary.rounds, summary.trials, summary.stop) == (3, 12, None)


def test_without_events_a_resumed_run_that_has_not_stopped_again_has_no_stop_either(runs, tmp_path):
    shutil.copytree(runs / "resumed" / "run", tmp_path / "run")
    (tmp_path / "run" / "events.jsonl").unlink()
    assert RunView.read(tmp_path / "run").summary.stop is None
    shutil.copytree(runs / "finished" / "run", tmp_path / "finished")
    (tmp_path / "finished" / "events.jsonl").unlink()
    assert RunView.read(tmp_path / "finished").summary.stop.by == "max_rounds"


def test_a_run_with_no_memory_yet_has_no_versions(runs):
    assert RunView.read(runs / "fresh" / "run").memory == ()


def test_the_first_version_of_the_memory_has_its_text_and_no_changes(runs):
    (version,) = RunView.read(runs / "finished" / "run").memory
    assert version.round == 1
    assert version.text == (runs / "finished" / "run" / "memory.md").read_text().rstrip("\n")
    assert version.changes is None


def test_every_version_after_the_first_says_what_changed_from_the_one_before(runs):
    versions = RunView.read(runs / "resumed" / "run").memory
    assert [v.round for v in versions] == [1, 2, 3]
    # The fake distiller numbers its memories by its calls, and the resume started a new fake: the
    # versions after rounds 1 and 2 are both its first memory, the one after round 3 its second.
    assert {c.change for c in versions[1].changes} == {"kept"}
    changes = versions[2].changes
    assert [(c.change, c.line) for c in changes if c.change != "kept"] == [
        ("removed", "- Distilled memory number 1: volume surprises keep working."),
        ("added", "- Distilled memory number 2: volume surprises keep working."),
    ]
    assert [c.line for c in changes if c.change != "removed"] == versions[2].text.splitlines()
