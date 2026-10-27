"""The health of runs real campaigns wrote, read from their events file: ``RunView.read(...).health``.

Every LLM role is a fake that reports token counts as an OpenAI-compatible
provider does. ``finished`` ran two rounds of four slots to its round
limit. ``mid`` was cut by a crash at its first reviewer call of round 2 of
4, in this process, which is still running. ``resumed`` is ``mid`` resumed
and cut at the same call again. ``crashed`` crashed at its first reviewer
call in a process of its own that has exited. ``again`` is ``finished``
resumed with the same round limit, so it stops at once. ``old`` is ``mid``
as a campaign without an events file leaves it.
"""

import json
import shutil
import subprocess
import sys
import textwrap
from datetime import timedelta
from pathlib import Path

import pytest

from factor_agent.llm.base import Reply
from factor_agent.web.run_view import RoundTokens, RunView, Tokens
from tests import synthetic_market
from tests.campaign_runner import candidate, run, write_config
from tests.fake_llm import FakeLLM, Interrupted, round_of

ROUND_1 = [
    candidate("planted", "alpha = close / open"),
    candidate("noise", "alpha = BackRef(volume, 10) / WindowedAvg(volume, 60)"),
    candidate("unknown name", "alpha = SecretOp(close)"),
    candidate("more noise", "alpha = BackRef(close, 20) / BackRef(WindowedAvg(close, 10), 5)"),
]


def later_round(number: int, directions: list[dict]) -> list[dict]:
    fields = ("open", "high", "low", "close", "volume")
    return [
        candidate(f"noise {number}.{slot}", f"alpha = BackRef({fields[slot]}, {(5, 10, 20, 40)[number - 2]}) / "
                                            f"BackRef(WindowedAvg({fields[slot]}, 20), 5)")
        for slot in range(len(directions))
    ]


def propose(number, directions):
    return ROUND_1 if number == 1 else later_round(number, directions)


#: What every call reports: 100 input tokens of which 30 cached, 20 output tokens of which 5 reasoning.
USAGE = {"input_tokens": 100, "output_tokens": 20}
RAW = json.dumps({
    "choices": [{"finish_reason": "stop", "message": {"content": "..."}}],
    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 30},
              "completion_tokens_details": {"reasoning_tokens": 5}},
})


class CountingLLM(FakeLLM):
    """The fake, reporting ``USAGE`` and an OpenAI-compatible raw reply on every call."""

    def complete(self, request):
        reply = super().complete(request)
        return Reply(text=reply.text, usage=USAGE, model_id=reply.model_id, raw=RAW)


ROUND = {"candidates": 4, "min_per_type": 0, "retries": 0}


def at_round_2_review(request):
    return request.role == "reviewer" and round_of(request.prompt) == 2


CRASH = textwrap.dedent("""
    import sys
    from pathlib import Path
    from tests.campaign_runner import run
    from tests.fake_llm import FakeLLM
    from tests.test_run_health import propose
    run(Path(sys.argv[1]), FakeLLM(propose, interrupt=lambda r: r.role == "reviewer"), Path(sys.argv[2]))
""")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("health")
    components = synthetic_market.build(root / "data")
    cache = root / "kunquant_cache"

    finished = write_config(root / "finished", components, round={**ROUND, "max_rounds": 2})
    run(finished, CountingLLM(propose), cache)

    mid = write_config(root / "mid", components, round={**ROUND, "max_rounds": 4})
    with pytest.raises(Interrupted):
        run(mid, CountingLLM(propose, interrupt=at_round_2_review), cache)

    shutil.copytree(root / "mid" / "run", root / "resumed" / "run")
    resumed = write_config(root / "resumed", components, round={**ROUND, "max_rounds": 4})
    with pytest.raises(Interrupted):
        run(resumed, CountingLLM(propose, interrupt=at_round_2_review), cache)

    crashed = write_config(root / "crashed", components, round={**ROUND, "max_rounds": 2})
    exited = subprocess.run([sys.executable, "-c", CRASH, str(crashed), str(cache)],
                            cwd=Path(__file__).parent.parent, capture_output=True, text=True)
    assert "Interrupted" in exited.stderr

    shutil.copytree(root / "finished" / "run", root / "again" / "run")
    run(write_config(root / "again", components, round={**ROUND, "max_rounds": 2}), CountingLLM(propose), cache)

    shutil.copytree(root / "mid" / "run", root / "old" / "run")
    (root / "old" / "run" / "events.jsonl").unlink()
    return root


def _health(runs, name, now=None):
    return RunView.read(runs / name / "run", now=now).health


def test_a_run_cut_mid_round_in_a_running_process_is_asking_the_llm_in_its_round_against_the_round_limit(runs):
    health = _health(runs, "mid")
    assert (health.phase, health.round, health.max_rounds) == ("asking", 2, 4)
    assert (health.process, health.crashed) == ("running", False)


def test_a_run_asking_the_llm_stalls_when_no_call_came_for_fifteen_minutes(runs):
    last = _health(runs, "mid").last_call
    assert last is not None and not _health(runs, "mid").stalled
    assert not _health(runs, "mid", now=last + timedelta(minutes=14)).stalled
    assert _health(runs, "mid", now=last + timedelta(minutes=16)).stalled


def test_a_run_whose_process_exited_without_recording_a_stop_reads_as_crashed(runs):
    view = RunView.read(runs / "crashed" / "run")
    assert view.summary.stop is None
    assert (view.health.process, view.health.crashed, view.health.phase) == ("gone", True, None)


def test_the_process_of_a_run_opened_on_another_host_is_unknown_and_never_called_gone(runs, monkeypatch):
    monkeypatch.setattr("socket.gethostname", lambda: "another-host")
    for name in ("mid", "crashed"):
        health = _health(runs, name)
        assert (health.process, health.crashed) == ("unknown", False)


def test_a_run_resumed_and_stopped_again_reads_as_stopped_with_no_phase_or_round(runs):
    for name in ("finished", "again"):
        view = RunView.read(runs / name / "run")
        assert view.summary.stop.by == "max_rounds"
        assert (view.health.phase, view.health.round, view.health.crashed, view.health.calls) == (None, None, False, {})


def _when(runs, tmp_path, kind: str, nth: int = 1) -> Path:
    """``finished`` as it read just after its ``nth`` event of type ``kind``: a copy cut there, its state with it."""
    shutil.copytree(runs / "finished" / "run", tmp_path / "run")
    path = tmp_path / "run" / "events.jsonl"
    lines = path.read_text().splitlines(keepends=True)
    cut = [i for i, line in enumerate(lines) if json.loads(line)["type"] == kind][nth - 1] + 1
    path.write_text("".join(lines[:cut]))
    written = sum(json.loads(line)["type"] == "round_end" for line in lines[:cut])
    state = json.loads((tmp_path / "run" / "state.json").read_text())
    (tmp_path / "run" / "state.json").write_text(json.dumps({**state, "rounds": written, "stop": None}))
    return tmp_path / "run"


def test_a_run_evaluating_its_round_stalls_after_thirty_minutes_of_evaluation(runs, tmp_path):
    run_dir = _when(runs, tmp_path, "evaluation")
    health = RunView.read(run_dir).health
    assert (health.phase, health.round) == ("evaluating", 1)
    # The evaluation starts as soon as the round's last call is back.
    started = health.last_call
    assert not RunView.read(run_dir, now=started + timedelta(minutes=29)).health.stalled
    assert RunView.read(run_dir, now=started + timedelta(minutes=31)).health.stalled


def test_a_run_whose_round_is_written_is_planning_the_next_and_never_stalls(runs, tmp_path):
    run_dir = _when(runs, tmp_path, "round_end")
    health = RunView.read(run_dir).health
    assert (health.phase, health.round) == ("planning", 2)
    assert not RunView.read(run_dir, now=health.last_call + timedelta(hours=2)).health.stalled


def test_tokens_are_counted_per_round_and_in_total_as_the_provider_reports_them(runs):
    # Three calls a round (planner, proposer, reviewer), each 100 in (30 cached) and 20 out (5 reasoning).
    health = _health(runs, "finished")
    assert health.tokens == Tokens(input=600, output=120, cache_hits=180, reasoning=30)
    assert health.round_tokens == (
        RoundTokens(round=1, tokens=Tokens(input=300, output=60, cache_hits=90, reasoning=15)),
        RoundTokens(round=2, tokens=Tokens(input=300, output=60, cache_hits=90, reasoning=15)),
    )


def test_a_resume_drops_the_half_written_rounds_calls_from_its_progress_but_keeps_their_tokens(runs):
    mid, resumed = _health(runs, "mid"), _health(runs, "resumed")
    assert mid.calls == resumed.calls == {"planner": 1, "proposer": 1}
    assert resumed.round == 2
    # Round 2's planner and proposer answered twice, once before the crash and once after the resume.
    assert resumed.round_tokens[-1] == RoundTokens(
        round=2, tokens=Tokens(input=400, output=80, cache_hits=120, reasoning=20))


def test_the_time_left_is_estimated_from_the_written_rounds_and_counts_down(runs):
    later = _health(runs, "mid").last_call + timedelta(seconds=1)
    left = _health(runs, "mid", now=later).time_left
    # Round 1 of 4 is written: three more rounds as long as it took, round 2 begun.
    assert left > 0
    assert _health(runs, "mid", now=later + timedelta(seconds=0.5)).time_left == pytest.approx(left - 0.5)
    assert _health(runs, "mid", now=later + timedelta(days=1)).time_left == 0
    assert _health(runs, "finished").time_left is None and _health(runs, "crashed").time_left is None


def test_a_run_without_an_events_file_has_only_its_coarse_progress_from_the_call_files(runs):
    health = _health(runs, "old")
    assert health.calls == {"planner": 1, "proposer": 1}
    assert (health.phase, health.process, health.crashed, health.round, health.max_rounds) == (
        None, "unknown", False, 2, None)
    assert (health.tokens, health.round_tokens, health.stalled) == (None, (), False)
