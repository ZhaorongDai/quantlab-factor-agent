"""The health of runs real campaigns wrote, read from their events file: ``RunView.read(...).health``.

Every LLM role is a fake that reports token counts as an OpenAI-compatible
provider does. ``finished`` ran two rounds of four slots to its round
limit. ``mid`` was cut by a crash at its first reviewer call of round 2 of
4, in this process, which is still running; the call failed a second
after it was sent. ``resumed`` is ``mid`` resumed
and cut at the same call again. ``crashed`` crashed at its first reviewer
call in a process of its own that has exited. ``again`` is ``finished``
resumed with the same round limit, so it stops at once. ``evaluating``
crashed in the Evaluator in its first round, and ``planning`` between the
end of its first round and the first call of its second, both in this
process. ``old`` is ``mid`` as a campaign without an events file leaves it.
"""

import json
import shutil
import subprocess
import sys
import textwrap
import time
from datetime import timedelta
from pathlib import Path

import pytest

from factor_agent import _admission, _scheduler
from factor_agent.config import AdmissionConfig
from factor_agent.evaluator import Evaluator
from factor_agent.llm.base import Reply
from factor_agent.web.run_view import RoundTokens, RunView, Tokens, list_runs
from tests import synthetic_market
from tests.campaign_runner import candidate, crash_at, run, write_config
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


def slowly_at_round_2_review(request):
    """``at_round_2_review``, the failing call taking a second."""
    if at_round_2_review(request):
        time.sleep(1)
        return True
    return False


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
        run(mid, CountingLLM(propose, interrupt=slowly_at_round_2_review), cache)

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

    evaluating = write_config(root / "evaluating", components, round={**ROUND, "max_rounds": 2})
    with crash_at(Evaluator, "evaluate"), pytest.raises(Interrupted):
        run(evaluating, CountingLLM(propose), cache)

    planning = write_config(root / "planning", components, round={**ROUND, "max_rounds": 2})
    with crash_at(_scheduler, "quota", when=lambda number, *_: number == 2), pytest.raises(Interrupted):
        run(planning, CountingLLM(propose), cache)

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


def test_a_call_that_came_back_keeps_a_run_asking_the_llm_from_stalling_as_one_sent_does(runs):
    view = RunView.read(runs / "mid" / "run")
    reviewer = view.running.calls[-1]
    assert reviewer.role == "reviewer" and reviewer.returned - reviewer.sent >= timedelta(seconds=1)
    assert not _health(runs, "mid", now=reviewer.sent + timedelta(minutes=15, seconds=0.5)).stalled
    assert _health(runs, "mid", now=reviewer.returned + timedelta(minutes=15, seconds=0.5)).stalled


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


def test_the_trials_page_places_each_trial_against_the_threshold_admission_judged_its_round_by(runs):
    # Round 1 evaluates three candidates and round 2 four: admission judged round 2 as at seven trials.
    view = RunView.read(runs / "finished" / "run")
    expected = _admission.t_threshold(AdmissionConfig(), 7)
    assert [p.threshold for p in view.evaluated if p.round == 2] == [pytest.approx(expected)] * 4
    assert view.threshold == pytest.approx(expected)


def test_a_run_evaluating_its_round_never_stalls(runs):
    run_dir = runs / "evaluating" / "run"
    health = RunView.read(run_dir).health
    assert (health.phase, health.round) == ("evaluating", 1)
    # The evaluation starts as soon as the round's last call is back; an intraday round's lasts hours.
    started = health.last_call
    assert not RunView.read(run_dir, now=started + timedelta(hours=3)).health.stalled


def test_a_run_whose_round_is_written_is_planning_the_next_and_never_stalls(runs):
    run_dir = runs / "planning" / "run"
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


def test_the_run_list_says_whether_each_run_stopped_runs_or_crashed_as_its_health_does(runs):
    listed = {s.name.split("/")[0]: s for s in list_runs(runs)}
    assert {name: listed[name].state for name in ("finished", "mid", "crashed", "old")} == {
        "finished": "stopped", "mid": "running", "crashed": "crashed", "old": "unknown"}
    assert (listed["mid"].max_rounds, listed["finished"].max_rounds, listed["old"].max_rounds) == (4, 2, None)


def test_the_run_list_never_calls_a_run_opened_on_another_host_crashed(runs, monkeypatch):
    monkeypatch.setattr("socket.gethostname", lambda: "another-host")
    listed = {s.name.split("/")[0]: s for s in list_runs(runs)}
    assert (listed["mid"].state, listed["crashed"].state) == ("unknown", "unknown")
