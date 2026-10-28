"""The round in progress of runs real campaigns wrote: ``RunView.read(...).running`` and ``read_call``.

``finished`` ran one round of four slots with one retry: the planner's
reply is empty (a reasoning model that ran out of output tokens, so its
directions are the default ones), slot 0 passes and is found consistent,
slot 1 is refused by the Gate and fixed, slot 2 is found inconsistent,
rewritten and found inconsistent again, slot 3 passes and is found
consistent. Every reviewer reply is cut off at the output limit, as the
provider reports it. The round in progress is read from the same round
cut short in a process that is still running: ``after_1``, ``after_2``,
``after_3`` and ``after_5`` crashed at the LLM call after their first 1, 2,
3 or 5 calls came back (the proposer's first call, its fix, the reviewer's
first call and its review of the rewrite); ``waiting`` is ``after_3`` as it
read while that reviewer call was out; ``evaluating`` crashed in the
Evaluator, once every call had come back. ``resumed`` crashed at its first reviewer call of round 1 and was resumed,
to crash at once at its planner call.
"""

import json
import shutil

import pytest

from factor_agent.evaluator import Evaluator
from factor_agent.llm.base import Reply
from factor_agent.web.run_view import RunView, Tokens, read_call
from tests import synthetic_market
from tests.campaign_runner import candidate, crash_at, run, write_config
from tests.fake_llm import FakeLLM, Interrupted, round_of

ROUND = [
    candidate("planted", "alpha = close / open"),
    candidate("unknown name", "alpha = SecretOp(close)"),
    candidate("inconsistent thing", "alpha = Rank(BackRef(Min(open, close) / low, 10))"),
    candidate("noise", "alpha = BackRef(volume, 10) / WindowedAvg(volume, 60)"),
]

FIXED = candidate("volume surprise", "alpha = volume / WindowedAvg(volume, 20)")

#: Why the reviewer found a candidate consistent.
AGREED = "the code divides the close by the open, as the description says"

REASONING = "Comparing each candidate's code with its description."


def propose(number, directions):
    return ROUND


def fix(number, failures):
    return [FIXED if f["refused_version"]["code"] == "alpha = SecretOp(close)" else f["refused_version"]
            for f in failures]


class ReportingLLM(FakeLLM):
    """The fake as a reasoning model behind an OpenAI-compatible provider reports it.

    Each reply reports 100 input and 20 output tokens and the model's
    reasoning. The planner's reply is empty and the reviewer's is cut off
    at the output limit; the reviewer gives ``AGREED`` as its reason for
    every consistent candidate.
    """

    def complete(self, request):
        reply = super().complete(request)
        text = "" if request.role == "planner" else reply.text
        if request.role == "reviewer":
            text = text.replace('"reason": "consistent"', json.dumps({"reason": AGREED})[1:-1])
        finish = "length" if request.role == "reviewer" else "stop"
        raw = json.dumps({
            "choices": [{"finish_reason": finish, "message": {"content": text, "reasoning_content": REASONING}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                      "completion_tokens_details": {"reasoning_tokens": 5}},
        })
        return Reply(text=text, usage={"input_tokens": 100, "output_tokens": 20}, model_id=reply.model_id, raw=raw)


SETTINGS = {"candidates": 4, "min_per_type": 0, "retries": 1}


def _crashing_after(calls: int, copy: tuple | None = None) -> ReportingLLM:
    """The round's fake, crashing at the call after its first ``calls`` came back; ``copy`` copies the run then."""
    def interrupt(request):
        if len(fake.requests) < calls:
            return False
        if copy:
            shutil.copytree(*copy)
        return True
    fake = ReportingLLM(propose, fix=fix, interrupt=interrupt)
    return fake


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("round")
    components = synthetic_market.build(root / "data")
    cache = root / "kunquant_cache"

    def config(name: str, rounds: int = 1):
        return write_config(root / name, components, round={**SETTINGS, "max_rounds": rounds})

    run(config("finished"), ReportingLLM(propose, fix=fix), cache)

    for calls in (1, 2, 3, 5):
        copy = (root / "after_3" / "run", root / "waiting" / "run") if calls == 3 else None
        with pytest.raises(Interrupted):
            run(config(f"after_{calls}"), _crashing_after(calls, copy), cache)

    with crash_at(Evaluator, "evaluate"), pytest.raises(Interrupted):
        run(config("evaluating"), ReportingLLM(propose, fix=fix), cache)

    resumed = config("resumed", rounds=2)
    with pytest.raises(Interrupted):
        run(resumed, ReportingLLM(propose, fix=fix, interrupt=lambda r: r.role == "reviewer"), cache)
    shutil.copytree(root / "resumed" / "run", root / "crashed" / "run")
    with pytest.raises(Interrupted):
        run(resumed, ReportingLLM(propose, fix=fix, interrupt=lambda r: r.role == "planner"), cache)
    return root


def _running(runs, name: str):
    return RunView.read(runs / name / "run").running


def test_a_round_being_evaluated_shows_each_slots_pending_trial_at_the_stage_it_reached(runs):
    running = _running(runs, "evaluating")
    assert running.round == 1
    assert [(p.slot, p.stage) for p in running.pending] == [
        (0, "consistent"), (1, "consistent"), (2, "inconsistent"), (3, "consistent")]


@pytest.mark.parametrize("name, stages", [
    ("after_1", ["planned"] * 4),
    ("after_2", ["proposed"] * 4),
    ("after_3", ["proposed", "fixed", "proposed", "proposed"]),
    ("after_5", ["consistent", "consistent", "rewritten", "consistent"]),
])
def test_each_slot_reaches_its_stage_as_the_round_goes_on(runs, name, stages):
    assert [p.stage for p in _running(runs, name).pending] == stages


def test_a_pending_trial_holds_its_direction_its_latest_version_and_its_gate_and_reviewer_outcomes(runs):
    pending = _running(runs, "evaluating").pending
    # The planner's reply was empty: every slot has a new direction of its type.
    assert pending[1].direction["action"] == "new" and pending[1].direction["direction"]
    assert pending[1].latest["code"] == FIXED["code"]
    refused, fixed, review = pending[1].checks
    assert (refused.by, refused.step, refused.passed, refused.stage) == ("gate", "proposed", False, "gate")
    assert any("SecretOp" in reason for reason in refused.reasons)
    assert (fixed.step, fixed.passed, fixed.stage, fixed.reasons) == ("fixed", True, None, ())
    assert (review.by, review.passed, review.reasons) == ("reviewer", True, (AGREED,))
    assert [(c.by, c.step, c.passed) for c in pending[2].checks] == [
        ("gate", "proposed", True), ("reviewer", None, False), ("gate", "rewritten", True), ("reviewer", None, False)]
    assert pending[0].latest["code"] == "alpha = close / open"


def test_the_round_in_progress_shows_its_llm_calls_in_order_with_the_slots_each_concerned(runs):
    running = _running(runs, "evaluating")
    assert [(c.role, c.slots) for c in running.calls] == [
        ("planner", ()), ("proposer", (0, 1, 2, 3)), ("proposer", (1,)), ("reviewer", (0, 1, 2, 3)),
        ("proposer", (2,)), ("reviewer", (2,))]
    assert [c.number for c in running.calls] == [1, 2, 3, 4, 5, 6]
    assert all(c.sent <= c.returned and c.error is None for c in running.calls)
    assert running.started <= running.calls[0].sent and running.evaluation >= running.calls[-1].returned


def test_a_call_not_back_yet_is_on_the_timeline_without_a_number(runs):
    running = _running(runs, "waiting")
    assert [(c.role, c.number, c.returned, c.error) for c in running.calls][-1] == ("reviewer", None, None, None)
    assert running.evaluation is None


def test_a_call_that_failed_is_on_the_timeline_with_its_error_and_no_number(runs):
    reviewer = _running(runs, "after_3").calls[-1]
    assert (reviewer.role, reviewer.number, reviewer.returned is not None) == ("reviewer", None, True)
    assert "Interrupted" in reviewer.error


def test_a_round_dropped_by_a_resume_disappears_from_the_pending_trials(runs):
    crashed = RunView.read(runs / "crashed" / "run").running
    assert (crashed.round, len(crashed.pending)) == (1, 4)
    resumed = RunView.read(runs / "resumed" / "run").running
    assert (resumed.round, resumed.pending) == (1, ())
    # The planner's call after the resume crashed: it is on the timeline, failed, with no call file to read.
    [planner] = resumed.calls
    assert (planner.role, planner.number, planner.error is not None) == ("planner", None, True)


def test_a_stopped_run_has_no_round_in_progress(runs):
    assert RunView.read(runs / "finished" / "run").running is None


def test_a_run_without_an_events_file_shows_no_pending_trials_or_timeline_only_its_coarse_progress(runs, tmp_path):
    shutil.copytree(runs / "crashed" / "run", tmp_path / "run")
    (tmp_path / "run" / "events.jsonl").unlink()
    view = RunView.read(tmp_path / "run")
    assert view.running is None
    assert view.health.calls == {"planner": 1, "proposer": 2}


def test_an_llm_call_is_read_by_its_number_with_its_tokens_reply_reasoning_and_prompt(runs):
    run_dir = runs / "finished" / "run"
    review = read_call(run_dir, 4)
    assert (review.number, review.role, review.round, review.slots) == (4, "reviewer", 1, (0, 1, 2, 3))
    assert review.tokens == Tokens(input=100, output=20, cache_hits=None, reasoning=5)
    assert (review.truncated, review.empty) == (True, False)
    assert AGREED in review.reply and review.reasoning == REASONING
    assert "<candidates>" in review.prompt and review.seconds is not None
    planner = read_call(run_dir, 1)
    assert (planner.role, planner.slots, planner.truncated, planner.empty, planner.reply) == (
        "planner", (), False, True, "")


def test_a_call_number_the_run_has_no_file_for_reads_as_none(runs):
    assert read_call(runs / "finished" / "run", 99) is None
    assert read_call(runs / "finished" / "run", 0) is None


def test_a_call_of_a_run_without_an_events_file_is_read_from_its_call_file_alone(runs, tmp_path):
    shutil.copytree(runs / "finished" / "run", tmp_path / "run")
    (tmp_path / "run" / "events.jsonl").unlink()
    review = read_call(tmp_path / "run", 4)
    assert (review.role, review.round, review.slots, review.seconds, review.truncated) == (
        "reviewer", None, (), None, True)
    assert review.tokens.input == 100
