"""What the evaluated trials measure: the planner's groups of them, and a new direction sent back for repeating one.

With ``context.theme_correlation`` the Evaluator keeps every evaluated
trial on a sample of the training dates, the planner reads the trials
grouped by the correlation of their factors, and a new direction's
candidate as correlated with an earlier trial is sent back once, before it
is evaluated. Every role is answered by ``tests/fake_llm.FakeLLM``; the
Gate and the Evaluator are the real ones on the synthetic store.
"""

import json

import numpy as np
import pytest

from factor_agent.config import ContextConfig
from tests import synthetic_market
from tests.campaign_runner import candidate, run, write_config
from tests.fake_llm import FakeLLM, Interrupted, round_of

INTRADAY = candidate("intraday move", "alpha = close / open")
#: The intraday move again, measured another way: its normal scores are those of the intraday move.
AGAIN = candidate("intraday move again", "alpha = Log(close / open) * 2")
VOLUME = candidate("volume shock", "alpha = volume / WindowedAvg(volume, 20)")


def noise(number: int, slot: int) -> dict:
    return candidate(f"noise of round {number} slot {slot}", f"alpha = BackRef(close, {5 + 10 * number + slot}) / BackRef(open, 5)")


@pytest.fixture(scope="module")
def market(tmp_path_factory):
    root = tmp_path_factory.mktemp("market")
    return {"components": synthetic_market.build(root / "data"), "cache": root / "kunquant_cache"}


def _config(root, market, theme_correlation=0.5, send_back_repeats=True, **round_settings):
    settings = {"candidates": 5, "min_per_type": 1, "retries": 0, "max_rounds": 2, **round_settings}
    context = {} if theme_correlation is None else {"theme_correlation": theme_correlation}
    if send_back_repeats:
        context["send_back_repeats"] = True
    return write_config(root, market["components"], round=settings, context=context)


def _first_slot(first: dict, later: dict | None = None):
    """The proposer: ``first`` in the first slot of round 1, ``later`` in that of the later rounds, noise elsewhere."""
    def propose(number, directions):
        chosen = first if number == 1 else later
        return [chosen if slot == 0 and chosen else noise(number, slot) for slot in range(len(directions))]
    return propose


def _fix_to(item: dict):
    """The proposer's rewrite: ``item`` for a candidate sent back for repeating a trial, the same version otherwise."""
    return lambda number, failures: [
        item if any("measures what earlier trial" in r for r in f["refused_because"]) else f["refused_version"]
        for f in failures
    ]


@pytest.fixture(scope="module")
def repeated(tmp_path_factory, market):
    fake = FakeLLM(_first_slot(INTRADAY, AGAIN), fix=_fix_to(VOLUME))
    root = tmp_path_factory.mktemp("repeated")
    result = run(_config(root, market), fake, market["cache"])
    return {"result": result, "fake": fake, "run_dir": root / "run"}


def _slot(records, number: int, slot: int):
    return [r for r in records if r.round == number][slot]


def test_a_new_direction_repeating_an_earlier_trial_is_rewritten_before_it_is_evaluated(repeated):
    record = _slot(repeated["result"].records, 2, 0)
    assert record.candidate["description"] == "volume shock"
    assert record.evaluated
    [attempt] = [a for a in record.attempts if a["stage"] == "novelty"]
    assert attempt["item"]["description"] == "intraday move again"
    [reason] = attempt["reasons"]
    assert reason.startswith(
        "the research direction is new, but the factor measures what earlier trial T1-01 measured: "
        "its correlation with T1-01's factor is +1.00, at least 0.5 in absolute value (T1-01 computed: intraday move; "
    )


def test_the_sent_back_version_is_neither_evaluated_nor_a_trial(repeated):
    records = repeated["result"].records
    assert not any(r.candidate.get("description") == "intraday move again" for r in records)
    events = [json.loads(line) for line in (repeated["run_dir"] / "events.jsonl").read_text().splitlines()]
    admitted = [e for e in events if e["type"] == "admission" and e["round"] == 2]
    assert len(admitted) == sum(r.evaluated for r in records if r.round == 2)
    assert {e["slot"] for e in admitted} >= {0}


def test_the_rewrite_request_names_the_trial_and_no_statistic_of_the_repeated_version(repeated):
    [fix] = [r for r in repeated["fake"].of_role("proposer") if "<refused_candidates>" in r.prompt]
    assert round_of(fix.prompt) == 2
    failures = json.loads(fix.prompt.split("<refused_candidates>\n", 1)[1].split("\n</refused_candidates>", 1)[0])
    [failure] = failures
    assert failure["refused_version"]["description"] == "intraday move again"
    assert "residual" not in json.dumps(failure) and "IC" not in json.dumps(failure)


def test_the_planner_reads_the_trials_grouped_by_what_their_factors_measure(repeated):
    first, second = repeated["fake"].of_role("planner")
    assert "<themes>" not in first.prompt
    themes = second.prompt.split("<themes>\n", 1)[1].split("\n</themes>", 1)[0]
    assert "at least 0.5 in absolute value" in themes
    assert "- opened by T1-01 (intraday move): 1 trial(s) (T1-01), 1 accepted (T1-01)" in themes


def test_a_refinement_measuring_what_its_parent_measured_is_not_sent_back(tmp_path, market):
    def plan(number, quota, prompt):
        directions = [
            {"mechanism_type": kind, "action": "new", "parent": "", "direction": f"a {kind} direction"}
            for kind, count in quota.items() for _ in range(count)
        ]
        if number > 1:
            directions[0] = {**directions[0], "action": "refinement", "parent": "T1-01"}
        return directions

    fake = FakeLLM(_first_slot(INTRADAY, AGAIN), fix=_fix_to(VOLUME), plan=plan)
    result = run(_config(tmp_path, market), fake, market["cache"])
    record = _slot(result.records, 2, 0)
    assert record.candidate["description"] == "intraday move again"
    assert not any(a["stage"] == "novelty" for a in record.attempts)


def test_a_resumed_run_still_knows_the_trials_of_the_rounds_written(tmp_path, market):
    interrupted = FakeLLM(_first_slot(INTRADAY, AGAIN), fix=_fix_to(VOLUME), interrupt=lambda r: round_of(r.prompt) == 2)
    with pytest.raises(Interrupted):
        run(_config(tmp_path, market), interrupted, market["cache"])
    fake = FakeLLM(_first_slot(INTRADAY, AGAIN), fix=_fix_to(VOLUME))
    result = run(_config(tmp_path, market), fake, market["cache"])
    record = _slot(result.records, 2, 0)
    assert record.candidate["description"] == "volume shock"
    assert any(a["stage"] == "novelty" for a in record.attempts)


def test_without_the_settings_the_repeat_is_evaluated_and_no_group_is_shown(tmp_path, market):
    fake = FakeLLM(_first_slot(INTRADAY, AGAIN), fix=_fix_to(VOLUME))
    result = run(_config(tmp_path, market, theme_correlation=None, send_back_repeats=False), fake, market["cache"])
    assert _slot(result.records, 2, 0).candidate["description"] == "intraday move again"
    assert not any("<themes>" in r.prompt for r in fake.of_role("planner"))
    snapshot = json.dumps(json.loads((tmp_path / "run" / "snapshot.json").read_text()))
    assert "theme_correlation" not in snapshot and "send_back_repeats" not in snapshot


def test_the_groups_alone_send_nothing_back_and_do_not_say_they_would(tmp_path, market):
    fake = FakeLLM(_first_slot(INTRADAY, AGAIN), fix=_fix_to(VOLUME))
    result = run(_config(tmp_path, market, send_back_repeats=False), fake, market["cache"])
    record = _slot(result.records, 2, 0)
    assert record.candidate["description"] == "intraday move again"
    assert not any(a["stage"] == "novelty" for a in record.attempts)
    second = fake.of_role("planner")[1].prompt
    assert "- opened by T1-01 (intraday move)" in second and "sent back" not in second.split("<themes>", 1)[1]


@pytest.mark.parametrize("value", [0, 1, 1.5, -0.2])
def test_the_theme_correlation_lies_strictly_between_zero_and_one(value):
    with pytest.raises(ValueError, match="theme_correlation"):
        ContextConfig(theme_correlation=value)


def test_the_groups_need_the_earlier_trials():
    with pytest.raises(ValueError, match="needs feedback"):
        ContextConfig(feedback=False, theme_correlation=0.5)


def test_the_theme_correlation_is_a_number():
    with pytest.raises(TypeError, match="theme_correlation"):
        ContextConfig(theme_correlation=True)


def test_sending_back_a_repeat_needs_the_correlation_at_which_a_trial_repeats_one():
    with pytest.raises(ValueError, match="send_back_repeats needs theme_correlation"):
        ContextConfig(send_back_repeats=True)
    with pytest.raises(TypeError, match="send_back_repeats"):
        ContextConfig(theme_correlation=0.5, send_back_repeats="yes")


def test_a_trial_joins_the_group_of_the_earlier_trial_it_is_most_correlated_with():
    from factor_agent._feedback import theme_groups

    names = ["T1-01", "T1-02", "T2-01", "T2-02"]
    matrix = np.array([
        [1.0, 0.8, 0.1, 0.2],
        [0.8, 1.0, 0.0, -0.6],
        [0.1, 0.0, 1.0, 0.55],
        [0.2, -0.6, 0.55, 1.0],
    ])
    # T2-02 is +0.55 with T2-01 and -0.6 with T1-02: it joins T1-02's group, by the larger absolute correlation.
    assert theme_groups(names, matrix, 0.5) == [["T1-01", "T1-02", "T2-02"], ["T2-01"]]
    assert theme_groups(names, matrix, 0.9) == [[name] for name in names]


def test_the_groups_follow_the_trials_order_whatever_the_order_they_are_given_in():
    from factor_agent._feedback import theme_groups

    names = ["T1-01", "T1-02", "T2-01", "T2-02"]
    matrix = np.array([
        [1.0, 0.8, 0.1, 0.2],
        [0.8, 1.0, 0.0, -0.6],
        [0.1, 0.0, 1.0, 0.55],
        [0.2, -0.6, 0.55, 1.0],
    ])
    shuffled = [3, 0, 2, 1]
    assert theme_groups([names[k] for k in shuffled], matrix[np.ix_(shuffled, shuffled)], 0.5) == (
        theme_groups(names, matrix, 0.5)
    )
