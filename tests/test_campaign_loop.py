"""Campaign.run() over several rounds: roles, scheduling, feedback, memory, stop and resume.

Every role is answered by ``tests/fake_llm.FakeLLM``; the Gate and the
Evaluator are the real ones on the synthetic store. The proposer fills
every liquidity slot with a planted signal it has not yet seen accepted and
every other slot with noise: a lagged field over its moving average, a
different one per round and slot.
"""

import collections
import json
import re

import pytest

from factor_agent.config import DEFAULT_MECHANISMS
from tests import synthetic_market
from tests.campaign_runner import candidate, run, write_config
from tests.fake_llm import FakeLLM, Interrupted, has_tag, round_of, tagged

SIGNALS = [
    candidate("intraday move", "alpha = close / open"),
    candidate("volume shock", "alpha = volume / WindowedAvg(volume, 20)"),
]
_FIELDS = ("open", "high", "low", "close", "volume")
_LAGS = (5, 10, 20, 40, 60, 120, 250)
_WINDOWS = (5, 10, 20, 40, 60)
#: Noise: every bar either side reads is five or more bars old, which no planted shock reaches (a
#: moving average over the latest bars would carry today's shocks, a signal in one direction or the other).
NOISE_CODES = [
    f"alpha = BackRef({field}, {lag}) / BackRef(WindowedAvg({field}, {window}), 5)"
    for window in _WINDOWS for lag in _LAGS for field in _FIELDS
]


def noise(number: int, slot: int) -> dict:
    return candidate(f"noise of round {number} slot {slot}", NOISE_CODES[(number - 1) * 12 + slot])


def liquidity_wins(number: int, directions: list[dict], seen: str = "") -> list[dict]:
    """Planted signals in the liquidity slots while there are fresh ones, noise elsewhere."""
    fresh = [s for s in SIGNALS if s["description"] not in seen]
    items = []
    for slot, direction in enumerate(directions):
        if direction["mechanism_type"] == "liquidity" and fresh:
            items.append(fresh.pop(0))
        else:
            items.append(noise(number, slot))
    return items


def proposer(fake_holder: list) -> callable:
    """``liquidity_wins`` that skips the signals already accepted, as the prompt's best mechanisms say."""
    def propose(number, directions):
        prompt = fake_holder[0].of_role("proposer")[-1].prompt
        best = prompt.split("Current best mechanisms", 1)[-1].split("</best>", 1)[0]
        return liquidity_wins(number, directions, best)
    return propose


def make_fake(**options) -> FakeLLM:
    holder = []
    fake = FakeLLM(proposer(holder), **options)
    holder.append(fake)
    return fake


@pytest.fixture(scope="module")
def market(tmp_path_factory):
    root = tmp_path_factory.mktemp("market")
    return {"components": synthetic_market.build(root / "data"), "cache": root / "kunquant_cache"}


def _campaign(tmp_path, market, fake, **round_settings):
    settings = {"candidates": 10, "min_per_type": 1, "retries": 0, "max_rounds": 3, **round_settings}
    path = write_config(tmp_path, market["components"], round=settings)
    return run(path, fake, market["cache"])


def _types_by_round(records) -> dict[int, collections.Counter]:
    counts = collections.defaultdict(collections.Counter)
    for record in records:
        counts[record.round][record.direction["mechanism_type"]] += 1
    return counts


@pytest.fixture(scope="module")
def five_rounds(tmp_path_factory, market):
    fake = make_fake()
    result = _campaign(tmp_path_factory.mktemp("five_rounds"), market, fake, max_rounds=5)
    return {"result": result, "fake": fake}


def test_every_round_holds_the_quota_of_every_mechanism_type(five_rounds):
    records = five_rounds["result"].records
    by_round = _types_by_round(records)
    assert sorted(by_round) == [1, 2, 3, 4, 5]
    for counts in by_round.values():
        assert sum(counts.values()) == 10
        assert all(counts[kind] >= 1 for kind in [m.name for m in DEFAULT_MECHANISMS])
    assert all(r.candidate["mechanism_type"] == r.direction["mechanism_type"] for r in records)


def test_the_extra_slots_shift_toward_the_type_that_keeps_being_accepted(five_rounds):
    # Round 1 accepts both planted signals, in liquidity slots; noise fills every other slot.
    result = five_rounds["result"]
    assert [(r.round, r.candidate["description"]) for r in result.accepted] == [
        (1, "intraday move"), (1, "volume shock"),
    ]
    assert {r.direction["mechanism_type"] for r in result.accepted} == {"liquidity"}
    by_round = _types_by_round(result.records)
    extra = collections.Counter()
    for n in (2, 3, 4, 5):
        extra.update({kind: by_round[n][kind] - 1 for kind in [m.name for m in DEFAULT_MECHANISMS]})
    assert extra.most_common(1)[0][0] == "liquidity", extra
    assert extra["liquidity"] >= 8, extra  # of 20 extra slots, against 4 for a uniform draw
    assert by_round[2]["liquidity"] - 1 >= 3, by_round[2]


def test_the_planner_is_asked_for_the_quota_it_is_held_to(five_rounds):
    fake, records = five_rounds["fake"], five_rounds["result"].records
    by_round = _types_by_round(records)
    for request in fake.of_role("planner"):
        quota = tagged(request.prompt, "quota")
        assert collections.Counter(quota) == by_round[round_of(request.prompt)]


def only_noise(number, directions):
    return [noise(number, slot) for slot in range(len(directions))]


REWRITTEN = "Rewritten: the move from the open to the close persists."
REVIEW_REASON = "the reviewer found it inconsistent: the code does not compute what the description says"


def _ledger(result) -> list[dict]:
    return [json.loads(line) for line in result.ledger_path.read_text().splitlines()]


def _fix_requests(fake: FakeLLM) -> list:
    return [r for r in fake.of_role("proposer") if has_tag(r.prompt, "refused_candidates")]


def test_a_candidate_still_inconsistent_after_its_rewrite_is_rejected_without_evaluation(tmp_path, market):
    fake = make_fake(consistent=lambda c: c["description"] != "intraday move")  # the default fix changes nothing
    result = _campaign(tmp_path, market, fake, max_rounds=1)
    record = next(r for r in result.records if r.candidate["description"] == "intraday move")
    assert (record.status, record.stage) == ("rejected", "review")
    assert record.metrics is None and record.fingerprint
    assert record.reasons == [REVIEW_REASON]
    assert [a["stage"] for a in record.attempts] == ["review"]
    assert result.evaluated_trials == 9
    assert [r.candidate["description"] for r in result.accepted] == ["volume shock"]
    first, second = fake.of_role("reviewer")
    assert first.model == second.model == "model-reviewer"
    assert "intraday move" in first.prompt and "alpha = close / open" in first.prompt
    assert [c["description"] for c in tagged(second.prompt, "candidates")] == ["intraday move"]
    assert len(_fix_requests(fake)) == 1


def test_an_inconsistent_candidate_rewritten_to_agree_is_reviewed_again_and_evaluated(tmp_path, market):
    def fix(number, failures):
        return [{**f["refused_version"], "hypothesis": REWRITTEN} for f in failures]

    fake = make_fake(consistent=lambda c: c["description"] != "intraday move" or c["hypothesis"] == REWRITTEN, fix=fix)
    result = _campaign(tmp_path, market, fake, max_rounds=1)
    [record] = [r for r in _ledger(result) if r["candidate"]["description"] == "intraday move"]
    assert (record["status"], record["stage"]) == ("accepted", "admission")
    assert record["candidate"]["hypothesis"] == REWRITTEN and record["metrics"] is not None
    [attempt] = record["attempts"]
    assert attempt["stage"] == "review" and attempt["reasons"] == [REVIEW_REASON]
    assert attempt["item"]["hypothesis"] == "Hypothesis behind intraday move."
    assert result.evaluated_trials == 10
    [fix_request] = _fix_requests(fake)
    [entry] = tagged(fix_request.prompt, "refused_candidates")
    assert entry["refused_because"] == [REVIEW_REASON]
    first, second = fake.of_role("reviewer")
    assert len(tagged(first.prompt, "candidates")) == 10
    assert [c["hypothesis"] for c in tagged(second.prompt, "candidates")] == [REWRITTEN]


def test_a_rewrite_the_gate_refuses_fails_at_the_gate_without_further_retries(tmp_path, market):
    def fix(number, failures):
        return [
            {**f["refused_version"], "code": "alpha = close /"} if f["refused_because"] == [REVIEW_REASON]
            else f["refused_version"]
            for f in failures
        ]

    fake = make_fake(consistent=lambda c: c["description"] != "intraday move", fix=fix)
    result = _campaign(tmp_path, market, fake, max_rounds=1, retries=2)
    [record] = [r for r in _ledger(result) if r["candidate"]["description"] == "intraday move"]
    assert (record["status"], record["stage"]) == ("failed", "gate")
    assert record["candidate"]["code"] == "alpha = close /" and record["metrics"] is None
    assert [a["stage"] for a in record["attempts"]] == ["review", "gate"]
    assert len(_fix_requests(fake)) == 1
    assert len(fake.of_role("reviewer")) == 1
    assert result.evaluated_trials == 9


_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")


def _assert_nothing_leaks(fake: FakeLLM, records) -> None:
    """No request holds a date after the training segment, a held-out metric or a return figure."""
    end = synthetic_market.TRAINING[1]
    returns = [x for r in records if r.evaluated for x in r.metrics["decile_returns"]]
    assert len(returns) > 40
    for request in fake.requests:
        prompt = request.prompt
        assert all("-".join(date) <= end for date in _DATE.findall(prompt))
        assert all(int(year) <= int(end[:4]) for year in _YEAR.findall(prompt))
        assert not any(word in prompt.lower() for word in ("validation", "hold-out", "holdout"))
        # Six decimals: at four, a decile return can equal an IC the feedback shows.
        assert not any(f"{x:.6f}" in prompt or repr(x) in prompt for x in returns)


def test_no_request_holds_a_later_date_a_held_out_metric_or_a_return(five_rounds):
    assert {r.role for r in five_rounds["fake"].requests} == {"planner", "proposer", "reviewer"}
    _assert_nothing_leaks(five_rounds["fake"], five_rounds["result"].records)


def test_no_distiller_request_holds_a_later_date_or_a_return(eleven_rounds):
    assert eleven_rounds["fake"].of_role("distiller")
    _assert_nothing_leaks(eleven_rounds["fake"], eleven_rounds["result"].records)


def test_the_feedback_holds_the_training_statistics_and_the_accepted_pool(five_rounds):
    fake, records = five_rounds["fake"], five_rounds["result"].records
    round_two = next(r.prompt for r in fake.of_role("planner") if round_of(r.prompt) == 2)
    noisy = next(r for r in records if r.round == 1 and r.status == "rejected")
    assert f"training IC {noisy.metrics['ic_mean']:+.4f} (Newey-West t {noisy.metrics['ic_t']:+.2f})" in round_two
    assert f"residual IC {noisy.metrics['partial_ic_mean']:+.4f}" in round_two
    assert noisy.reasons[0] in round_two
    best = round_two.split("Current best mechanisms", 1)[1]
    assert "volume shock (accepted in round 1)" in best and "alpha = close / open" in best
    weights = re.findall(r"marginal weight ([-+]\d\.\d{3}), residual IC ([-+]\d\.\d{4})", best)
    assert len(weights) == 2 and all(float(w) > 0 and float(ic) > 0 for w, ic in weights)
    first = fake.of_role("planner")[0].prompt
    assert "training IC" not in first and "Recent history" not in first and "Current best" not in first


@pytest.fixture(scope="module")
def eleven_rounds(tmp_path_factory, market):
    fake = FakeLLM(only_noise)
    root = tmp_path_factory.mktemp("eleven_rounds")
    result = _campaign(root, market, fake, candidates=5, max_rounds=11, patience=20)
    return {"result": result, "fake": fake, "run_dir": root / "run"}


def test_the_memory_is_distilled_every_five_rounds_and_carried_into_later_prompts(eleven_rounds):
    fake, result = eleven_rounds["fake"], eleven_rounds["result"]
    assert result.rounds == 11
    first, second = fake.of_role("distiller")
    assert "- round 5," in first.prompt and "- round 6," not in first.prompt
    assert "- round 10," in second.prompt and "- round 11," not in second.prompt
    assert not re.search(r"\d\.\d", first.prompt)  # the distiller sees outcomes and reasons, no statistic
    for request in fake.of_role("planner") + fake.of_role("proposer"):
        number = round_of(request.prompt)
        assert ("Distilled memory number 1" in request.prompt) == (6 <= number <= 10), number
        assert ("Distilled memory number 2" in request.prompt) == (number == 11), number
    assert "Distilled memory number 2" in (eleven_rounds["run_dir"] / "memory.md").read_text()


@pytest.mark.parametrize(
    ("settings", "options", "by", "rounds"),
    [
        ({"max_rounds": 2}, {}, "max_rounds", 2),
        ({"max_rounds": 10, "max_trials": 12}, {}, "max_trials", 2),
        ({"max_rounds": 10, "patience": 2}, {}, "patience", 2),
        # Three calls a round at a dollar each: round 2 reaches the budget before its review and is dropped.
        ({"max_rounds": 10, "budget_usd": 5.0}, {"cost": 1.0}, "budget", 1),
    ],
)
def test_each_stop_condition_ends_the_run_and_is_recorded(tmp_path, market, settings, options, by, rounds):
    fake = FakeLLM(only_noise, **options)
    result = _campaign(tmp_path, market, fake, candidates=5, **settings)
    assert result.stop.by == by
    assert result.rounds == rounds
    assert {r.round for r in result.records} == set(range(1, rounds + 1))
    state = json.loads((tmp_path / "run" / "state.json").read_text())
    assert state["stop"] == {"by": by, "reason": result.stop.reason}
    assert state["rounds"] == rounds
    if by == "budget":
        assert result.spent == state["spent"] == 5.0 and "$5.00" in result.stop.reason
        assert [r.role for r in fake.requests] == ["planner", "proposer", "reviewer", "planner", "proposer"]


def test_an_interrupted_campaign_resumes_from_the_last_written_round(tmp_path_factory, market):
    settings = {"candidates": 7, "max_rounds": 4}
    straight = tmp_path_factory.mktemp("straight")
    _campaign(straight, market, make_fake(), **settings)

    resumed = tmp_path_factory.mktemp("resumed")
    crashing = make_fake(interrupt=lambda r: r.role == "reviewer" and round_of(r.prompt) == 3)
    with pytest.raises(Interrupted):
        _campaign(resumed, market, crashing, **settings)
    state = json.loads((resumed / "run" / "state.json").read_text())
    assert state["rounds"] == 2 and state["stop"] is None
    with (resumed / "run" / "ledger.jsonl").open("a") as ledger:
        ledger.write('{"round": 3, "candidate": {"descr')  # a line a crash cut short
    fresh = make_fake()
    result = _campaign(resumed, market, fresh, **settings)

    assert result.rounds == 4 and result.stop.by == "max_rounds"
    assert {round_of(r.prompt) for r in fresh.requests} == {3, 4}
    assert (resumed / "run" / "ledger.jsonl").read_text() == (straight / "run" / "ledger.jsonl").read_text()


SLOPE = candidate("price slope", "alpha = WindowedLinearRegressionSlope(close, 20)")


def slope_and_move(number, directions):
    """A factor in price units, one that is a pure number, and noise in the other slots."""
    return [SLOPE, SIGNALS[0], *(noise(number, slot) for slot in range(2, len(directions)))]


def test_the_reviewer_is_told_each_candidates_output_unit(tmp_path, market):
    fake = FakeLLM(slope_and_move)
    _campaign(tmp_path, market, fake, max_rounds=1)
    [review] = fake.of_role("reviewer")
    units = {c["description"]: c["output_unit"] for c in tagged(review.prompt, "candidates")}
    assert units["price slope"] == "price"
    assert units["intraday move"] == "pure number"


def test_a_candidate_sent_back_by_the_reviewer_is_listed_in_the_rewrite_request_with_its_output_unit(tmp_path, market):
    fake = FakeLLM(slope_and_move, consistent=lambda c: c["description"] != "price slope")
    _campaign(tmp_path, market, fake, max_rounds=1)
    [fix_request] = _fix_requests(fake)
    [entry] = tagged(fix_request.prompt, "refused_candidates")
    assert entry["refused_version"]["description"] == "price slope"
    assert entry["output_unit"] == "price"


def one_of_each_outcome(number, directions):
    """Round 1: accepted, refused at admission, inconsistent, unparsable, refused by the Gate; noise afterwards."""
    if number > 1:
        return only_noise(number, directions)
    return [
        SIGNALS[0],
        noise(number, 1),
        candidate("inconsistent move", "alpha = high / low"),
        candidate("unknown kind", "alpha = low / open", mechanism_type="astrology"),
        candidate("unknown operator", "alpha = NoSuchOperator(close)"),
    ]


@pytest.fixture(scope="module")
def two_recent_rounds(tmp_path_factory, market):
    fake = FakeLLM(one_of_each_outcome)
    root = tmp_path_factory.mktemp("two_recent_rounds")
    path = write_config(
        root, market["components"], context={"recent_rounds": 2},
        round={"candidates": 5, "min_per_type": 1, "retries": 0, "max_rounds": 4, "patience": 20, "memory_every": 1},
    )
    return {"result": run(path, fake, market["cache"]), "fake": fake, "run_dir": root / "run"}


def _section(prompt: str, tag: str) -> str:
    [body] = re.findall(rf"^<{tag}>\n(.*?)\n</{tag}>$", prompt, re.DOTALL | re.MULTILINE)
    return body


def test_the_planner_lists_the_recent_rounds_trial_by_trial_and_counts_the_earlier_ones(two_recent_rounds):
    fake, records = two_recent_rounds["fake"], two_recent_rounds["result"].records
    [round_one] = [(r.status, r.stage) for r in records if r.round == 1 and r.direction["mechanism_type"] == "liquidity"]
    assert round_one == ("rejected", "review")
    planner = next(r.prompt for r in fake.of_role("planner") if round_of(r.prompt) == 4)
    recent = _section(planner, "recent_history").splitlines()
    assert len(recent) == 10 and all(line.startswith(("- round 2,", "- round 3,")) for line in recent)
    assert "## Recent history" in planner and "## Earlier summary" in planner
    summary = _section(planner, "earlier_summary")
    assert summary.splitlines() == [
        "The trials of rounds 1 to 1, by mechanism type:",
        "- behavioural_bias: 1 trials, 1 evaluated, 1 accepted",
        "- risk_compensation: 1 trials, 1 evaluated, 0 accepted; refused: admission (partial IC) 1",
        "- liquidity: 1 trials, 0 evaluated, 0 accepted; refused: review inconsistent 1",
        "- microstructure: 1 trials, 0 evaluated, 0 accepted; refused: parse 1",
        "- information_diffusion: 1 trials, 0 evaluated, 0 accepted; refused: gate 1",
    ]
    assert "- round 1," not in planner
    proposer = next(r.prompt for r in fake.of_role("proposer") if round_of(r.prompt) == 4)
    assert not has_tag(proposer, "recent_history") and not has_tag(proposer, "earlier_summary")


def test_the_distiller_folds_the_trials_since_the_last_distillation_into_the_previous_memory(two_recent_rounds):
    first, second, third = (r.prompt for r in two_recent_rounds["fake"].of_role("distiller"))
    assert "Distilled memory number" not in first and "- round 1," in first
    trials = _section(second, "history").splitlines()
    assert len(trials) == 5 and all(line.startswith("- round 2,") for line in trials)
    assert "Distilled memory number 1" in _section(second, "previous_memory")
    assert "Distilled memory number 2" in _section(third, "previous_memory") and "- round 2," not in third


def test_every_memory_version_is_saved_and_memory_md_is_the_current_one(two_recent_rounds):
    run_dir = two_recent_rounds["run_dir"]
    versions = sorted(p.name for p in (run_dir / "memory").iterdir())
    assert versions == ["round_02.md", "round_03.md", "round_04.md"]
    assert "Distilled memory number 2" in (run_dir / "memory" / "round_03.md").read_text()
    assert (run_dir / "memory.md").read_text() == (run_dir / "memory" / "round_04.md").read_text()


def test_an_empty_or_unparsable_distiller_reply_keeps_the_previous_memory(tmp_path, market):
    from tests.fake_llm import memory
    replies = {1: memory(1), 2: "", 3: "Some thoughts, cut short by the token lim"}
    fake = FakeLLM(only_noise, distill=replies.get)
    result = _campaign(tmp_path, market, fake, candidates=5, max_rounds=4, patience=20, memory_every=1)
    assert result.memory == memory(1)
    run_dir = tmp_path / "run"
    assert (run_dir / "memory.md").read_text() == memory(1) + "\n"
    assert sorted(p.name for p in (run_dir / "memory").iterdir()) == ["round_02.md"]
    planner = next(r.prompt for r in fake.of_role("planner") if round_of(r.prompt) == 4)
    assert "Distilled memory number 1" in _section(planner, "memory")
    third = fake.of_role("distiller")[2].prompt
    assert "Distilled memory number 1" in _section(third, "previous_memory")
    assert {line.split(",")[0] for line in _section(third, "history").splitlines()} == {"- round 2", "- round 3"}


def test_a_memory_part_keeps_at_most_six_items(tmp_path, market):
    long = (
        "Working mechanism patterns\n" + "".join(f"- pattern {i}\n" for i in range(9))
        + "Failure constraints\n- rule\nDirectional findings\n- finding"
    )
    fake = FakeLLM(only_noise, distill=lambda call: long)
    result = _campaign(tmp_path, market, fake, candidates=5, max_rounds=2, memory_every=1)
    assert "- pattern 5" in result.memory and "- pattern 6" not in result.memory
    assert result.memory.endswith("Failure constraints\n- rule\nDirectional findings\n- finding")


MARKDOWN_MEMORY = (
    "## Working mechanism patterns:\n* Volume surprises: attention fades.\n"
    "**Failure constraints**\n• Write ratios: levels rank by price.\n"
    "### directional findings\n- Volume surges were followed by outperformance."
)
NUMBERED_MEMORY = (
    "Working mechanism patterns\n1. Volume surprises: attention fades.\n2) Gaps: they revert.\n"
    "Failure constraints\n1) Write ratios: levels rank by price.\n"
    "Directional findings\n1. Volume surges were followed by outperformance."
)
NORMALISED = {
    MARKDOWN_MEMORY: (
        "Working mechanism patterns\n- Volume surprises: attention fades.\n"
        "Failure constraints\n- Write ratios: levels rank by price.\n"
        "Directional findings\n- Volume surges were followed by outperformance."
    ),
    NUMBERED_MEMORY: (
        "Working mechanism patterns\n- Volume surprises: attention fades.\n- Gaps: they revert.\n"
        "Failure constraints\n- Write ratios: levels rank by price.\n"
        "Directional findings\n- Volume surges were followed by outperformance."
    ),
}


@pytest.mark.parametrize("reply", [MARKDOWN_MEMORY, NUMBERED_MEMORY], ids=["markdown", "numbered"])
def test_a_memory_with_markdown_headings_or_other_bullets_is_kept_with_plain_headings_and_dashes(tmp_path, market, reply):
    fake = FakeLLM(only_noise, distill=lambda call: reply)
    result = _campaign(tmp_path, market, fake, candidates=5, max_rounds=2, memory_every=1)
    assert result.memory == NORMALISED[reply]


def test_a_memory_reply_with_the_headings_but_no_item_keeps_the_previous_memory_and_says_so(tmp_path, market, capsys):
    from tests.fake_llm import memory
    bare = "Working mechanism patterns\nFailure constraints\nDirectional findings"
    fake = FakeLLM(only_noise, distill={1: memory(1), 2: bare}.get)
    result = _campaign(tmp_path, market, fake, candidates=5, max_rounds=3, memory_every=1)
    assert result.memory == memory(1)
    out = capsys.readouterr().out
    assert "round 3" in out and "previous memory is kept" in out
