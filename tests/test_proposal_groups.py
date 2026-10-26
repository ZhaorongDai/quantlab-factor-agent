"""Proposal groups through Campaign.run(): a round's directions split into groups whose calls run side by side.

Every role is answered by ``tests/fake_llm.FakeLLM``, which answers from
each prompt's content, so its replies do not depend on how the round's
directions are split or in which order the calls land. The script plans
one numbered direction per slot and answers each direction by its text:
the planted signal in the first slot, a candidate the Gate refuses until
it is fixed, one the reviewer sends back to be rewritten, a near-duplicate
of the planted signal in the last slot, and noise everywhere else.
"""

import json

import pytest

from factor_agent.config import DEFAULT_MECHANISMS
from tests import synthetic_market
from tests.campaign_runner import candidate, run, write_config
from tests.fake_llm import FakeLLM, Interrupted, has_tag, round_of, tagged

SETTINGS = {"candidates": 10, "min_per_type": 1, "retries": 1, "max_rounds": 2, "memory_every": 1}
#: The groups of a round of 10 directions split four ways: contiguous runs of near-equal size.
SIZES = (3, 3, 2, 2)
REWRITTEN = "Rewritten."
_TYPES = [m.name for m in DEFAULT_MECHANISMS]


def _plan(number, quota, prompt):
    """One numbered direction per slot, so every direction of a round has its own text."""
    return [
        {"mechanism_type": kind, "action": "new", "parent": "", "direction": f"{kind} idea {number}.{i}"}
        for kind in _TYPES for i in range(quota[kind])
    ]


def _slot(direction: dict) -> tuple[int, int]:
    """The type's place and the direction's number within it, read from the direction's text."""
    kind, _, numbered = direction["direction"].rpartition(" idea ")
    return _TYPES.index(kind), int(numbered.split(".")[1])


def _noise(number: int, direction: dict) -> dict:
    kind, i = _slot(direction)
    lag, window = (5, 10, 20, 40, 60, 120, 250)[i % 7], (5, 10, 20, 40, 60)[kind]
    field = ("open", "high", "low", "close", "volume")[(number + i) % 5]
    return candidate(
        f"noise {direction['direction']}", f"alpha = BackRef({field}, {lag}) / BackRef(WindowedAvg({field}, {window}), 5)",
    )


def _propose(number, directions):
    items = []
    for direction in directions:
        kind, i = _slot(direction)
        if number == 1 and (kind, i) == (0, 0):
            items.append(candidate("intraday move", "alpha = close / open"))
        elif number == 1 and (kind, i) == (len(_TYPES) - 1, 0):
            items.append(candidate("intraday move again", "alpha = close / open"))  # a duplicate of slot 1
        elif number == 1 and kind == 1 and i == 0:
            items.append({**_noise(number, direction), "description": "broken noise", "code": "alpha = close /"})
        elif number == 1 and kind == 2 and i == 0:
            items.append({**_noise(number, direction), "description": "inconsistent noise"})
        else:
            items.append(_noise(number, direction))
    return items


def _fix(number, failures):
    """Rewrite the hypothesis of a candidate the reviewer sent back, mend the broken one, keep any other."""
    fixed = []
    for failure in failures:
        version = failure["refused_version"]
        if failure["refused_because"][0].startswith("the reviewer"):
            fixed.append({**version, "hypothesis": REWRITTEN})
        elif version["description"] == "broken noise":
            fixed.append({**version, "code": "alpha = BackRef(high, 60) / BackRef(WindowedAvg(low, 40), 5)"})
        else:
            fixed.append(version)
    return fixed


def _consistent(item) -> bool:
    return "inconsistent" not in item["description"] or item["hypothesis"] == REWRITTEN


def _fake(**options) -> FakeLLM:
    return FakeLLM(_propose, fix=_fix, consistent=_consistent, plan=_plan, **options)


@pytest.fixture(scope="module")
def market(tmp_path_factory):
    root = tmp_path_factory.mktemp("market")
    return {"components": synthetic_market.build(root / "data"), "cache": root / "kunquant_cache"}


def _campaign(root, market, fake, **settings):
    return run(write_config(root, market["components"], round={**SETTINGS, **settings}), fake, market["cache"])


@pytest.fixture(scope="module")
def grouped(tmp_path_factory, market):
    root = tmp_path_factory.mktemp("grouped")
    fake = _fake()
    result = _campaign(root, market, fake, groups=4)
    return {"root": root, "result": result, "fake": fake}


def _chunks(directions: list[dict]) -> list[list[dict]]:
    """The round's directions cut into the groups of ``SIZES``."""
    starts = [sum(SIZES[:g]) for g in range(len(SIZES) + 1)]
    return [directions[a:b] for a, b in zip(starts, starts[1:])]


def _directions_by_round(result) -> dict[int, list[dict]]:
    by_round = {}
    for record in result.records:
        by_round.setdefault(record.round, []).append(record.direction)
    return by_round


def _plain(direction: dict) -> dict:
    return {key: direction[key] for key in ("mechanism_type", "action", "parent", "direction")}


def test_four_groups_and_one_group_give_the_same_ledger(tmp_path, grouped, market):
    result = grouped["result"]
    by_status = {r.candidate.get("description"): (r.status, r.stage) for r in result.records if r.round == 1}
    assert by_status["intraday move"] == ("accepted", "admission")
    assert by_status["intraday move again"] == ("failed", "gate")  # a duplicate written in another group
    assert by_status["broken noise"][1] != "gate" and by_status["inconsistent noise"][1] != "review"
    _campaign(tmp_path, market, _fake(), groups=1)
    assert (tmp_path / "run" / "ledger.jsonl").read_text() == (grouped["root"] / "run" / "ledger.jsonl").read_text()


def test_each_grouped_request_carries_only_its_groups_directions_or_candidates(grouped):
    fake, by_round = grouped["fake"], _directions_by_round(grouped["result"])
    for number, directions in by_round.items():
        chunks = _chunks(directions)
        descriptions = [
            {r.candidate.get("description") for r in grouped["result"].records if r.direction in chunk and r.round == number}
            | {a["item"].get("description") for r in grouped["result"].records if r.direction in chunk and r.round == number
               for a in r.attempts if isinstance(a["item"], dict)}
            for chunk in chunks
        ]
        requests = [r for r in fake.requests if round_of(r.prompt) == number]
        proposals = [r for r in requests if r.role == "proposer" and has_tag(r.prompt, "directions")]
        asked = [[_plain(d) for d in tagged(r.prompt, "directions")] for r in proposals]
        assert sorted(asked, key=lambda c: directions.index(c[0])) == chunks
        fixes = [r for r in requests if r.role == "proposer" and has_tag(r.prompt, "refused_candidates")]
        if number == 1:
            assert fixes
        for request in fixes:
            asked = [f["direction"] for f in tagged(request.prompt, "refused_candidates")]
            assert any(all(d in chunk for d in asked) for chunk in chunks), asked
        reviews = [r for r in requests if r.role == "reviewer"]
        assert len(reviews) >= len(SIZES)
        for request in reviews:
            asked = {c["description"] for c in tagged(request.prompt, "candidates")}
            assert any(asked <= group for group in descriptions), asked
        assert len([r for r in requests if r.role == "planner"]) == 1
        assert len([r for r in requests if r.role == "distiller"]) == (1 if number > 1 else 0)


def test_a_failing_group_call_lets_the_stages_other_calls_land_in_the_log_and_raises(tmp_path, market):
    def failing(request):
        return request.role == "proposer" and has_tag(request.prompt, "directions") and any(
            _slot(d) == (2, 0) for d in tagged(request.prompt, "directions")
        )

    with pytest.raises(Interrupted):
        _campaign(tmp_path, market, _fake(interrupt=failing), groups=4)
    calls = [json.loads(p.read_text())["request"] for p in sorted((tmp_path / "run" / "llm_log").glob("*.json"))]
    assert [c["role"] for c in calls] == ["planner", "proposer", "proposer", "proposer"]
    logged = [_slot(d) for c in calls[1:] for d in tagged(c["prompt"], "directions")]
    assert (2, 0) not in logged and len(logged) >= 10 - max(SIZES)


def test_the_number_of_groups_may_change_on_resume(tmp_path, market):
    first = _campaign(tmp_path, market, _fake(), groups=1, max_rounds=1)
    assert first.rounds == 1
    fake = _fake()
    resumed = _campaign(tmp_path, market, fake, groups=4)
    assert resumed.rounds == 2
    proposals = [r for r in fake.of_role("proposer") if has_tag(r.prompt, "directions")]
    assert [round_of(r.prompt) for r in proposals] == [2] * 4
