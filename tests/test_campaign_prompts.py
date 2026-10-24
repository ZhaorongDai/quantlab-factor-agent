"""The prompts through Campaign.run(): what the config says about the market, the label and the research.

Every role is answered by ``tests/fake_llm.FakeLLM``, which keeps every
request; the Gate and the Evaluator are the real ones on the synthetic
store. A test reads what each role was asked.
"""

import re
from pathlib import Path

import pytest
from quantlab.core.component import rebuild

from factor_agent.config import CampaignConfig
from factor_agent.prompts import PromptError
from factor_agent.snapshot import SnapshotMismatch
from tests import synthetic_market
from tests.campaign_runner import SYNTHETIC_MARKET, candidate, run, write_config
from tests.fake_llm import FakeLLM, last_block

#: A market the synthetic store could be, described in no words the default prompts use.
MARKET = "daily bars of every company listed on the Lunar exchange"
#: A label horizon other than the synthetic store's five bars.
HORIZON = 10


def _propose(number, directions):
    items = [
        candidate("intraday move", "alpha = close / open"),
        candidate("syntax error", "alpha = close /"),
    ]
    items += [
        candidate(f"noise {slot}", f"alpha = BackRef(volume, {(5, 10, 20, 40, 60)[slot]}) / WindowedAvg(volume, 20)")
        for slot in range(len(directions))
    ]
    return items[:len(directions)]


@pytest.fixture(scope="module")
def market(tmp_path_factory):
    root = tmp_path_factory.mktemp("market")
    components = synthetic_market.build(root / "data")
    label = dict(components["market_label"])
    label.update(
        kwargs={"n_forward_periods": HORIZON}, factor_names=[f"ret_{HORIZON}"], warmup_bars=2 * HORIZON + 5,
        file_path=str(root / "data" / "labels" / "market" / "ret_10" / "ret_10.zarr"),
    )
    rebuilt = rebuild(label)
    rebuilt.build(synthetic_market.START, synthetic_market.END)
    return {"components": components, "label_10": rebuilt.get_config(), "cache": root / "kunquant_cache"}


@pytest.fixture(scope="module")
def lunar(tmp_path_factory, market):
    """Two rounds on the Lunar market and the 10-bar label, with a distillation and a fix request."""
    root = tmp_path_factory.mktemp("lunar")
    path = write_config(
        root, {**market["components"], "label": market["label_10"]}, market=MARKET,
        round={"candidates": 5, "min_per_type": 1, "retries": 1, "max_rounds": 2, "memory_every": 1},
    )
    fake = FakeLLM(_propose)
    run(path, fake, market["cache"])
    return fake


def test_every_role_is_asked(lunar):
    assert {request.role for request in lunar.requests} == {"planner", "proposer", "reviewer", "distiller"}


def test_every_prompt_describes_the_configured_market_and_the_labels_horizon(lunar):
    for request in lunar.requests:
        assert MARKET in request.prompt, request.role
        assert f"{HORIZON} bars" in request.prompt, request.role


def test_no_prompt_mentions_us_stocks_or_five_days(lunar):
    for request in lunar.requests:
        text = request.prompt
        assert not re.search(r"\bUS\b|\bstocks?\b|\bfive\b|\b5 (?:trading )?days\b", text, re.IGNORECASE), (
            request.role, re.findall(r".{40}(?:\bUS\b|\bstocks?\b|\bfive\b).{40}", text, re.IGNORECASE),
        )


def test_no_prompt_asks_for_a_literature_reference(lunar):
    for request in lunar.requests:
        assert "literature" not in request.prompt.lower(), request.role
        assert '"reference"' not in request.prompt, request.role


#: A taxonomy of another research area.
THREE = [
    {"name": "value", "description": "cheap firms earn more than dear ones"},
    {"name": "momentum", "description": "recent winners keep winning"},
    {"name": "carry", "description": "the yield of holding the asset"},
]


def _typed(number, directions):
    """One item per direction with its type, except the last, which names a type the config does not have."""
    items = [
        candidate(f"slot {slot}", f"alpha = BackRef(volume, {(5, 10, 20, 40, 60)[slot]}) / WindowedAvg(volume, 20)",
                  mechanism_type=direction["mechanism_type"])
        for slot, direction in enumerate(directions)
    ]
    items[-1] = {**items[-1], "mechanism_type": "seasonality"}
    return items


@pytest.fixture(scope="module")
def three_types(tmp_path_factory, market):
    root = tmp_path_factory.mktemp("three_types")
    path = write_config(
        root, market["components"], mechanisms=THREE,
        round={"candidates": 4, "min_per_type": 1, "retries": 0, "max_rounds": 1},
    )
    fake = FakeLLM(_typed)
    result = run(path, fake, market["cache"])
    return {"fake": fake, "result": result}


def test_a_round_is_planned_over_the_configured_mechanism_types(three_types):
    [plan] = three_types["fake"].of_role("planner")
    quota = last_block(plan.prompt)
    assert list(quota) == ["value", "momentum", "carry"] and sum(quota.values()) == 4
    for kind in THREE:
        assert f"- {kind['name']}: {kind['description']}" in plan.prompt
    assert {r.direction["mechanism_type"] for r in three_types["result"].records} == {"value", "momentum", "carry"}


def test_a_candidate_is_validated_against_the_configured_mechanism_types(three_types):
    records = three_types["result"].records
    assert all(r.stage != "parse" for r in records[:-1])
    last = records[-1]
    assert (last.status, last.stage) == ("failed", "parse")
    assert "'seasonality' is not one of ['value', 'momentum', 'carry']" in last.reasons[0]


def test_a_quota_the_configured_types_cannot_hold_fails_at_start(tmp_path, market):
    path = write_config(tmp_path, market["components"], mechanisms=THREE, round={"candidates": 5, "min_per_type": 2})
    with pytest.raises(ValueError, match="5 candidates cannot hold 2 of each of 3 mechanism types"):
        CampaignConfig.from_file(path)


REVIEW = """Judge each candidate on {market} below, horizon {horizon} bars. Answer with a JSON array of
objects with "number", "consistent" and "reason".
Round: {round}
```json
{candidates}
```
"""

DATA = "MARKET {market}; SCORING {scoring}; HORIZON {horizon} bars; INPUTS\n{inputs}"


def _replaced(root: Path, market, **templates) -> Path:
    """A one-round campaign whose ``templates`` (name to text) replace the built-in ones from files."""
    (root / "prompts").mkdir(parents=True, exist_ok=True)
    files = {}
    for name, text in templates.items():
        files[name] = root / "prompts" / f"{name}.txt"
        files[name].write_text(text)
    return write_config(
        root, market["components"], prompts={name: str(path) for name, path in files.items()},
        round={"candidates": 5, "min_per_type": 1, "retries": 0, "max_rounds": 1},
    )


def test_replaced_templates_are_the_ones_the_roles_are_asked_with(tmp_path, market):
    fake = FakeLLM(_propose)
    run(_replaced(tmp_path, market, review=REVIEW, data=DATA), fake, market["cache"])
    [review] = fake.of_role("reviewer")
    assert review.prompt.startswith(f"Judge each candidate on {SYNTHETIC_MARKET} below, horizon 5 bars.")
    for request in fake.of_role("planner") + fake.of_role("proposer"):
        assert f"MARKET {SYNTHETIC_MARKET}; SCORING A factor is ranked" in request.prompt
        assert "HORIZON 5 bars; INPUTS\n- price_volume (adjusted daily bars):\n  - open (price): opening price" in request.prompt
        assert "The data:" not in request.prompt


def test_a_replaced_template_is_snapshotted(tmp_path, market):
    run(_replaced(tmp_path, market, review=REVIEW), FakeLLM(_propose), market["cache"])
    assert (tmp_path / "run" / "snapshot" / "prompts" / "review" / "review.txt").read_text() == REVIEW


@pytest.mark.parametrize("name, text, placeholder", [
    ("review", REVIEW.replace("{candidates}", ""), "{candidates}"),
    ("data", DATA.replace("{inputs}", ""), "{inputs}"),
    ("plan", "Plan round {round} for {market}.\n{data}\n{context}\n", "{quota}"),
], ids=["review", "data", "plan"])
def test_a_replaced_template_missing_a_placeholder_fails_at_start_naming_it(tmp_path, market, name, text, placeholder):
    fake = FakeLLM(_propose)
    with pytest.raises(PromptError, match=re.escape(f"lacks the placeholder {placeholder}")):
        run(_replaced(tmp_path, market, **{name: text}), fake, market["cache"])
    assert fake.requests == []
    assert not (tmp_path / "run" / "snapshot.json").exists()


def test_a_replaced_template_using_an_unknown_placeholder_fails_at_start_naming_it(tmp_path, market):
    with pytest.raises(PromptError, match=re.escape("unknown placeholder(s) ['{reviewer_mood}']")):
        run(_replaced(tmp_path, market, review=REVIEW + "{reviewer_mood}"), FakeLLM(_propose), market["cache"])


def test_a_replaced_data_template_may_not_use_the_round(tmp_path, market):
    # The data section is the same in every round's prompts, so it is not filled with the round.
    with pytest.raises(PromptError, match=re.escape("data (") + ".*" + re.escape("unknown placeholder(s) ['{round}']")):
        run(_replaced(tmp_path, market, data=DATA + "\nRound {round}"), FakeLLM(_propose), market["cache"])


def test_an_unknown_template_name_fails_at_start(tmp_path, market):
    with pytest.raises(PromptError, match="judge: not a template"):
        run(_replaced(tmp_path, market, judge=REVIEW), FakeLLM(_propose), market["cache"])


def test_a_resume_after_a_replaced_template_changed_fails(tmp_path, market):
    path = _replaced(tmp_path, market, review=REVIEW)
    run(path, FakeLLM(_propose), market["cache"])
    (tmp_path / "prompts" / "review.txt").write_text(REVIEW + "Be strict.\n")
    fake = FakeLLM(_propose)
    with pytest.raises(SnapshotMismatch, match=r"prompts/review/review.txt changed.*new run directory"):
        run(path, fake, market["cache"])
    assert fake.requests == []


@pytest.fixture(scope="module")
def defaults(tmp_path_factory, market):
    """One round with no market, mechanism or template settings beyond the defaults."""
    root = tmp_path_factory.mktemp("defaults")
    path = write_config(
        root, market["components"], directions=["behavioural_bias: a seed direction"],
        round={"candidates": 5, "min_per_type": 1, "retries": 0, "max_rounds": 1},
    )
    fake = FakeLLM(_propose)
    run(path, fake, market["cache"])
    return fake


def _in_order(text: str, markers: list[str]) -> bool:
    positions = [text.find(marker) for marker in markers]
    return -1 not in positions and positions == sorted(positions)


def test_with_the_defaults_the_planner_prompt_carries_todays_sections(defaults):
    [plan] = defaults.of_role("planner")
    assert _in_order(plan.prompt, [
        "You are the research planner", "Round: 1", "The data: " + SYNTHETIC_MARKET,
        "A factor is ranked across the whole market each day, neutralised", "the next 5 bars",
        "The inputs, by family:", "## Seed research directions", "The mechanism types:",
        "- information_diffusion: ", "Plan exactly the number of directions", "```json",
    ]), plan.prompt


def test_with_the_defaults_the_proposer_prompt_carries_todays_sections(defaults):
    [propose] = defaults.of_role("proposer")
    assert _in_order(propose.prompt, [
        "You are a quantitative researcher", "the next 5 bars", "Round: 1", "The data: " + SYNTHETIC_MARKET,
        "The inputs, by family:", "Write each factor as Python code", "Inputs: open, high, low, close, volume",
        "Operators:", "Parameter values", "Builtins:", '"horizon_rationale": why the effect plays out over about 5 bars',
        "The directions:", "```json",
    ]), propose.prompt
