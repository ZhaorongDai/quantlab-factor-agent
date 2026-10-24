"""The prompts through Campaign.run(): what the config says about the market, the label and the research.

Every role is answered by ``tests/fake_llm.FakeLLM``, which keeps every
request; the Gate and the Evaluator are the real ones on the synthetic
store. A test reads what each role was asked.
"""

import json
import re
from pathlib import Path

import pytest
from quantlab.core.component import rebuild

from factor_agent.config import CampaignConfig
from factor_agent.prompts import TEMPLATES, PromptError, Templates
from factor_agent.snapshot import SnapshotMismatch
from tests import synthetic_market
from tests.campaign_runner import SYNTHETIC_MARKET, candidate, run, write_config
from tests.fake_llm import FakeLLM, tagged

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


#: The XML tag that holds what each kind of request asks about.
INPUT_TAG = {"planner": "quota", "proposer": "directions", "fix": "refused_candidates", "reviewer": "candidates"}
_JSON_FENCE = re.compile(r"```json\n(.*?)```", re.DOTALL)


def _kind(request) -> str:
    """The kind of a request: its role, or "fix" for the proposer's fix request."""
    return "fix" if request.role == "proposer" and "\n<refused_candidates>\n" in request.prompt else request.role


def test_the_campaign_sends_every_kind_of_request(lunar):
    assert {_kind(request) for request in lunar.requests} == {*INPUT_TAG, "distiller"}


@pytest.mark.parametrize("kind", list(INPUT_TAG))
def test_a_json_role_prompt_ends_with_its_output_skeleton_in_the_only_json_fence(lunar, kind):
    for request in (r for r in lunar.requests if _kind(r) == kind):
        prompt = request.prompt.rstrip()
        [skeleton] = _JSON_FENCE.findall(prompt)
        assert prompt.endswith("```json\n" + skeleton + "```"), prompt[-400:]
        assert prompt.index(f"</{INPUT_TAG[kind]}>") < prompt.index("```json")


@pytest.mark.parametrize("kind", list(INPUT_TAG))
def test_a_role_reads_its_input_from_its_tag_not_from_a_json_fence(lunar, kind):
    for request in (r for r in lunar.requests if _kind(r) == kind):
        given = tagged(request.prompt, INPUT_TAG[kind])
        [skeleton] = _JSON_FENCE.findall(request.prompt)
        values = given.values() if isinstance(given, dict) else given
        for value in values:
            shown = value if isinstance(value, str) else json.dumps(value)
            assert shown not in skeleton


@pytest.mark.parametrize("kind", [*INPUT_TAG, "distiller"])
def test_a_role_prompt_opens_with_the_role_and_who_uses_its_output(lunar, kind):
    request = next(r for r in lunar.requests if _kind(r) == kind)
    assert request.prompt.startswith("You are "), request.prompt[:200]


def test_the_fix_request_asks_for_each_corrected_candidate_itself(lunar):
    fix = next(r for r in lunar.requests if _kind(r) == "fix").prompt
    entries = tagged(fix, "refused_candidates")
    assert all("refused_version" in entry and "candidate" not in entry for entry in entries)
    ending = fix[fix.index("</refused_candidates>"):]
    assert "corrected candidate object itself" in ending
    assert 'Leave out "number", "direction", "refused_because" and "how_to_write_it"' in ending
    [skeleton] = _JSON_FENCE.findall(fix)
    assert all(f'"{field}"' in skeleton for field in candidate("x", "y"))
    assert "refused_version" not in skeleton and '"direction"' not in skeleton


def test_the_reviewer_is_asked_for_the_reason_before_the_verdict(lunar):
    review = next(r for r in lunar.requests if r.role == "reviewer").prompt
    [skeleton] = _JSON_FENCE.findall(review)
    assert skeleton.index('"number"') < skeleton.index('"reason"') < skeleton.index('"consistent"')
    assert "1. " in review and "2. " in review
    assert "length of the code" in review


def test_the_reviewer_reads_the_fields_and_judges_whether_the_code_uses_what_the_factor_is_built_from(lunar):
    review = next(r for r in lunar.requests if r.role == "reviewer").prompt
    data = review[review.index("<data>"):review.index("</data>")]
    assert "- volume (volume): shares traded in the day" in data
    assert review.index("</data>") < review.index("\n<candidates>\n")
    criteria = review[review.index("</candidates>"):review.index("Otherwise it is consistent")]
    assert "5. " in criteria and "built from" in criteria
    assert all(word in criteria for word in ("background", "consequence", "contrast"))
    assert "earnings surprise" in criteria and "price and volume" in criteria
    assert "gross profit" in criteria


def test_the_distiller_prompt_ends_with_the_three_parts_of_the_memory(lunar):
    distill = lunar.of_role("distiller")[0].prompt.rstrip()
    assert distill.endswith("```")
    ending = distill[distill.index("</history>"):]
    assert _in_order(ending, ["Working mechanism patterns", "Failure constraints", "Directional findings"])
    assert "at most 6" in ending


def test_the_built_in_templates_load_as_replaced_templates(tmp_path):
    files = {}
    for name, text in TEMPLATES.items():
        files[name] = tmp_path / f"{name}.txt"
        files[name].write_text(text)
    loaded = Templates.load(files)
    assert loaded.format("review", market="M", horizon=3, round=1, data="D", candidates="[]").rstrip().endswith("```")


def test_no_prompt_shouts(lunar):
    for request in lunar.requests:
        assert not re.search(r"\b(?:MUST|NEVER|ALWAYS|DO NOT|EXACTLY|IMPORTANT|ONLY)\b", request.prompt), request.role


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
    quota = tagged(plan.prompt, "quota")
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
<data>
{data}
</data>
<candidates>
{candidates}
</candidates>
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


def test_with_the_defaults_the_planner_prompt_carries_every_section_in_order(defaults):
    [plan] = defaults.of_role("planner")
    assert _in_order(plan.prompt, [
        "You are the research planner", "Round: 1", "\n<data>\n", "The data: " + SYNTHETIC_MARKET,
        "A factor is ranked across the whole market each day, neutralised", "the next 5 bars",
        "The inputs, by family:", "\n</data>\n",
        "## Seed research directions", "\n<seed>\n", "a seed direction", "\n</seed>\n",
        "The mechanism types:", "- information_diffusion: ",
        '"refinement"', "\n<quota>\n", "\n</quota>\n", "Now plan this round's research directions", "```json",
    ]), plan.prompt


def test_with_the_defaults_the_proposer_prompt_carries_every_section_in_order(defaults):
    [propose] = defaults.of_role("proposer")
    assert _in_order(propose.prompt, [
        "You are a quantitative researcher", "the next 5 bars", "Round: 1", "\n<data>\n", "The data: " + SYNTHETIC_MARKET,
        "The inputs, by family:", "\n</data>\n",
        "Write each factor as Python code", "\n<vocabulary>\n", "Inputs: open, high, low, close, volume",
        "Operators:", "Parameter values", "Builtins:", "\n</vocabulary>\n", "\n<directions>\n", "\n</directions>\n",
        "Now write one candidate", '"horizon_rationale": why the effect plays out over about 5 bars', "```json",
    ]), propose.prompt
