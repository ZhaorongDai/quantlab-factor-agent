"""The proposer's prompts with an intraday family open: the two stages, the operators by stage, the examples.

The prompts are rendered from the built-in templates on the vocabulary of
the price-volume and minute-bar field files and the example operator file;
the examples they hold go through a real Gate, sandbox included.
"""

import re

import pytest

from factor_agent.candidate import Candidate
from factor_agent.config import GateConfig
from factor_agent.gate import Gate, Passed
from factor_agent.prompts import Prompts, Templates
from factor_agent.zoo import Zoo
from tests.vocabulary_files import intraday, price_volume


def _prompts(vocabulary) -> Prompts:
    return Prompts(Templates({}), "daily bars of a market", 5, "A factor is ranked.", vocabulary, {"momentum": "m"})


def _propose(vocabulary) -> str:
    return _prompts(vocabulary).render("propose", round=1, context="", directions="[]", n=1)


def _retry(vocabulary) -> str:
    return _prompts(vocabulary).render("retry", round=1, failures="[]", n=1)


def _between(text: str, start: str, end: str) -> str:
    return text[text.index(start):text.index(end, text.index(start))]


@pytest.mark.parametrize("render", [_propose, _retry], ids=["propose", "retry"])
def test_with_an_intraday_family_the_proposer_is_told_the_two_stages_and_their_rules(render):
    prompt = render(intraday())
    section = _between(prompt, "Intraday candidates", "\n<vocabulary>\n")
    assert "minute_bars" in section
    for rule in (
        "an intraday field reaches the daily stage only through a session aggregation",
        "the intraday stage reads no daily field",
        "no aggregation of a daily value or of another aggregation",
        "exponential averages",
        "missing on a session's first bars until it fits",
    ):
        assert rule in section, rule
    assert "5, 10, 15, 30, 60, 120" in section


def test_the_operators_are_listed_by_the_stages_they_may_be_used_in():
    vocabulary = _between(_propose(intraday()), "\n<vocabulary>\n", "\n</vocabulary>\n")
    both = _between(vocabulary, "In both stages:", "In the daily stage only:")
    daily = _between(vocabulary, "In the daily stage only:", "Session aggregations")
    aggregations = _between(vocabulary, "Session aggregations", "Parameter values")
    assert "- Add(x, y)" in both and "- WindowedAvg(x, window)" in both and "- Rank(x)" in both
    assert "- ExpMovingAvg(x, window)" in daily and "- CMRA(" in daily
    assert "ExpMovingAvg" not in both and "SessionMean" not in both + daily
    for name in ("SessionMean(x)", "SessionStd(x)", "SessionWeightedMean(x, w)", "SessionCorr(x, y)"):
        assert f"- {name}" in aggregations, name
    assert "- intraday window" in _between(vocabulary, "Parameter values", "Builtins")


def _examples(prompt: str) -> list[str]:
    """The code of each example candidate in the intraday section: its indented lines after "Example"."""
    section = _between(prompt, "Intraday candidates", "\n<vocabulary>\n")
    return [
        "\n".join(line[4:] for line in block.splitlines() if line.startswith("    "))
        for block in re.split(r"\nExample \d+:", section)[1:]
    ]


def test_the_examples_in_the_prompt_pass_the_gate():
    examples = _examples(_propose(intraday()))
    assert len(examples) == 2
    gate = Gate(GateConfig(), intraday())
    for code in examples:
        assert code.rstrip().splitlines()[-1].startswith("alpha = ")
        candidate = Candidate(
            hypothesis="Intraday trading carries information.", mechanism_type="momentum",
            horizon_rationale="A week.", description="An example.", code=code,
        )
        result = gate.check(candidate, Zoo())
        assert isinstance(result, Passed), (code, getattr(result, "reasons", None))


def test_the_intraday_family_is_marked_in_the_data():
    data = _between(_propose(intraday()), "<data>", "</data>")
    assert "- minute_bars, the intraday family (one-minute trade bars of each regular session):" in data
    assert "- price_volume (adjusted daily bars):" in data


@pytest.mark.parametrize("render", [_propose, _retry], ids=["propose", "retry"])
def test_without_an_intraday_family_the_prompt_says_nothing_of_stages(render):
    prompt = render(price_volume())
    assert "intraday" not in prompt.lower() and "stage" not in prompt and "Session" not in prompt
