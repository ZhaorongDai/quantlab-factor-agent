"""The migrated us3000 config asks the first round as the campaign before spec #10 did, section for section.

``tests/prompts_2bd6904/`` holds the first-round planner and proposer
prompts of commit 2bd6904 (the running campaign's prompts, all their wording
in code), written by that commit's ``Researcher.plan`` and ``propose`` with
a backend that records the prompt and stops, for the context, quota and
directions below (``generate.py`` there, run on a checkout of 2bd6904). The same calls are made here with the prompts the migrated
``configs/us3000_ret5.yaml`` gives. The scoring's sentence is built from the
config's scoring section without its stores, which are on the server only.

A section is a block between blank lines, named by the first two words of its
first line. Every section of the old prompts is in the new ones, in the same
order; only the ones whose wording now comes from the config (the market,
the label's horizon, the scoring, the field and operator files) differ, and
the only new sections are the inputs by family and the mechanism types.
"""

import re
from pathlib import Path

import pytest

from factor_agent.config import CampaignConfig
from factor_agent.prompts import Prompts
from factor_agent.researcher import Direction, Researcher
from factor_agent.scoring import Neutralization, Scoring
from tests.test_config import REAL

OLD = Path(__file__).parent / "prompts_2bd6904"
CONTEXT = {"seed": "SEED", "families": "FAMILIES"}
QUOTA = {"behavioural_bias": 4, "risk_compensation": 4, "liquidity": 4, "microstructure": 4, "information_diffusion": 4}
DIRECTIONS = [Direction("liquidity", "new", "", "DIRECTION A"), Direction("microstructure", "new", "", "DIRECTION B")]
#: Sections whose wording comes from the config now.
CONFIGURED = {"You are", "The data", "Inputs", "Answer with"}
#: Sections the config adds: the data template's inputs by family, the plan's mechanism types.
ADDED = {"plan": ["The inputs", "The mechanism"], "propose": ["The inputs"]}


class _Stop(Exception):
    pass


class _Recorder:
    """A backend of every role that keeps the prompt and stops the call."""

    def __init__(self):
        self.prompts = {}

    def complete(self, request):
        self.prompts[request.role] = request.prompt
        raise _Stop


@pytest.fixture(scope="module")
def config():
    return CampaignConfig.from_file(REAL)


@pytest.fixture(scope="module")
def prompts(config):
    neutralize = config.scoring.neutralize
    scoring = Scoring(
        membership=config.data.membership,
        neutralization=Neutralization(neutralize.exposures, neutralize.size_column, neutralize.industry_column),
    )
    recorder = _Recorder()
    researcher = Researcher(
        {role: (recorder, "m") for role in ("planner", "proposer", "reviewer", "distiller")},
        Prompts(
            templates=config.templates(), market=config.market, horizon=config.data.load("label").span_bars(),
            scoring=scoring.describe(), vocabulary=config.vocabulary(),
            mechanisms={m.name: m.description for m in config.mechanisms},
        ),
    )
    with pytest.raises(_Stop):
        researcher.plan(1, QUOTA, CONTEXT, [])
    with pytest.raises(_Stop):
        researcher.propose(1, DIRECTIONS, CONTEXT, None, None, 0)
    return {"plan": recorder.prompts["planner"], "propose": recorder.prompts["proposer"]}


def _sections(prompt: str) -> list[tuple[str, str]]:
    """The blocks of ``prompt`` between blank lines, each named by up to two words before any ':', '(' or ','."""
    blocks = [block for block in prompt.strip().split("\n\n") if block.strip()]
    return [(" ".join(re.split(r"[:(,]", block.splitlines()[0])[0].split()[:2]), block) for block in blocks]


@pytest.mark.parametrize("name", ["plan", "propose"])
def test_the_first_round_prompts_keep_every_section_in_order(prompts, name):
    old, new = _sections((OLD / f"{name}.txt").read_text()), _sections(prompts[name])
    added = [heading for heading, _ in new if heading not in {h for h, _ in old}]
    assert added == ADDED[name]
    assert [h for h, _ in new if h not in added] == [h for h, _ in old]


@pytest.mark.parametrize("name", ["plan", "propose"])
def test_only_the_sections_worded_by_the_config_differ(prompts, name):
    old = _sections((OLD / f"{name}.txt").read_text())
    new = [(h, block) for h, block in _sections(prompts[name]) if h not in ADDED[name]]
    changed = {h for (h, before), (_, after) in zip(old, new) if before != after}
    assert changed <= CONFIGURED
    assert {"You are", "The data"} <= changed


def test_the_configured_wording_is_the_configs(config, prompts):
    for name in prompts:
        assert f"The data: {config.market}." in (OLD / f"{name}.txt").read_text()  # the market as it was worded
    for prompt in prompts.values():
        sections = dict(_sections(prompt))
        assert config.market in sections["You are"] and config.market in sections["The data"]
        assert "the next 5 bars" in sections["The data"]
        assert "neutralised against the industry (industry) and the logarithm of the market cap (marketcap)" in (
            sections["The data"]
        )
        assert "scored on the members of the research universe that day" in sections["The data"]
    mechanisms = dict(_sections(prompts["plan"]))["The mechanism"]
    assert [line.split(":")[0] for line in mechanisms.splitlines()[1:]] == [f"- {kind}" for kind in QUOTA]
    assert "about 5 bars" in dict(_sections(prompts["propose"]))["Answer with"]
