"""Run a campaign from a YAML config on the synthetic store, with the fake LLM as its backends."""

import json
from pathlib import Path

import pytest
import yaml

from factor_agent.campaign import Campaign
from factor_agent.config import ROLES, CampaignConfig
from quantlab.factor.kunquant_compile import CACHE_ENV
from tests import synthetic_market
from tests.vocabulary_files import EXAMPLE_OPERATORS, PRICE_VOLUME

def candidate(description: str, code: str, **overrides) -> dict:
    """A candidate item with neutral text: no date, no number."""
    return {
        "hypothesis": f"Hypothesis behind {description}.",
        "mechanism_type": "behavioural_bias",
        "horizon_rationale": "Resolves within a week.",
        "description": description,
        "code": code,
        **overrides,
    }


#: The synthetic store as the prompts describe it.
SYNTHETIC_MARKET = "adjusted daily bars of every synthetic test symbol"

#: The import path of quantlab's xgb, the acceptance model of a config that names none.
XGB = "quantlab.model.predefined.xgb.XGBoostRegressor"

#: The synthetic store's exposure columns.
NEUTRALIZE = {"size_column": "marketcap", "industry_column": "industry"}


def write_config(root: Path, components: dict, reference=(), **sections) -> Path:
    """Write a campaign config under ``root``; ``reference`` holds factor ``get_config()`` dicts.

    ``components`` are ``synthetic_market.build``'s; by default the campaign is scored on the
    members of day t and neutralised against the synthetic exposures.
    """
    path = root / "campaign.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(json.loads(json.dumps({
        "run_dir": str(root / "run"),
        "market": SYNTHETIC_MARKET,
        "segments": {"training": {"start": synthetic_market.TRAINING[0], "end": synthetic_market.TRAINING[1]}},
        "data": {name: components[name] for name in ("prices", "label", "membership")},
        "scoring": {"neutralize": {"exposures": components["exposures"], **NEUTRALIZE}},
        "fields": [str(PRICE_VOLUME)],
        "operators": str(EXAMPLE_OPERATORS),
        "reference": list(reference),
        "evaluator": {"warmup_bars": 60, "njobs": 4},
        "llm": {role: {"backend": "scripted", "model": f"model-{role}", "replies": "unused.json"} for role in ROLES},
        "acceptance": {"model": {"name": XGB}},
        **sections,
    }))))
    return path


def run(config_path: Path, fake, cache: Path):
    """Run the campaign of ``config_path`` with every role answered by ``fake``, or by the config's backends if None."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(CACHE_ENV, str(cache))
        return Campaign(CampaignConfig.from_file(config_path), backends=fake.backends() if fake else None).run()
