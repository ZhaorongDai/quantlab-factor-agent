"""Run a campaign from a YAML config on the synthetic store, with the fake LLM as its backends."""

import json
from pathlib import Path

import pytest
import yaml

from factor_agent.campaign import Campaign
from factor_agent.config import ROLES, CampaignConfig
from quantlab.factor.kunquant_compile import CACHE_ENV
from tests import synthetic_market

def candidate(description: str, code: str, **overrides) -> dict:
    """A candidate item with neutral text: no keyword the mechanism binding acts on, no date, no number."""
    return {
        "hypothesis": f"Hypothesis behind {description}.",
        "mechanism_type": "behavioural_bias",
        "expected_sign": 1,
        "horizon_rationale": "Resolves within a week.",
        "reference": "Synthetic test reference",
        "description": description,
        "code": code,
        **overrides,
    }


def write_config(root: Path, components: dict, reference=(), **sections) -> Path:
    """Write a campaign config under ``root``; ``reference`` holds factor ``get_config()`` dicts."""
    path = root / "campaign.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(json.loads(json.dumps({
        "run_dir": str(root / "run"),
        "segments": {"training": {"start": synthetic_market.TRAINING[0], "end": synthetic_market.TRAINING[1]}},
        "data": components,
        "reference": list(reference),
        "evaluator": {"warmup_bars": 60, "njobs": 4},
        "llm": {role: {"backend": "scripted", "model": f"model-{role}", "replies": "unused.json"} for role in ROLES},
        **sections,
    }))))
    return path


def run(config_path: Path, fake, cache: Path):
    """Run the campaign of ``config_path`` with every role answered by ``fake``, or by the config's backends if None."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(CACHE_ENV, str(cache))
        return Campaign(CampaignConfig.from_file(config_path), backends=fake.backends() if fake else None).run()
