"""A trial's daily rank IC and partial IC, kept by the campaign for each evaluated trial and read by the frontend.

``run`` ran one round of four slots on the synthetic market: the planted factor,
noise, a name the Gate refuses, and the planted factor ranked.
"""

import json

import numpy as np
import pytest

from factor_agent.web.run_view import RunView, read_ic
from tests import synthetic_market
from tests.campaign_runner import candidate, run, write_config
from tests.fake_llm import FakeLLM

ROUND = {"candidates": 4, "min_per_type": 0, "max_rounds": 1}
PROPOSALS = [
    candidate("planted", "alpha = close / open"),
    candidate("noise", "alpha = BackRef(volume, 10) / WindowedAvg(volume, 60)"),
    candidate("unknown name", "alpha = SecretOp(close)"),
    candidate("planted again", "alpha = Rank(close / open)"),
]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("ic")
    components = synthetic_market.build(root / "data")
    run(write_config(root / "run", components, round=ROUND), FakeLLM(lambda number, directions: PROPOSALS),
        root / "kunquant_cache")
    return root / "run" / "run"


def test_an_evaluated_trial_keeps_its_daily_ic_whose_mean_is_the_ledgers(runs):
    trials = {t.slot: t for t in RunView.read(runs).trials}
    evaluated = [slot for slot, t in trials.items() if t.metrics]
    assert evaluated
    for slot in evaluated:
        series = read_ic(runs, 1, slot)
        assert len(series.days) == len(series.ic) == len(series.partial_ic)
        ic = np.array([np.nan if v is None else v for v in series.ic])
        assert np.isfinite(ic).sum() == trials[slot].metrics["n_days"]
        assert np.nanmean(ic) == pytest.approx(trials[slot].metrics["ic_mean"])
        partial = np.array([np.nan if v is None else v for v in series.partial_ic])
        assert np.nanmean(partial) == pytest.approx(trials[slot].metrics["partial_ic_mean"])


def test_a_trial_never_evaluated_or_a_run_without_kept_series_has_no_daily_ic(runs):
    trials = {t.slot: t for t in RunView.read(runs).trials}
    refused = next(slot for slot, t in trials.items() if not t.metrics)
    assert read_ic(runs, 1, refused) is None
    assert read_ic(runs, 2, 0) is None


def test_the_daily_ic_is_kept_apart_from_the_ledger(runs):
    line = json.loads((runs / "ledger.jsonl").read_text().splitlines()[0])
    assert "ic" not in line and all(not isinstance(v, list) or len(v) <= 10 for v in (line.get("metrics") or {}).values())
