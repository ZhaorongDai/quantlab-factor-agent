"""An intraday family end to end: a two-stage candidate on the minute bars runs through a campaign.

The campaign reads the daily bars of ``synthetic_intraday_market`` as its
prices and its one-minute bars as its intraday dataset, with the price-volume
family and the minute-bar family (``configs/fields/massive_minute_bars.yaml``)
open. The fake LLM proposes a candidate measuring the share of each
session's volume in its last bars, averaged over five days, which is the
planted signal, beside daily candidates, which cannot see it.
"""

import json

import pandas as pd
import pytest

from factor_agent.gate.vocabulary import VocabularyError
from factor_agent.snapshot import SnapshotMismatch
from tests import synthetic_intraday_market as market
from tests.campaign_runner import candidate, run, write_config
from tests.fake_llm import FakeLLM
from tests.vocabulary_files import MINUTE_BARS, PRICE_VOLUME

LATE_SHARE = candidate(
    "late volume share",
    f"late = SessionMean(Select(session_bars_left < {market.LATE_BARS}, bar_volume, ConstantOp(0)))\n"
    "alpha = WindowedAvg(late / SessionMean(bar_volume), 5)",
)
#: The same measurement normalised by a daily field, with a second intraday branch: still the signal.
MIXED = candidate(
    "late share against daily turnover",
    f"late = SessionMean(Select(session_bars_left < {market.LATE_BARS}, bar_volume, ConstantOp(0)))\n"
    "alpha = Rank(WindowedAvg(late / SessionMean(bar_volume), 5)) + Rank(SessionMean(bar_close) / close) * 0.1",
)
DAILY_VOLUME = candidate("daily volume", "alpha = WindowedAvg(volume / WindowedAvg(volume, 20), 5)")
DAILY_PRICE = candidate("daily price", "alpha = WindowedAvg(close / open, 5)")
CANDIDATES = [LATE_SHARE, MIXED, DAILY_VOLUME, DAILY_PRICE]


def _config(root, components=None, fields=(PRICE_VOLUME, MINUTE_BARS), training=market.TRAINING, **data):
    components = components or market.build(root / "data")
    return write_config(
        root, {**components, "exposures": None},  # scored without neutralisation, below
        segments={"training": {"start": training[0], "end": training[1]}},
        data={**{name: components[name] for name in ("prices", "label", "membership", "intraday")}, **data},
        scoring={"neutralize": None},
        fields=[str(path) for path in fields],
        round={"candidates": len(CANDIDATES), "min_per_type": 0, "max_rounds": 1, "retries": 0},
    ), components


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    root = tmp_path_factory.mktemp("intraday_family")
    config_path, components = _config(root)
    fake = FakeLLM(lambda number, directions: CANDIDATES[:len(directions)])
    result = run(config_path, fake, root / "kunquant_cache")
    return {"root": root, "config": config_path, "components": components,
            "records": {record.candidate["description"]: record for record in result.records}}


def test_a_two_stage_candidate_passes_the_gate_is_evaluated_and_admitted(campaign):
    record = campaign["records"]["late volume share"]
    assert record.status == "accepted", record.reasons
    assert record.metrics["ic_t"] > 3
    # The daily windows reaching back before the minute bars begin are missing, not refused.
    assert record.metrics["coverage"] == pytest.approx(1.0, abs=0.05)


def test_aggregated_values_combine_with_daily_fields_across_several_intraday_branches(campaign):
    record = campaign["records"]["late share against daily turnover"]
    assert record.stage != "gate", record.reasons
    assert record.metrics is not None and record.metrics["ic_t"] > 3


@pytest.mark.parametrize("description", ["daily volume", "daily price"])
def test_a_daily_candidate_cannot_find_the_signal_the_minute_bars_hold(campaign, description):
    record = campaign["records"][description]
    assert record.metrics is not None
    assert abs(record.metrics["ic_t"]) < 3


def test_the_run_snapshot_records_the_intraday_dataset(campaign):
    snapshot = json.loads((campaign["root"] / "run" / "snapshot.json").read_text())
    assert snapshot["settings"]["intraday"]["dataset"] == json.loads(json.dumps(campaign["components"]["intraday"]))
    assert "fields/massive_minute_bars.yaml" in snapshot["files"]


def test_a_resume_with_another_intraday_dataset_is_refused_before_any_llm_call(campaign, tmp_path):
    root = tmp_path / "resumed"
    other = market.build(root / "data")
    config_path, _ = _config(root, components={**campaign["components"], "intraday": other["intraday"]})
    _resume_in(campaign, config_path)
    fake = FakeLLM(lambda number, directions: [])
    with pytest.raises(SnapshotMismatch, match="intraday.dataset was"):
        run(config_path, fake, tmp_path / "kunquant_cache")
    assert fake.requests == []


def test_a_resume_with_another_intraday_window_set_is_refused_before_any_llm_call(campaign, tmp_path):
    fields = tmp_path / "massive_minute_bars.yaml"
    fields.write_text(MINUTE_BARS.read_text().replace("windows: [5, 10, 15, 30, 60, 120]", "windows: [5, 10]"))
    config_path, _ = _config(tmp_path / "resumed", components=campaign["components"], fields=(PRICE_VOLUME, fields))
    _resume_in(campaign, config_path)
    fake = FakeLLM(lambda number, directions: [])
    with pytest.raises(SnapshotMismatch, match="fields/massive_minute_bars.yaml changed"):
        run(config_path, fake, tmp_path / "kunquant_cache")
    assert fake.requests == []


def test_an_intraday_family_without_an_intraday_dataset_fails_at_start_before_any_llm_call(campaign, tmp_path):
    config_path, _ = _config(tmp_path, components=campaign["components"], intraday=None)
    fake = FakeLLM(lambda number, directions: [])
    with pytest.raises(VocabularyError, match="minute_bars is an intraday family, but the config names no data.intraday"):
        run(config_path, fake, tmp_path / "kunquant_cache")
    assert fake.requests == []


def test_an_intraday_field_reading_a_column_the_minute_bars_lack_fails_at_start(campaign, tmp_path):
    fields = tmp_path / "massive_minute_bars.yaml"
    fields.write_text(MINUTE_BARS.read_text().replace("column: sell_volume", "column: sell_volume_typo"))
    config_path, _ = _config(tmp_path, components=campaign["components"], fields=(PRICE_VOLUME, fields))
    fake = FakeLLM(lambda number, directions: [])
    with pytest.raises(VocabularyError, match="column 'sell_volume_typo' is not in the intraday dataset"):
        run(config_path, fake, tmp_path / "kunquant_cache")
    assert fake.requests == []


def test_a_training_segment_starting_at_the_first_session_counts_the_days_before_a_window_fills_as_missing(
    campaign, tmp_path,
):
    training = (market.MINUTE_START, market.TRAINING[1])
    config_path, _ = _config(tmp_path, components=campaign["components"], training=training)
    fake = FakeLLM(lambda number, directions: CANDIDATES[:len(directions)])
    result = run(config_path, fake, tmp_path / "kunquant_cache")
    record = next(r for r in result.records if r.candidate["description"] == "late volume share")
    assert record.metrics is not None, record.reasons
    # The five-day mean has no value on the segment's first four days, which have no session before them.
    days = len(pd.bdate_range(*training))
    # Not a day more or less.
    assert record.metrics["coverage"] == pytest.approx((days - 4) / days, abs=3e-3)
    assert record.metrics["ic_t"] > 3


def _resume_in(campaign, config_path) -> None:
    """Point ``config_path`` at the finished campaign's run directory, so running it resumes that run."""
    text = config_path.read_text().replace(str(config_path.parent / "run"), str(campaign["root"] / "run"))
    config_path.write_text(text)


def test_a_resume_evaluates_the_accepted_intraday_factor_again_alone(campaign):
    # The accepted factor reads only session columns; the resume puts it back in the reference pool.
    fake = FakeLLM(lambda number, directions: [])
    result = run(campaign["config"], fake, campaign["root"] / "kunquant_cache")
    assert result.stop.by == "max_rounds"
    assert [r.candidate["description"] for r in result.records if r.status == "accepted"] == ["late volume share"]
