"""A high-frequency family end to end: daily statistics of a stored panel merged beside the prices.

The campaign's price dataset merges the synthetic bars with a synthetic
daily-statistics store, read under a prefix through quantlab's
``FactorStoreDataset``; the family's field file reads the prefixed columns
with ``bar: false``. The store has no value on every third bar, as a
session with too few trades leaves a statistic missing.
"""

import pytest

from factor_agent.gate.vocabulary import VocabularyError
from quantlab.core.component import rebuild
from quantlab.dataset.merged import MergedDataset
from tests import synthetic_market
from tests.campaign_runner import candidate, run, write_config
from tests.fake_llm import FakeLLM
from tests.vocabulary_files import PRICE_VOLUME

PREFIX = "hf_"

FIELD_FILE = """\
family: session_statistics
description: daily statistics of each session's minute bars
bar: false

fields:
  - name: rvol
    column: {rvol}
    unit: 1
    description: realized volatility of the session's minute returns
  - name: imbalance
    column: {imbalance}
    unit: 1
    description: signed share of the session's volume traded on upticks
"""

HIGH_FREQUENCY = candidate("high frequency", "alpha = Log(rvol) - Log(imbalance)")
#: Reads only prices, through a NaN-preserving operator that a missing bar blanks.
PRICES_ONLY = candidate("prices only", "alpha = high / Max(open, close)")


def _config(root, rvol=f"{PREFIX}rvol", imbalance=f"{PREFIX}imbalance", **sections):
    components = synthetic_market.build(root / "data")
    statistics = synthetic_market.statistics(root / "data", prefix=PREFIX)
    components["prices"] = MergedDataset([rebuild(components["prices"]), rebuild(statistics)]).get_config()
    fields = root / "session_statistics.yaml"
    fields.write_text(FIELD_FILE.format(rvol=rvol, imbalance=imbalance))
    return write_config(root, components, fields=[str(PRICE_VOLUME), str(fields)], **sections)


@pytest.fixture(scope="module")
def outcome(tmp_path_factory):
    root = tmp_path_factory.mktemp("high_frequency")
    config_path = _config(root, round={"candidates": 2, "min_per_type": 0, "max_rounds": 1, "retries": 0})
    fake = FakeLLM(lambda number, directions: [HIGH_FREQUENCY, PRICES_ONLY])
    result = run(config_path, fake, root / "kunquant_cache")
    return {record.candidate["description"]: record for record in result.records}


def test_a_candidate_reading_a_high_frequency_field_passes_the_gate_and_is_evaluated(outcome):
    record = outcome["high frequency"]
    assert record.status in ("accepted", "rejected"), record.reasons
    assert record.stage != "gate"
    assert record.metrics is not None
    # The statistic is missing on every third bar, so a third of the days have no value.
    assert 0.55 < record.metrics["coverage"] < 0.75


def test_a_missing_statistic_leaves_the_bar_and_its_prices_alone(outcome):
    record = outcome["prices only"]
    assert record.metrics is not None
    assert record.metrics["coverage"] == pytest.approx(1.0)


def test_a_field_reading_a_column_absent_from_the_merged_panel_fails_at_start_before_any_llm_call(tmp_path):
    config_path = _config(tmp_path, rvol="rvol", imbalance=f"{PREFIX}imbalanceTypo")
    fake = FakeLLM(lambda number, directions: [])
    with pytest.raises(VocabularyError) as refused:
        run(config_path, fake, tmp_path / "kunquant_cache")
    message = str(refused.value)
    # The stored name without the prefix is not a column of the panel; both problems are listed.
    assert "column 'rvol' is not in the price dataset" in message
    assert f"column '{PREFIX}imbalanceTypo' is not in the price dataset" in message
    assert fake.requests == []
