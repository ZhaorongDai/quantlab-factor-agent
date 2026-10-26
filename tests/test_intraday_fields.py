"""The repository's two high-frequency field files and the example high-frequency campaign config.

Nothing reads server data: the field files are loaded with the vocabulary
check against the columns the merged panel would hold (SEP's, and each
statistic store's outputs under its prefix, as quantlab names the outputs),
and the config is checked by its text.
"""

import re

import pytest
from quantlab.factor.predefined.intraday import INTRADAY_OUTPUTS
from quantlab.factor.predefined.price_volume_intraday import PRICE_VOLUME_INTRADAY_STATS

from factor_agent.candidate import Candidate
from factor_agent.config import DEFAULT_MECHANISMS, CampaignConfig, GateConfig
from factor_agent.gate import Gate, Passed
from factor_agent.gate.vocabulary import Vocabulary
from factor_agent.zoo import Zoo
from tests.vocabulary_files import CONFIGS, EXAMPLE_OPERATORS, PRICE_VOLUME

SESSION = CONFIGS / "fields" / "intraday_session.yaml"
PRICE_VOLUME_SESSION = CONFIGS / "fields" / "intraday_price_volume.yaml"

#: SEP's columns, which neither prefix may reach.
SEP_COLUMNS = ("open", "high", "low", "close", "volume", "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume",
               "anomaly_flag", "cumfacshr", "divCash", "splitFactor")
SESSION_PREFIX, PRICE_VOLUME_PREFIX = "trade_", "pv_"
SESSION_COLUMNS = tuple(SESSION_PREFIX + name for name in INTRADAY_OUTPUTS)
PRICE_VOLUME_COLUMNS = tuple(PRICE_VOLUME_PREFIX + name for name in PRICE_VOLUME_INTRADAY_STATS)


@pytest.fixture(scope="module")
def vocabulary() -> Vocabulary:
    return Vocabulary.load(
        [PRICE_VOLUME, SESSION, PRICE_VOLUME_SESSION], EXAMPLE_OPERATORS,
        columns=SEP_COLUMNS + SESSION_COLUMNS + PRICE_VOLUME_COLUMNS,
    )


def _family(vocabulary, family):
    return [f for f in vocabulary.fields.values() if f.family == family]


@pytest.mark.parametrize(
    ("family", "columns"),
    [("intraday_session", SESSION_COLUMNS), ("intraday_price_volume", PRICE_VOLUME_COLUMNS)],
)
def test_every_statistic_of_a_store_is_a_field_read_under_the_stores_prefix(vocabulary, family, columns):
    fields = _family(vocabulary, family)
    assert sorted(f.column for f in fields) == sorted(columns)
    assert all(f.name == f.column and f.dimension is not None and f.description for f in fields)


def test_the_prefixes_clash_with_no_sep_column_and_with_each_other():
    assert not any(c.startswith((SESSION_PREFIX, PRICE_VOLUME_PREFIX)) for c in SEP_COLUMNS)
    assert not set(SESSION_COLUMNS) & set(PRICE_VOLUME_COLUMNS)


def test_the_minute_amihud_ratio_is_per_dollar_and_every_other_statistic_a_pure_number(vocabulary):
    fields = _family(vocabulary, "intraday_session") + _family(vocabulary, "intraday_price_volume")
    units = {f.name: tuple(f.dimension) for f in fields}
    assert units.pop("trade_amihud") == (-1, 0)
    assert set(units.values()) == {(0, 0)}
    trade_size = vocabulary.fields["trade_trade_size"].description
    assert "shares per trade" in trade_size


@pytest.mark.parametrize(
    ("code", "dimension"),
    [
        ("alpha = WindowedAvg(trade_rskew, 20)", "pure number"),
        ("alpha = WindowedStddev(trade_amihud, 20) / WindowedAvg(trade_amihud, 20)", "pure number"),
        ("alpha = trade_amihud * close * volume", "pure number"),
        ("alpha = pv_UP_RET_STD - pv_DOWN_RET_STD", "pure number"),
    ],
)
def test_a_candidate_on_session_statistics_passes_the_gate_with_its_dimension(vocabulary, code, dimension):
    candidate = Candidate(
        hypothesis="Session statistics carry information daily bars do not.", mechanism_type="microstructure",
        horizon_rationale="Intraday patterns persist for days.", description="A session statistic.", code=code,
    )
    result = Gate(GateConfig(timeout_seconds=20.0), vocabulary).check(candidate, Zoo())
    assert isinstance(result, Passed), getattr(result, "reasons", None)
    assert result.stats["dimension"] == dimension


#: The example campaigns' configs; their data is not in the repository, so only their text is checked.
REAL_HF = CONFIGS / "us3000_hf_ret5.yaml"
OTHER_EXAMPLES = (CONFIGS / "us3000_ret5.yaml", CONFIGS / "us3000_fund_ret5.yaml")
FACTORS = "/data/quantlab/factors/market"
#: The known factors: whole-market factor libraries and the minute price-volume factors.
KNOWN = tuple(f"{FACTORS}/{name}" for name in (
    "alpha101_neutral", "alpha158_neutral", "fundamental_alpha_neutral", "daily_anomaly_neutral",
    "short_horizon_neutral", "barra_style", "price_volume_intraday_neutral",
))


@pytest.fixture(scope="module")
def hf():
    return CampaignConfig.from_file(REAL_HF)


def test_the_high_frequency_config_reads_with_its_segments_inside_the_minute_data(hf):
    segments = hf.segments
    assert (segments.training.start, segments.training.end) == ("2016-10-11", "2021-12-31")
    assert (segments.validation.start, segments.validation.end) == ("2022-01-03", "2023-12-29")
    assert (segments.holdout.start, segments.holdout.end) == ("2024-01-02", "2026-10-15")


def test_the_high_frequency_config_opens_price_volume_and_both_session_families_and_no_fundamentals(hf):
    assert hf.fields == (str(PRICE_VOLUME), str(SESSION), str(PRICE_VOLUME_SESSION))
    assert hf.operators == str(EXAMPLE_OPERATORS)


def test_the_price_dataset_merges_masked_sep_with_both_stores_under_the_field_files_prefixes(hf):
    prices = hf.data.prices
    assert prices["name"] == "quantlab.dataset.bad_prints.BadPrintMaskedDataset"
    sep, *stores = prices["datasets"]
    assert sep["name"] == "quantlab.dataset.sharadar.stock.SharadarStockDataset"
    assert {(store["name"], store["zarr_file_path"], store["prefix"]) for store in stores} == {
        ("quantlab.dataset.factor_store.FactorStoreDataset",
         f"{FACTORS}/intraday_stats/intraday_stats_resample_1d.zarr", SESSION_PREFIX),
        ("quantlab.dataset.factor_store.FactorStoreDataset",
         f"{FACTORS}/price_volume_intraday_stats/price_volume_intraday_stats_resample_1d.zarr", PRICE_VOLUME_PREFIX),
    }


def test_the_known_factors_are_the_earlier_campaigns_and_the_minute_price_volume_factors(hf):
    assert hf.known == KNOWN


@pytest.mark.parametrize("other", OTHER_EXAMPLES, ids=lambda path: path.stem)
def test_the_high_frequency_config_keeps_the_other_examples_reference_scoring_label_roles_and_acceptance(hf, other):
    other = CampaignConfig.from_file(other)
    assert hf.reference == other.reference
    assert (hf.data.label, hf.data.membership, hf.scoring) == (other.data.label, other.data.membership, other.scoring)
    assert hf.roles == other.roles and hf.acceptance == other.acceptance
    assert hf.run_dir != other.run_dir and not set(hf.directions) & set(other.directions)
    assert hf.round.groups == 4


def test_every_mechanism_type_has_two_seed_directions_and_none_names_a_year(hf):
    types = [direction.split(":")[0] for direction in hf.directions]
    assert sorted(types) == sorted(2 * [m.name for m in DEFAULT_MECHANISMS])
    assert not re.search(r"\b(19|20)\d\d\b", " ".join(hf.directions))  # no date reaches a prompt


@pytest.mark.parametrize("path", [REAL_HF, SESSION, PRICE_VOLUME_SESSION], ids=lambda path: path.name)
def test_the_files_name_no_path_outside_the_data_root_and_no_ticket(path):
    text = path.read_text()
    assert all(p.startswith("/data/quantlab/") for p in re.findall(r"(?<![\w.:/])/[\w./-]+", text))
    assert not re.search(r"#\d|\bticket\b|\bADR\b|training server", text)
