"""The example intraday field file and the example intraday campaign config.

Nothing reads server data: the field files pass the vocabulary check against
the columns the datasets would hold (SEP's for the daily family; for the
intraday family the variables of quantlab's one-minute trade bars and the
session inputs the campaign adds), the config's intraday dataset is rebuilt
from its config without opening a store, and the rest is checked by its text.
"""

import dataclasses
import re

import pytest
from quantlab.core.component import rebuild
from quantlab.dataset.massive.mapped_bars import MappedTradeBarDataset
from quantlab.dataset.massive.resample import TRADE_BAR_VARIABLES
from quantlab.dataset.roster import RosterDataset

from factor_agent.candidate import Candidate
from factor_agent.config import DEFAULT_MECHANISMS, CampaignConfig, GateConfig
from factor_agent.gate import Gate, Passed
from factor_agent.gate.vocabulary import VocabularyError
from factor_agent.prompts import INTRADAY_EXAMPLES
from factor_agent.session import SESSION_INPUTS
from factor_agent.zoo import Zoo
from tests.test_intraday_fields import SEP_COLUMNS
from tests.vocabulary_files import CONFIGS, EXAMPLE_OPERATORS, MINUTE_BARS, PRICE_VOLUME

EXAMPLE = CONFIGS / "us3000_intraday_ret5.yaml"
SECOND = CONFIGS / "us3000_intraday2_ret5.yaml"
DAILY = CONFIGS / "us3000_ret5.yaml"
#: The columns of the one-minute trade bars and the session inputs added to them.
INTRADAY_COLUMNS = set(TRADE_BAR_VARIABLES) | set(SESSION_INPUTS)
#: The trade bars' auction columns, which the field file leaves out.
AUCTIONS = {"open_auction_price", "open_auction_volume", "close_auction_price", "close_auction_volume"}
#: The first session of the one-minute trade bars.
MINUTE_START = "2016-10-10"


@pytest.fixture(scope="module")
def config() -> CampaignConfig:
    return CampaignConfig.from_file(EXAMPLE)


@pytest.fixture(scope="module")
def vocabulary(config):
    return config.vocabulary(columns=SEP_COLUMNS, intraday_columns=INTRADAY_COLUMNS)


def test_the_example_opens_the_daily_price_volume_family_and_the_minute_bars(config, vocabulary):
    assert config.fields == (str(PRICE_VOLUME), str(MINUTE_BARS))
    assert config.operators == str(EXAMPLE_OPERATORS)
    assert vocabulary.intraday_family == "minute_bars"
    assert vocabulary.intraday_windows == (5, 10, 15, 30, 60, 120)
    assert vocabulary.intraday_lags == (1, 5, 10, 15, 30, 60)


def test_every_trade_bar_column_but_the_auctions_and_every_session_input_is_a_field(vocabulary):
    columns = {f.column for f in vocabulary.fields.values() if f.family == "minute_bars"}
    assert columns == INTRADAY_COLUMNS - AUCTIONS
    assert vocabulary.intraday_columns == columns


def test_each_minute_field_has_its_unit_and_a_description(vocabulary):
    units = {f.name: tuple(f.dimension) for f in vocabulary.fields.values() if f.family == "minute_bars"}
    assert units["bar_close"] == units["last_price"] == (1, -1)
    assert units["bar_volume"] == units["bar_oddlot_volume"] == (0, 1)
    assert units["bar_dollar_volume"] == (1, 0)
    assert units["bar_trades"] == units["session_position"] == units["session_bars_left"] == (0, 0)
    assert all(vocabulary.fields[name].description for name in units)


def test_a_minute_field_reading_a_column_the_bars_lack_is_refused(config):
    with pytest.raises(VocabularyError, match="oddlot_volume.*not in the intraday dataset"):
        config.vocabulary(columns=SEP_COLUMNS, intraday_columns=INTRADAY_COLUMNS - {"oddlot_volume"})


def test_the_intraday_dataset_is_the_mapped_minute_bars_on_the_daily_prices_roster(config):
    minute = rebuild(config.data.intraday)
    assert isinstance(minute, RosterDataset) and isinstance(minute.dataset, MappedTradeBarDataset)
    assert minute.dataset.config.zarr_file_path.endswith("massive_permaticker_bars_1m.zarr")
    assert minute.dataset.config.source.config.trade_bars.config.bar_interval == "1m"
    assert minute.config.roster.get_config() == rebuild(config.data.prices["datasets"][0]).get_config()


def test_every_segment_lies_inside_the_minute_bars(config):
    segments = config.segments
    assert segments.training.start >= MINUTE_START
    assert segments.training.end < segments.validation.start and segments.validation.end < segments.holdout.start


@pytest.mark.parametrize(("description", "code"), INTRADAY_EXAMPLES, ids=["late share", "volatility"])
def test_the_prompts_example_candidates_pass_the_gate_on_the_example(vocabulary, description, code):
    candidate = Candidate(
        hypothesis="Intraday trading reveals what daily bars do not.", mechanism_type="microstructure",
        horizon_rationale="Intraday patterns persist for days.", description=description, code=code,
    )
    result = Gate(GateConfig(timeout_seconds=20.0), vocabulary).check(candidate, Zoo())
    assert isinstance(result, Passed), getattr(result, "reasons", None)


def test_the_example_keeps_the_daily_examples_label_scoring_reference_roles_and_acceptance(config):
    daily = CampaignConfig.from_file(DAILY)
    assert config.data.prices == daily.data.prices
    assert (config.data.label, config.data.membership, config.scoring) == (daily.data.label, daily.data.membership, daily.scoring)
    assert config.reference == daily.reference
    assert config.roles == daily.roles and config.acceptance == daily.acceptance
    assert config.run_dir != daily.run_dir and not set(config.directions) & set(daily.directions)


def test_the_known_factors_are_the_high_frequency_examples_every_stored_minute_factor_among_them(config):
    assert config.known == CampaignConfig.from_file(CONFIGS / "us3000_hf_ret5.yaml").known


def test_every_mechanism_type_has_two_seed_directions_on_the_minute_fields_and_none_names_a_year(config, vocabulary):
    types = [direction.split(":")[0] for direction in config.directions]
    assert sorted(types) == sorted(2 * [m.name for m in DEFAULT_MECHANISMS])
    assert not re.search(r"\b(19|20)\d\d\b", " ".join(config.directions))
    minute_fields = {f.name for f in vocabulary.fields.values() if f.family == "minute_bars"}
    assert all(set(re.findall(r"\b\w+\b", direction)) & minute_fields for direction in config.directions)


@pytest.mark.parametrize("path", [EXAMPLE, SECOND, MINUTE_BARS], ids=lambda path: path.name)
def test_the_files_name_no_path_outside_the_data_root_and_no_ticket(path):
    text = path.read_text()
    assert all(p.startswith("/data/quantlab/") for p in re.findall(r"(?<![\w.:/])/[\w./-]+", text))
    assert not re.search(r"#\d|\bticket\b|\bADR\b|training server", text)


def test_the_second_campaign_differs_from_the_first_only_in_its_run_dir_directions_threshold_and_memory(config):
    second = CampaignConfig.from_file(SECOND)
    assert second.run_dir == "/data/quantlab/runs/factor_agent_us3000_intraday2_ret5" != config.run_dir
    assert len(second.directions) == 10 and not set(second.directions) & set(config.directions)
    assert (second.admission.trial_exponent, second.round.memory_every) == (0.5, 1)
    assert dataclasses.replace(
        second, run_dir=config.run_dir, directions=config.directions,
        admission=config.admission, round=config.round,
    ) == config


def test_the_second_campaigns_directions_cover_every_type_twice_on_the_minute_fields_and_name_no_year(vocabulary):
    test_every_mechanism_type_has_two_seed_directions_on_the_minute_fields_and_none_names_a_year(
        CampaignConfig.from_file(SECOND), vocabulary)
