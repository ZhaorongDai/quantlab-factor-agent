"""CampaignConfig.from_file: components given as a store folder with chosen outputs, and the real config."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from quantlab.core.store_folder import save_component
from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.stock import StockDataset
from quantlab.factor.config import FactorConfig
from quantlab.factor.predefined.alpha158 import Alpha158Stock

from factor_agent.candidate import MECHANISM_TYPES
from factor_agent.config import ROLES, CampaignConfig, load


@pytest.fixture
def store(tmp_path):
    """A folder whose ``component.json`` records an Alpha158Stock of every output."""
    xr.Dataset(
        {"Close": (["timestamp", "symbol"], np.ones((3, 2)))},
        coords={"timestamp": pd.bdate_range("2024-01-01", periods=3), "symbol": ["A", "B"]},
    ).to_zarr(tmp_path / "prices.zarr")
    dataset = StockDataset(DatasetConfig(
        zarr_file_path=str(tmp_path / "prices.zarr"), raw_data_dir_path=str(tmp_path), market="us_equity", frequency="1d",
    ))
    factor = Alpha158Stock(FactorConfig(
        warmup_bars=60, dataset=dataset, mode="batch", data_columns=("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"),
        file_path=str(tmp_path / "alpha158" / "alpha158.zarr"),
    ))
    save_component(factor, tmp_path / "alpha158")
    return tmp_path / "alpha158"


def test_a_store_entry_is_the_stores_factor_pinned_to_the_named_outputs(store):
    factor = load({"store": str(store), "factor_names": ["KMID", "STD20"]})
    assert isinstance(factor, Alpha158Stock)
    assert factor.get_factor_names() == ("KMID", "STD20")
    assert factor.config.file_path == str(store / "alpha158.zarr")


def test_a_store_entry_without_names_is_the_whole_store(store):
    assert len(load({"store": str(store)}).get_factor_names()) == 169


def test_a_store_entry_naming_an_output_the_store_lacks_is_refused(store):
    with pytest.raises(ValueError, match="NOPE"):
        load({"store": str(store), "factor_names": ["KMID", "NOPE"]})


def test_the_reference_of_a_config_file_may_be_store_entries(store, tmp_path):
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump({
        "run_dir": str(tmp_path / "run"),
        "segments": {"training": {"start": "2024-01-01", "end": "2024-01-03"}},
        "data": {"prices": "p", "label": "l", "membership": "m", "exposures": "e"},
        "reference": [{"store": str(store), "factor_names": ["KMID"]}],
        "llm": {role: {"backend": "scripted", "model": "m", "replies": "r.json"} for role in ROLES},
    }))
    [factor] = [load(c) for c in CampaignConfig.from_file(path).reference]
    assert factor.get_factor_names() == ("KMID",)


#: The first real campaign's config (#8); its data is on the server only, so only its text is checked here.
REAL = Path(__file__).parent.parent / "configs" / "us3000_ret5.yaml"


def test_the_real_config_reads_with_sel0s_191_reference_factors_and_a_direction_per_mechanism_type():
    config = CampaignConfig.from_file(REAL)
    names = [name for entry in config.reference for name in entry["factor_names"]]
    assert len(names) == len(set(names)) == 191
    assert {direction.split(":")[0] for direction in config.directions} == set(MECHANISM_TYPES)
    assert not re.search(r"\b(19|20)\d\d\b", " ".join(config.directions))  # no date reaches a prompt
    assert {role.model for role in config.roles.values()} == {"deepseek/deepseek-v4-pro-0813"}
