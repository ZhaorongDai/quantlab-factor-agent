"""A small synthetic whole-market store with a planted signal, built through quantlab's public API.

Two hundred symbols of adjusted OHLCV from 2009-06-01 to 2013-12-31, a
membership panel and the 5-day open-to-open label (quantlab's ``Return``
read on the members of day t through ``MembershipMaskedLabel``), as the
shared ``labels/market/ret_5`` store is built.

The planted signal is the intraday move ``close / open``: its shock ``z`` on
day t moves the next open-to-open return, so it predicts the label at t. It
has the stated (positive) sign only on the members of day t and only inside
``TRAINING``; for non-members and on every other date the effect is
reversed. A candidate on ``close / open`` is therefore accepted only by an
evaluator that masks by membership on day t and reads the training segment
alone.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.dataset.config import ConstituentDatasetConfig, DatasetConfig
from quantlab.dataset.raw import IndexConstituentDataset
from quantlab.dataset.stock import StockDataset
from quantlab.factor.config import FactorConfig
from quantlab.label.predefined.fret import Return
from quantlab.label.predefined.membership_mask import MembershipMaskedLabel

START, END = "2009-06-01", "2013-12-31"
TRAINING = ("2010-01-01", "2011-12-30")
N_SYMBOLS = 200
#: The first half are members throughout; the rest join after the training segment.
MEMBERS = N_SYMBOLS // 2
LATE_JOIN = "2013-07-01"
COVERAGE_START = "2009-01-02"
HORIZON = 5
SEED = 7


class IntervalMembership(IndexConstituentDataset):
    """A membership panel built from the class-level ``intervals``; a rebuilt one reads its store."""

    intervals: list[tuple] = []

    def _pit_coverage_start(self, config) -> str:
        return COVERAGE_START

    def _build_intervals(self) -> pl.DataFrame:
        return pl.DataFrame(self.intervals, schema=["symbol", "start_date", "end_date"], orient="row")


def symbols() -> list[str]:
    return [f"S{i:03d}" for i in range(N_SYMBOLS)]


def _prices(timestamps: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    n_bars = len(timestamps)
    z = rng.normal(size=(n_bars, N_SYMBOLS))
    in_training = (timestamps >= TRAINING[0]) & (timestamps <= TRAINING[1])
    sign = np.where(in_training, 1.0, -1.0)[:, None] * np.where(np.arange(N_SYMBOLS) < MEMBERS, 1.0, -1.0)
    # step[k] is the open-to-open log return into bar k. The label at t spans
    # the opens of t + 1 to t + 6, that is steps t + 2 to t + 6; step t + 2
    # carries bar t's shock.
    step = 0.01 * rng.normal(size=(n_bars, N_SYMBOLS))
    step[2:] += 0.01 * sign[:-2] * z[:-2]
    open_ = 40.0 * np.exp(np.cumsum(step, axis=0) - step[:1])
    close = open_ * np.exp(0.01 * z)
    return {
        "adjOpen": open_,
        "adjHigh": np.maximum(open_, close) * 1.01,
        "adjLow": np.minimum(open_, close) * 0.99,
        "adjClose": close,
        "adjVolume": rng.uniform(1e5, 1e6, size=(n_bars, N_SYMBOLS)),
    }


def build(root: Path) -> dict:
    """Write the stores under ``root`` and return the component configs a campaign config holds.

    Returns ``{"prices": ..., "label": ...}``, each a ``get_config()`` dict.
    """
    root = Path(root)
    timestamps = pd.bdate_range(START, END)
    panel = xr.Dataset(
        {name: (["timestamp", "symbol"], values) for name, values in _prices(timestamps).items()},
        coords={"timestamp": timestamps, "symbol": symbols()},
    )
    zarr_path = root / "market" / "synthetic_1d" / "synthetic_1d.zarr"
    zarr_path.parent.mkdir(parents=True)
    panel.to_zarr(zarr_path, mode="w")
    prices = StockDataset(DatasetConfig(
        zarr_file_path=str(zarr_path), raw_data_dir_path=str(root / "downloads"),
        market="us_equity", frequency="1d",
    ))

    IntervalMembership.intervals = [
        (symbol, COVERAGE_START if i < MEMBERS else LATE_JOIN, None) for i, symbol in enumerate(symbols())
    ]
    membership = IntervalMembership(ConstituentDatasetConfig(
        zarr_file_path=str(root / "universe" / "synthetic" / "membership.zarr"), as_of=END,
    ))
    membership.from_raw_data().save()

    ret = Return(FactorConfig(
        warmup_bars=2 * HORIZON + 5, dataset=prices, mode="batch", data_columns=("adjOpen",),
        kwargs={"n_forward_periods": HORIZON},
        file_path=str(root / "labels" / "market" / "ret_5" / "ret_5.zarr"), njobs=4,
    ))
    label = MembershipMaskedLabel(ret, membership)
    label.build(START, END)
    return {"prices": prices.get_config(), "label": label.get_config()}
