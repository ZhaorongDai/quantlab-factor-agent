"""A small synthetic whole-market store with planted signals, built through quantlab's public API.

Two hundred symbols of adjusted OHLCV from 2009-06-01 to 2013-12-31, a
membership panel, the 5-day open-to-open label (quantlab's ``Return`` read
on the members of day t through ``MembershipMaskedLabel``), as the shared
``labels/market/ret_5`` store is built, and the exposures (market cap and
industry) that neutralisation regresses on.

Each signal is a daily shock on bar t that moves the open-to-open step into
bar t + 2, the first step of the label at t. Every other shock of a field
moves nothing, so a field read with a lag of five bars or more is noise.

- The intraday move ``close / open`` (shock ``z``) has the stated
  (positive) sign only on the members of day t and only inside
  ``TRAINING``; for non-members and on every other date the effect is
  reversed. It is accepted only by an evaluator that masks by membership on
  day t and reads the training segment alone.
- The volume shock ``volume / WindowedAvg(volume, 20)`` predicts the label
  with a smaller, positive effect on every symbol and date: information
  that no price factor carries.
- The upper wick ``high / Max(open, close)`` predicts it weakly, with a
  Newey-West t a little above 3 on the training segment.
- The lower wick ``Min(open, close) / low`` predicts it only in its top
  decile: a shock above the 90th percentile moves the return, the rest do
  not.
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
from quantlab.factor.polars import FactorPolars
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
#: Open-to-open step, as a log return, per unit of each shock.
INTRADAY_EFFECT = 0.01
VOLUME_EFFECT = 0.0025
UPPER_WICK_EFFECT = 0.00062
TAIL_EFFECT = 0.012
#: Industries the exposures assign.
N_INDUSTRIES = 5


class IntervalMembership(IndexConstituentDataset):
    """A membership panel built from the class-level ``intervals``; a rebuilt one reads its store."""

    intervals: list[tuple] = []

    def _pit_coverage_start(self, config) -> str:
        return COVERAGE_START

    def _build_intervals(self) -> pl.DataFrame:
        return pl.DataFrame(self.intervals, schema=["symbol", "start_date", "end_date"], orient="row")


class AdjustedMomentum(FactorPolars):
    """A 20-bar momentum computed with Polars: a reference factor without a KunQuant graph."""

    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        close = pl.col("adjClose")
        return (
            lf.sort(["symbol", "timestamp"])
            .with_columns((close / close.shift(20).over("symbol") - 1.0).alias("momentum_20"))
            .select(["timestamp", "symbol", "momentum_20"])
        )


def symbols() -> list[str]:
    return [f"S{i:03d}" for i in range(N_SYMBOLS)]


def _prices(timestamps: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    n_bars = len(timestamps)
    shape = (n_bars, N_SYMBOLS)
    z = rng.normal(size=shape)
    in_training = (timestamps >= TRAINING[0]) & (timestamps <= TRAINING[1])
    sign = np.where(in_training, 1.0, -1.0)[:, None] * np.where(np.arange(N_SYMBOLS) < MEMBERS, 1.0, -1.0)
    volume_shock = rng.normal(size=shape)
    upper_wick = rng.normal(size=shape)
    lower_wick = rng.exponential(size=shape)
    effect = (
        INTRADAY_EFFECT * sign * z
        + VOLUME_EFFECT * volume_shock
        + UPPER_WICK_EFFECT * upper_wick
        + TAIL_EFFECT * (lower_wick > np.log(10.0))  # the exponential's top decile
    )
    # step[k] is the open-to-open log return into bar k. The label at t spans
    # the opens of t + 1 to t + 6, that is steps t + 2 to t + 6; step t + 2
    # carries bar t's shocks.
    step = 0.01 * rng.normal(size=shape)
    step[2:] += effect[:-2]
    open_ = 40.0 * np.exp(np.cumsum(step, axis=0) - step[:1])
    close = open_ * np.exp(0.01 * z)
    return {
        "adjOpen": open_,
        "adjHigh": np.maximum(open_, close) * np.exp(0.002 * (2.0 + np.tanh(upper_wick))),
        "adjLow": np.minimum(open_, close) * np.exp(-0.002 * (1.0 + lower_wick)),
        "adjClose": close,
        "adjVolume": 5e5 * np.exp(0.5 * volume_shock),
    }


def _exposures(timestamps: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """A market cap drifting around a fixed size per symbol and a fixed industry code."""
    rng = np.random.default_rng(SEED + 1)
    size = rng.normal(21.0, 1.5, size=N_SYMBOLS)
    drift = np.cumsum(0.01 * rng.normal(size=(len(timestamps), N_SYMBOLS)), axis=0)
    industry = rng.integers(0, N_INDUSTRIES, size=N_SYMBOLS).astype(float)
    return {
        "marketcap": np.exp(size + drift),
        "industry": np.tile(industry, (len(timestamps), 1)),
    }


def _write_dataset(root: Path, name: str, values: dict[str, np.ndarray], timestamps) -> StockDataset:
    """Write ``values`` as the store ``market/<name>`` and return a dataset over it."""
    panel = xr.Dataset(
        {column: (["timestamp", "symbol"], array) for column, array in values.items()},
        coords={"timestamp": timestamps, "symbol": symbols()},
    )
    zarr_path = root / "market" / name / f"{name}.zarr"
    zarr_path.parent.mkdir(parents=True)
    panel.to_zarr(zarr_path, mode="w")
    return StockDataset(DatasetConfig(
        zarr_file_path=str(zarr_path), raw_data_dir_path=str(root / "downloads"),
        market="us_equity", frequency="1d",
    ))


def build(root: Path) -> dict:
    """Write the stores under ``root`` and return the component configs a campaign config holds.

    Returns ``{"prices": ..., "label": ..., "membership": ..., "exposures": ...}``,
    each a ``get_config()`` dict.
    """
    root = Path(root)
    timestamps = pd.bdate_range(START, END)
    prices = _write_dataset(root, "synthetic_1d", _prices(timestamps), timestamps)
    exposures = _write_dataset(root, "synthetic_exposures", _exposures(timestamps), timestamps)

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
    return {
        "prices": prices.get_config(), "label": label.get_config(),
        "membership": membership.get_config(), "exposures": exposures.get_config(),
    }
