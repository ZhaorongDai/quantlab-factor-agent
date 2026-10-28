"""A small synthetic market with one-minute bars beside its daily bars and a signal only the minute bars show.

Forty symbols over nine months of 2020: adjusted daily bars, a membership
panel (every symbol a member throughout), the 5-day open-to-open label read
on the members (quantlab's ``Return`` through ``MembershipMaskedLabel``),
and a store of one-minute trade bars from ``MINUTE_START`` on, short
sessions of ``SESSION_BARS`` bars (``HALF_DAY`` has half as many) stamped at
their end on the UTC clock from 14:31, in the columns of Massive's trade bars
that ``configs/fields/massive_minute_bars.yaml`` reads. About one bar in ten
has no trade: its prices are missing and its volumes zero.

The planted signal is the share of each session's volume traded in its last
``LATE_BARS`` bars, a daily shock ``z``: the open-to-open step into bar
t + 2 moves with the mean of ``z`` over the five sessions up to t. The daily
bars are drawn apart from ``z``, so only a candidate computing on the minute
bars (the late share, averaged over five days) sees it.
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

START, END = "2020-01-02", "2020-09-30"
TRAINING = ("2020-03-02", "2020-08-31")
#: The first session of minute bars: the daily warm-up before it has none.
MINUTE_START = "2020-02-03"
N_SYMBOLS = 40
HORIZON = 5
SEED = 11
#: The bars of a session, and of the half day.
SESSION_BARS = 20
HALF_DAY = "2020-07-02"
#: The last bars of a session whose share of its volume is the signal.
LATE_BARS = 5
#: The usual late share, and how far one unit of the shock moves it.
LATE_SHARE, LATE_SPREAD = 0.3, 0.08
#: Open-to-open step, as a log return, per unit of the five-session mean of the shock.
EFFECT = 0.01
COVERAGE_START = "2019-01-02"
#: The columns of the minute bars: prices missing on a bar without a trade, volumes zero there.
PRICES = ("open", "high", "low", "close")
VOLUMES = ("volume", "dollar_volume", "n_trades", "buy_volume", "sell_volume")


class AllMembers(IndexConstituentDataset):
    """Every symbol a member from ``COVERAGE_START`` on; a rebuilt one reads its store."""

    def _pit_coverage_start(self, config) -> str:
        return COVERAGE_START

    def _build_intervals(self) -> pl.DataFrame:
        return pl.DataFrame([(s, COVERAGE_START, None) for s in symbols()], schema=["symbol", "start_date", "end_date"], orient="row")


def symbols() -> list[str]:
    return [f"S{i:03d}" for i in range(N_SYMBOLS)]


def days() -> pd.DatetimeIndex:
    return pd.bdate_range(START, END)


def _shock(days: pd.DatetimeIndex) -> np.ndarray:
    """The late-share shock of every day and symbol."""
    return np.random.default_rng(SEED).normal(size=(len(days), N_SYMBOLS))


def _daily(days: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED + 1)
    shape = (len(days), N_SYMBOLS)
    z = _shock(days)
    signal = pd.DataFrame(z).rolling(5, min_periods=5).mean().fillna(0.0).to_numpy()
    # step[k] is the open-to-open log return into bar k; the label at t starts with step t + 2.
    step = 0.01 * rng.normal(size=shape)
    step[2:] += EFFECT * signal[:-2]
    open_ = 40.0 * np.exp(np.cumsum(step, axis=0))
    close = open_ * np.exp(0.01 * rng.normal(size=shape))
    return {
        "adjOpen": open_,
        "adjHigh": np.maximum(open_, close) * np.exp(0.004 * rng.random(shape)),
        "adjLow": np.minimum(open_, close) * np.exp(-0.004 * rng.random(shape)),
        "adjClose": close,
        "adjVolume": 5e5 * np.exp(0.5 * rng.normal(size=shape)),
    }


def session_bars(day: pd.Timestamp) -> int:
    """The bars of the session of ``day``: half of ``SESSION_BARS`` on the half day."""
    return SESSION_BARS // 2 if day == pd.Timestamp(HALF_DAY) else SESSION_BARS


def minute_bars() -> xr.Dataset:
    """The minute bars of every session from ``MINUTE_START`` on, on ``(timestamp, symbol)``."""
    all_days = days()
    z = _shock(all_days)
    rng = np.random.default_rng(SEED + 2)
    stamps, late, sessions = [], [], []
    for d, day in enumerate(all_days):
        if day < pd.Timestamp(MINUTE_START):
            continue
        n = session_bars(day)
        stamps += list(day + pd.Timedelta("14h30min") + pd.Timedelta("1min") * np.arange(1, n + 1))
        late += [np.arange(n) >= n - LATE_BARS]
        sessions += [np.full(n, d)]
    stamps, late, sessions = pd.DatetimeIndex(stamps), np.concatenate(late), np.concatenate(sessions)
    shape = (len(stamps), N_SYMBOLS)
    # Each bar's volume: the session's late share spread over its last bars, the rest over the others.
    share = np.clip(LATE_SHARE + LATE_SPREAD * z[sessions], 0.05, 0.95)
    weight = np.where(late[:, None], share / LATE_BARS, (1 - share) / (SESSION_BARS - LATE_BARS))
    volume = np.round(1e4 * weight * np.exp(0.1 * rng.normal(size=shape)))
    close = 40.0 * np.exp(np.cumsum(1e-3 * rng.normal(size=shape), axis=0))
    traded = rng.random(shape) >= 0.1
    open_ = close * (1 + rng.normal(0, 2e-4, shape))
    spread = np.abs(rng.normal(0, 3e-4, shape)) * close
    buy = np.round(volume * rng.uniform(0.3, 0.6, shape))
    data = {
        "open": open_, "high": np.maximum(open_, close) + spread, "low": np.minimum(open_, close) - spread,
        "close": close, "volume": volume, "dollar_volume": volume * close, "n_trades": np.ceil(volume / 100),
        "buy_volume": buy, "sell_volume": np.round((volume - buy) * 0.9),
    }
    for name in PRICES:
        data[name] = np.where(traded, data[name], np.nan)
    for name in VOLUMES:
        data[name] = np.where(traded, data[name], 0.0)
    return xr.Dataset(
        {name: (("timestamp", "symbol"), values) for name, values in data.items()},
        coords={"timestamp": stamps, "symbol": symbols()},
    )


def _store(root: Path, name: str, panel: xr.Dataset, frequency: str) -> StockDataset:
    """Write ``panel`` as the store ``market/<name>`` and return a dataset over it."""
    zarr_path = root / "market" / name / f"{name}.zarr"
    zarr_path.parent.mkdir(parents=True)
    panel.to_zarr(zarr_path, mode="w")
    return StockDataset(DatasetConfig(
        zarr_file_path=str(zarr_path), raw_data_dir_path=str(root / "downloads"),
        market="us_equity", frequency=frequency,
    ))


def build(root: Path) -> dict:
    """Write the stores under ``root`` and return the component configs a campaign config holds.

    Returns ``{"prices": ..., "label": ..., "membership": ..., "intraday": ...}``, each a ``get_config()`` dict.
    """
    root = Path(root)
    timestamps = days()
    daily = xr.Dataset(
        {column: (["timestamp", "symbol"], values) for column, values in _daily(timestamps).items()},
        coords={"timestamp": timestamps, "symbol": symbols()},
    )
    prices = _store(root, "synthetic_1d", daily, "1d")
    minutes = _store(root, "synthetic_1m", minute_bars(), "1m")
    membership = AllMembers(ConstituentDatasetConfig(
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
        "membership": membership.get_config(), "intraday": minutes.get_config(),
    }
