"""A tiny synthetic intraday dataset: a few sessions of one-minute trade bars, built through quantlab's public API.

Three sessions of eight symbols: two full days of 390 bars and a half day
of 210 (2024-11-29, the day after Thanksgiving), each bar stamped at its
end on the UTC clock (14:31 to 21:00, or 18:00 on the half day), as the
Massive trade bars are. Prices follow a random walk; about one bar in ten
has no trade, where the prices are missing and the volumes zero. Besides:

- symbol ``S1`` has no trade in the first ``LATE_OPEN`` bars of every session,
  so nothing of a session's start may come from the session before;
- symbol ``S2`` has no trade at all on the second session (``EMPTY_DAY``).

``minute_bars`` gives the panel and ``intraday_dataset`` the quantlab
dataset over it (``FrameDataset``), so a test can change a panel's bars and
compute on the changed one.
"""

import numpy as np
import pandas as pd
import xarray as xr
from quantlab.dataset.memory import FrameDataset

DAYS = ("2024-11-26", "2024-11-27", "2024-11-29")
#: The half day: the market closes at 13:00 New York time.
HALF_DAY = "2024-11-29"
SYMBOLS = tuple(f"S{i}" for i in range(8))
#: The symbol without a trade in the first ``LATE_OPEN`` bars of each session.
LATE_SYMBOL, LATE_OPEN = "S1", 5
#: The symbol without a trade on ``EMPTY_DAY``.
EMPTY_SYMBOL, EMPTY_DAY = "S2", "2024-11-27"
#: Price columns, missing on a bar without a trade; volume columns, zero there.
PRICES = ("open", "high", "low", "close")
VOLUMES = ("volume", "n_trades")


def session_bars(day: str) -> int:
    """The bars of the session of ``day``: 210 on the half day, else 390."""
    return 210 if day == HALF_DAY else 390


def minute_bars(seed: int = 0) -> xr.Dataset:
    """The bars of every session on ``(timestamp, symbol)``; see the module docstring."""
    rng = np.random.default_rng(seed)
    stamps = []
    for day in DAYS:
        open_at = pd.Timestamp(day) + pd.Timedelta("14h30min")
        stamps += list(open_at + pd.Timedelta("1min") * np.arange(1, session_bars(day) + 1))
    stamps = pd.DatetimeIndex(stamps)
    shape = (len(stamps), len(SYMBOLS))
    close = 50.0 * np.exp(np.cumsum(rng.normal(0, 1e-3, shape), axis=0)) * np.arange(1, len(SYMBOLS) + 1)
    traded = rng.random(shape) >= 0.1
    position = np.concatenate([np.arange(session_bars(day)) for day in DAYS])
    traded[position < LATE_OPEN, SYMBOLS.index(LATE_SYMBOL)] = False
    traded[stamps.normalize() == pd.Timestamp(EMPTY_DAY), SYMBOLS.index(EMPTY_SYMBOL)] = False
    open_ = close * (1 + rng.normal(0, 2e-4, shape))
    spread = np.abs(rng.normal(0, 3e-4, shape)) * close
    volume = rng.integers(100, 1000, shape).astype(float)
    data = {
        "open": open_,
        "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread,
        "close": close,
        "volume": volume,
        "n_trades": np.ceil(volume / 100),
    }
    for name in PRICES:
        data[name] = np.where(traded, data[name], np.nan)
    for name in VOLUMES:
        data[name] = np.where(traded, data[name], 0.0)
    return xr.Dataset(
        {name: (("timestamp", "symbol"), values) for name, values in data.items()},
        coords={"timestamp": stamps, "symbol": list(SYMBOLS)},
    )


def intraday_dataset(panel: xr.Dataset | None = None) -> FrameDataset:
    """The quantlab dataset over ``panel``, ``minute_bars()`` by default."""
    return FrameDataset(minute_bars() if panel is None else panel)


def session_of(panel: xr.Dataset, day: str) -> np.ndarray:
    """Whether each bar of ``panel`` is in the session of ``day``."""
    return pd.DatetimeIndex(panel["timestamp"].values).normalize() == pd.Timestamp(day)
