"""Period lags read from data: the lag columns ``PeriodLag(x, k)`` reads, computed when a factor is.

A lag column (``Vocabulary.lag_columns``) is no column of the price
dataset but a derived column (``factor_agent.derived``): its provider,
``PeriodLags``, reads the field's column and the key column of its family
(the period end of the report each bar shows) and computes it with
quantlab's ``report_lags``, as ``FundamentalAlpha`` computes its report
lags. On a bar showing report r, the value ``k`` periods ago is the last
value shown during the k-th earlier report shown; it is missing unless that
report's period ends ``k * days`` (within ``tolerance_days``) before the
current one, so a skipped or missing report never passes for an earlier
period. A key that is a date counts in days; a numeric key is used as it
is.

A lag of ``k`` periods reads ``(k + 2) * bars`` bars before the first bar it
computes: the k earlier periods, the current one and one more for a late
filing.

Each lag column is described by a mapping, as ``Vocabulary.lag_columns``
gives it: ``column`` (the field's dataset column), ``k``, ``key``,
``days``, ``tolerance_days`` and ``bars``.
"""

from collections.abc import Iterable, Mapping

import numpy as np
import xarray as xr
from quantlab.dataset.base import BaseDataset
from quantlab.factor.predefined.fundamental_alpha import report_lags

from factor_agent import derived


def read(inputs: Iterable[str], lag_columns: Mapping[str, Mapping]) -> dict[str, dict]:
    """The entries of ``lag_columns`` (``Vocabulary.lag_columns``) for the columns ``inputs``, by sorted name."""
    return {name: dict(lag_columns[name]) for name in sorted(inputs) if name in lag_columns}


def warmup_bars(lags: Mapping[str, Mapping]) -> int:
    """The bars before the first computed bar the ``lags`` read; 0 without any."""
    return max(((int(lag["k"]) + 2) * int(lag["bars"]) for lag in lags.values()), default=0)


class PeriodLags(derived.Provider):
    """The lag columns ``lags`` (``read``), filled from the daily panel; see the module docstring.

    It reads the fields' columns and keys (``data_columns``), the keys from
    the panel only, and the warm-up the lags need (``warmup_bars``).
    """

    def __init__(self, lags: Mapping[str, Mapping]):
        self._lags = {name: dict(lag) for name, lag in lags.items()}
        self.columns = tuple(self._lags)
        self.data_columns = tuple(sorted({c for lag in self._lags.values() for c in (lag["column"], lag["key"])}))
        self.panel_columns = tuple(sorted({lag["key"] for lag in self._lags.values()}))
        self.warmup_bars = warmup_bars(self._lags)

    def fill(
        self, dataset: BaseDataset, panel: xr.Dataset, arrays: dict[str, np.ndarray],
        symbols: np.ndarray, timestamps: np.ndarray,
    ) -> dict[str, np.ndarray]:
        groups: dict[tuple, dict[str, list]] = {}
        for name, lag in self._lags.items():
            group = groups.setdefault((lag["key"], float(lag["days"]), float(lag["tolerance_days"])), {})
            group.setdefault(lag["column"], []).append((int(lag["k"]), name))
        lagged = {}
        for (key, days, tolerance), wanted in groups.items():
            periods = _key_values(panel[key].sel(timestamp=timestamps, symbol=symbols).transpose("timestamp", "symbol"))
            out = report_lags(
                {column: arrays[column] for column in wanted}, periods,
                {column: tuple(k for k, _ in ks) for column, ks in wanted.items()}, days, tolerance,
            )
            for column, ks in wanted.items():
                for k, name in ks:
                    lagged[name] = out[f"{column}__q{k}"]
        return lagged


def providers(lags: Mapping[str, Mapping]) -> list[derived.Provider]:
    """The provider of the lag columns ``lags``, in a list; none without any."""
    return [PeriodLags(lags)] if lags else []


def _key_values(key: xr.DataArray) -> np.ndarray:
    """The key as float: days since 1970 for dates (NaN where NaT), the numbers themselves otherwise."""
    values = key.to_numpy()
    if np.issubdtype(values.dtype, np.datetime64):
        days = values.astype("datetime64[D]").astype(np.float64)
        days[np.isnat(values)] = np.nan
        return days
    return values.astype(np.float64)
