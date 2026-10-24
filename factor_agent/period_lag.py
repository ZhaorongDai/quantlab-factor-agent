"""Period lags read from data: the lag columns ``PeriodLag(x, k)`` reads, computed when a factor is.

A lag column (``Vocabulary.lag_columns``) is no column of the price
dataset: ``PeriodLagFactor`` reads the field's column and the key column of
its family (the period end of the report each bar shows) and computes it
with quantlab's ``report_lags``, as ``FundamentalAlpha`` computes its report
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
from quantlab.factor.kunquant import FactorKunQuant
from quantlab.factor.predefined.fundamental_alpha import report_lags


def data_columns(inputs: Iterable[str], lags: Mapping[str, Mapping]) -> tuple[str, ...]:
    """The dataset columns to read for graph ``inputs``: each lag column becomes its field's column and key, sorted."""
    columns = set()
    for name in inputs:
        if name in lags:
            columns |= {lags[name]["column"], lags[name]["key"]}
        else:
            columns.add(name)
    return tuple(sorted(columns))


def warmup_bars(lags: Mapping[str, Mapping]) -> int:
    """The bars before the first computed bar the ``lags`` read; 0 without any."""
    return max(((int(lag["k"]) + 2) * int(lag["bars"]) for lag in lags.values()), default=0)


class PeriodLagFactor(FactorKunQuant):
    """A KunQuant factor whose graph may read lag columns, computed from its dataset; see the module docstring.

    A subclass says which lag columns its graph reads (``_period_lags``) and
    holds their fields' columns and keys in ``config.data_columns``
    (``data_columns``). The factor reads at least the warm-up its lags need.
    """

    def _period_lags(self) -> Mapping[str, Mapping]:
        """The lag columns the graph reads, each with how it is computed; none by default."""
        return {}

    @property
    def warmup_bars(self) -> int:
        """``config.warmup_bars``, or the warm-up the lags need when that is longer."""
        return max(self.config.warmup_bars, warmup_bars(self._period_lags()))

    def _kunquant_inputs(self, inputs: xr.Dataset) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        lags = self._period_lags()
        if not lags:
            return super()._kunquant_inputs(inputs)
        keys = {lag["key"] for lag in lags.values()}
        values = tuple(c for c in self.config.data_columns if c not in keys)
        arrays, symbols, timestamps = self.config.dataset.to_kunquant(data_columns=values, panel=inputs)
        panel = self.config.dataset.to_shared_names(inputs)
        groups: dict[tuple, dict[str, list]] = {}
        for name, lag in lags.items():
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
        return {**arrays, **lagged}, symbols, timestamps


def _key_values(key: xr.DataArray) -> np.ndarray:
    """The key as float: days since 1970 for dates (NaN where NaT), the numbers themselves otherwise."""
    values = key.to_numpy()
    if np.issubdtype(values.dtype, np.datetime64):
        days = values.astype("datetime64[D]").astype(np.float64)
        days[np.isnat(values)] = np.nan
        return days
    return values.astype(np.float64)
