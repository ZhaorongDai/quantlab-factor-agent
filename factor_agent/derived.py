"""Derived columns: columns a daily factor's graph reads that its dataset does not hold.

A ``DerivedColumnsFactor`` is a KunQuant factor on the daily dataset whose
graph may read derived columns, filled before the graph runs by its
providers (``Provider``): period lags (``factor_agent.period_lag``), from
the daily panel. Each provider says which columns it fills, the dataset
columns it reads and the warm-up it needs, and fills the arrays; the factor
reads the longest warm-up any of them needs.
"""

from collections.abc import Iterable, Sequence

import numpy as np
import xarray as xr
from quantlab.dataset.base import BaseDataset
from quantlab.factor.kunquant import FactorKunQuant


class Provider:
    """Fills derived columns of a ``DerivedColumnsFactor``.

    ``columns`` are the derived columns it fills; ``data_columns`` the
    dataset columns it reads, ``panel_columns`` those of them it reads from
    the panel only, never handed to the graph as arrays (a date key);
    ``warmup_bars`` the bars before the first computed bar it reads.
    """

    #: The derived columns the provider fills.
    columns: tuple[str, ...] = ()
    #: The dataset columns it reads.
    data_columns: tuple[str, ...] = ()
    #: The dataset columns it reads from the panel only.
    panel_columns: tuple[str, ...] = ()
    #: The bars before the first computed bar it reads.
    warmup_bars: int = 0

    def fill(
        self, dataset: BaseDataset, panel: xr.Dataset, arrays: dict[str, np.ndarray],
        symbols: np.ndarray, timestamps: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """The arrays of ``columns``, on the axes of ``arrays`` (``timestamps`` by ``symbols``).

        ``panel`` is the dataset's panel under its shared names
        (``to_shared_names``), warm-up included; ``arrays`` its columns as the
        graph reads them.
        """
        raise NotImplementedError


def data_columns(inputs: Iterable[str], providers: Iterable[Provider]) -> tuple[str, ...]:
    """The dataset columns to read for graph ``inputs``: each derived column becomes its provider's, sorted."""
    inputs, providers = set(inputs), list(providers)
    columns = inputs.difference(*(provider.columns for provider in providers))
    for provider in providers:
        if inputs.intersection(provider.columns):
            columns |= set(provider.data_columns)
    return tuple(sorted(columns))


class DerivedColumnsFactor(FactorKunQuant):
    """A KunQuant factor whose graph may read derived columns; see the module docstring.

    A subclass says which providers fill the derived columns its graph reads
    (``_providers``) and holds their dataset columns in ``config.data_columns``
    (``data_columns``). The factor reads at least the warm-up its providers need.
    """

    def _providers(self) -> Sequence[Provider]:
        """The providers of the derived columns the graph reads; none by default."""
        return ()

    @property
    def warmup_bars(self) -> int:
        """``config.warmup_bars``, or the longest warm-up a provider needs when that is longer."""
        return max([self.config.warmup_bars, *(provider.warmup_bars for provider in self._providers())])

    def _kunquant_inputs(self, inputs: xr.Dataset) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        providers = self._providers()
        if not providers:
            return super()._kunquant_inputs(inputs)
        hidden = {column for provider in providers for column in provider.panel_columns}
        values = tuple(c for c in self.config.data_columns if c not in hidden)
        dataset = self.config.dataset
        arrays, symbols, timestamps = dataset.to_kunquant(data_columns=values, panel=inputs)
        panel = dataset.to_shared_names(inputs)
        filled = {}
        for provider in providers:
            filled |= provider.fill(dataset, panel, arrays, symbols, timestamps)
        return {**arrays, **filled}, symbols, timestamps
