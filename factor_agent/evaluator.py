"""The Evaluator: the only module that reads data.

It is fixed to one segment when it is made. ``evaluate`` compiles all of a
round's graphs as one KunQuant graph, computes it on the whole-market
dataset (the segment plus the warm-up bars before it) and scores each
candidate by its daily rank IC against the forward label, which is masked
to the members of day t, and the Newey-West t of the mean IC. Only dates
whose label lies entirely inside the segment are scored.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from KunQuant.Op import OpBase, Output
from KunQuant.Stage import Function
from quantlab.analysis.factor_report import newey_west_lags, newey_west_t_stat
from quantlab.factor.config import FactorConfig
from quantlab.factor.kunquant import FactorKunQuant
from quantlab.model.evaluation import cross_sectional_rank_ic_series

from factor_agent.gate.graph import Graph
from factor_agent.segments import Segment


@dataclass(frozen=True)
class Metrics:
    """A candidate's training-segment statistics."""

    ic_mean: float
    ic_t: float
    n_days: int
    start: str
    end: str

    def to_dict(self) -> dict:
        """Return the metrics as the ledger stores them."""
        return dict(vars(self))


@dataclass(frozen=True)
class CompileFailure:
    """The round's graph did not compile or run; the error KunQuant gave."""

    error: str


class _RoundGraph(FactorKunQuant):
    """All of a round's graphs as one factor, one output per candidate."""

    def __init__(self, config: FactorConfig, graphs: list[Graph]):
        self._graphs = graphs
        super().__init__(config)

    def _get_factor_names(self) -> tuple[str, ...]:
        return tuple(_output_name(i) for i in range(len(self._graphs)))

    def _get_factor_func(self) -> Function:
        inputs: dict[str, OpBase] = {}
        ops: list[OpBase] = []
        for i, graph in enumerate(self._graphs):
            output, new = graph.build(inputs)
            ops += new
            ops.append(Output(output, _output_name(i)))
        return Function(ops)


def _output_name(i: int) -> str:
    return f"candidate_{i}"


class Evaluator:
    """Score graphs on one segment; see the module docstring.

    Make one with ``for_segment``; nothing changes its dates afterwards.
    """

    def __init__(self, segment: Segment, prices, label, warmup_bars: int, njobs: int):
        self._segment = segment
        self._prices = prices
        self._label = label
        self._warmup_bars = warmup_bars
        self._njobs = njobs

    @classmethod
    def for_segment(cls, segment: Segment, *, prices, label, warmup_bars: int, njobs: int) -> "Evaluator":
        """Return an evaluator of ``segment``.

        Parameters
        ----------
        segment : Segment
            The only dates it reads, besides ``warmup_bars`` bars before.
        prices : quantlab dataset
            The whole-market adjusted bars the graphs run on.
        label : quantlab label
            The forward return, masked to the members of day t, read from
            its store.
        warmup_bars, njobs : int
            Warm-up bars and KunQuant executor threads.
        """
        return cls(segment, prices, label, warmup_bars, njobs)

    @property
    def segment(self) -> Segment:
        """The segment this evaluator reads."""
        return self._segment

    def evaluate(self, graphs: list[Graph]) -> list[Metrics | CompileFailure]:
        """Return one entry per graph, in order; all graphs are compiled together."""
        if not graphs:
            return []
        columns = sorted({name for graph in graphs for name in graph.input_names})
        factor = _RoundGraph(
            FactorConfig(
                warmup_bars=self._warmup_bars, dataset=self._prices, mode="batch",
                data_columns=tuple(columns), njobs=self._njobs,
            ),
            graphs,
        )
        try:
            values = factor.compute(self._segment.start, self._segment.end)
        except Exception as exc:  # KunQuant reports a bad graph as a plain exception
            return [CompileFailure(f"{type(exc).__name__}: {exc}")] * len(graphs)
        label = self._scored_label()
        values, label = xr.align(values, label, join="inner")
        return [self._metrics(values[_output_name(i)].values, label.values, label) for i in range(len(graphs))]

    def _scored_label(self) -> xr.DataArray:
        """The label on the segment's dates whose label ends inside the segment."""
        name = self._label.get_factor_names()[0]
        label = self._label.read(self._segment.start, self._segment.end)[name]
        keep = label.sizes["timestamp"] - self._label.lookahead_bars()
        return label.isel(timestamp=slice(0, max(keep, 0)))

    def _metrics(self, values: np.ndarray, target: np.ndarray, label: xr.DataArray) -> Metrics:
        ic = cross_sectional_rank_ic_series(values, target)
        scored = np.isfinite(ic)
        days = pd.DatetimeIndex(label["timestamp"].values)[scored]
        series = pd.Series(ic[scored])
        t_stat, _ = newey_west_t_stat(series, newey_west_lags(len(series), self._label.span_bars()))
        return Metrics(
            ic_mean=float(series.mean()) if len(series) else float("nan"),
            ic_t=t_stat,
            n_days=len(series),
            start=days[0].strftime("%Y-%m-%d") if len(days) else "",
            end=days[-1].strftime("%Y-%m-%d") if len(days) else "",
        )
