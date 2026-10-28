"""Session columns: intraday graphs computed on intraday bars, one value per stock and session.

``SessionFactor`` is a quantlab KunQuant factor over an intraday dataset
(one-minute trade bars cut by session). Each of its graphs ends in a session
aggregation (``factor_agent.aggregations``) and is one output; at a
session's last bar the output holds the aggregation of that session's bars
and every other bar is NaN, so resampled to ``"1d"`` with ``"last"`` the
factor is one value per stock and session::

    daily = SessionFactor(config).resample("1d", "last").compute(start, end)

KunQuant has no notion of a session, so the factor adds the session inputs
before the graph runs, as quantlab's ``IntradayStats`` computes them
(``quantlab.factor.predefined.intraday.session_inputs``): a bar's position in
its session (``session_position``, 0 for the first bar), the bars left to the
session's last (``session_bars_left``, 0 for the last) and the last trade
price at or before the bar within the session (``last_price``, from the
bars' ``close``). A graph reads them as it reads the bars' own columns.
Nothing assumes a number of bars per session: a session is the bars the
dataset cuts into one ``"1d"`` bar (``resample_labels``).

Each aggregation is expanded into KunQuant accumulators: running sums that
restart at each session's first bar and skip a missing value. An
accumulator is named after what it accumulates, the canonical hash of the
aggregation's input subgraph and the kind of sum (``<hash>_sum``,
``<hash>_count``), because KunQuant's expression folding merges two
accumulators only when their names are the same: two aggregations of one
subgraph in one function, in one graph or in two, then share their sums.

``SessionColumns`` is the provider of session columns to a daily factor
(``factor_agent.derived``): it computes the intraday graphs of a stage split
(``factor_agent.gate.stages``) with ``SessionFactor`` and puts each session's
value on its day of the daily panel.
"""

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
import xarray as xr
from KunQuant.Op import Builder, ConstantOp, Input, OpBase, Output
from KunQuant.ops import Accumulator, Equals, LessEqual, Select, SetAccumulator, SetInfOrNanToValue
from KunQuant.Stage import Function
from quantlab.factor.config import FactorConfig
from quantlab.factor.kunquant import FactorKunQuant
from quantlab.factor.predefined.intraday import LAST_PRICE, SESSION_BARS_LEFT, SESSION_POSITION, session_inputs

from factor_agent.aggregations import SessionAggregation, SessionMean
from factor_agent.gate._canonical import canonical_hashes
from factor_agent.gate.graph import Graph, Node
from factor_agent.gate.vocabulary import resolve_classes
from factor_agent.derived import Provider

#: The inputs the factor adds to the bars' own columns, computed from the bar timestamps and prices.
SESSION_INPUTS = (SESSION_POSITION, SESSION_BARS_LEFT, LAST_PRICE)
#: The bars' column the last trade price is carried from: the bar's last trade, missing on a bar without one.
PRICE = "close"


def _zero_if_missing(value: OpBase) -> OpBase:
    """Return ``value`` with NaN and infinities replaced by 0."""
    return SetInfOrNanToValue(value, 0.0)


def _present(value: OpBase) -> OpBase:
    """Return 1 where ``value`` is finite, else 0."""
    return _zero_if_missing(value * 0.0 + 1.0)


class _Expansion:
    """What an aggregation expands into: its session sums, named after their terms, over one function's inputs."""

    def __init__(self, inputs: dict[str, OpBase]):
        self._inputs = inputs

    def input(self, name: str) -> OpBase:
        """The function's ``Input`` of ``name``, added when the graphs did not read it."""
        if name not in self._inputs:
            self._inputs[name] = Input(name)
        return self._inputs[name]

    def session_sum(self, term: OpBase, name: str) -> OpBase:
        """The running sum of ``term`` over the session's bars up to the current one, a missing term adding 0.

        The sum restarts at the session's first bar; it is the KunQuant
        accumulator ``name``, which must say what it sums.
        """
        first = LessEqual(self.input(SESSION_POSITION), ConstantOp(0.0))
        value = _zero_if_missing(term)
        total = Accumulator(value, name, is_whole_time_required=True)
        always = Equals(value, value)  # the value is finite
        return SetAccumulator(total, always, Select(first, value, total + value))


def _mean(expansion: _Expansion, inputs: list[OpBase], key: str) -> OpBase:
    [x] = inputs
    total = expansion.session_sum(x, f"{key}_sum")
    count = expansion.session_sum(_present(x), f"{key}_count")
    return Select(LessEqual(count, ConstantOp(0.0)), ConstantOp("nan"), total / count)


#: How each aggregation expands: from its input operators and the key naming its sums, its value at each bar.
_EXPANSIONS: dict[type, Callable[[_Expansion, list[OpBase], str], OpBase]] = {SessionMean: _mean}


def session_function(graphs: Mapping[str, Graph]) -> Function:
    """One KunQuant function with an output per graph, named after it: its value at each session's last bar.

    Every aggregation is expanded into its session sums; the bars before a
    session's last are NaN. The graphs share their ``Input`` operators, so
    an aggregation of one subgraph in two of them sums it once.

    Raises
    ------
    ValueError
        If a graph does not end in a session aggregation.
    """
    inputs: dict[str, OpBase] = {}
    expansion = _Expansion(inputs)
    ops: list[OpBase] = []
    for name, graph in graphs.items():
        if not _is_aggregation(graph, graph.nodes[-1]):
            raise ValueError(f"SessionFactor: graph {name} does not end in a session aggregation")
        hashes = canonical_hashes(graph)
        # Graph.build makes the operators in node order, so the aggregations' keys follow it too.
        keys = iter([
            "_".join(hashes[i] for i in node.inputs)
            for node in graph.nodes if _is_aggregation(graph, node)
        ])
        output, new = graph.build(inputs)
        replaced: dict[OpBase, OpBase] = {}
        for op in new:
            op.replace_inputs(replaced)
            if not isinstance(op, SessionAggregation):
                ops.append(op)
                continue
            builder = Builder()
            with builder:
                replaced[op] = _EXPANSIONS[type(op)](expansion, op.inputs, next(keys))
            ops += builder.ops
        value = replaced.get(output, output)
        builder = Builder()
        with builder:
            last = LessEqual(expansion.input(SESSION_BARS_LEFT), ConstantOp(0.0))
            Output(Select(last, value, ConstantOp("nan")), name)
        ops += builder.ops
    return Function(ops)


def _is_aggregation(graph: Graph, node: Node) -> bool:
    """Whether ``node`` of ``graph`` is a session aggregation."""
    return issubclass((graph.classes or {}).get(node.op, object), SessionAggregation)


class SessionFactor(FactorKunQuant):
    """Intraday graphs, each ending in a session aggregation, on an intraday dataset; see the module docstring.

    The graphs, name to ``Graph.to_json()``, are ``config.kwargs["graphs"]``
    and the class path of each operator they record
    ``config.kwargs["classes"]`` (``Vocabulary.class_paths``).
    ``config.data_columns`` must hold the dataset columns they read
    (``data_columns``). Batch mode only: the session inputs are computed
    from a whole panel.
    """

    def _validate_config(self) -> None:
        super()._validate_config()
        if self.config.mode != "batch":
            raise ValueError(f"{self.class_name}: batch mode only; the session inputs are computed from a whole panel")

    def _recorded(self) -> dict:
        return dict((self.config.kwargs or {})["graphs"])

    def _graphs(self) -> dict[str, Graph]:
        classes = resolve_classes(dict((self.config.kwargs or {})["classes"]))
        return {name: Graph.from_json(graph, classes) for name, graph in self._recorded().items()}

    def _get_factor_names(self) -> tuple[str, ...]:
        return tuple(self._recorded())

    def _get_factor_func(self) -> Function:
        return session_function(self._graphs())

    def _kunquant_inputs(self, inputs: xr.Dataset) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        arrays, symbols, timestamps = super()._kunquant_inputs(inputs)
        session = np.asarray(self.config.dataset.resample_labels(np.asarray(timestamps), "1d"))
        close = arrays[PRICE].astype(np.float64)
        # Only the position, the bars left and the last price are read; the opens and auctions do not enter them.
        added = session_inputs(session, close, close, np.full_like(close, np.nan))
        arrays.update({name: added[name] for name in SESSION_INPUTS})
        read = _inputs(self._recorded()) | {SESSION_POSITION, SESSION_BARS_LEFT}
        return {name: values for name, values in arrays.items() if name in read}, symbols, timestamps

    @staticmethod
    def data_columns(graphs: Mapping[str, list]) -> tuple[str, ...]:
        """The dataset columns the recorded graphs read besides the session inputs, and the price, sorted."""
        return tuple(sorted((_inputs(graphs) - set(SESSION_INPUTS)) | {PRICE}))


def _inputs(graphs: Mapping[str, list]) -> set[str]:
    """The names of the ``Input`` nodes of recorded graphs."""
    return {dict(node[1])["name"] for graph in graphs.values() for node in graph if node[0] == "Input"}


class SessionColumns(Provider):
    """The session columns of intraday graphs, filled for a daily factor from an intraday dataset.

    ``graphs`` maps each session column to its intraday graph
    (``Graph.to_json()``, by the names ``stages.split`` gives them) and
    ``classes`` gives the class path of every operator they record
    (``Vocabulary.class_paths``). ``fill`` computes them with
    ``SessionFactor`` on ``dataset``, the intraday dataset, over the days of
    the daily axes, and puts each session's value on its day: a session's
    label is its date, and a symbol is matched by its label as text, so the
    permatickers of an integer axis find the symbols of a text one. A day
    without a session (before the intraday data begins, say) and a symbol
    the intraday dataset does not hold are NaN. It reads no column of the
    daily dataset and needs no warm-up: an intraday graph sees its own
    session only.
    """

    def __init__(self, dataset, graphs: Mapping[str, list], classes: Mapping[str, str], njobs: int = 1):
        self._dataset = dataset
        self._graphs = {name: list(graph) for name, graph in graphs.items()}
        self._classes = dict(classes)
        self._njobs = njobs
        self.columns = tuple(self._graphs)

    def fill(self, dataset, panel, arrays, symbols, timestamps) -> dict[str, np.ndarray]:
        days = pd.DatetimeIndex(timestamps).normalize()
        factor = SessionFactor(FactorConfig(
            warmup_bars=0, dataset=self._dataset, mode="batch", njobs=self._njobs,
            data_columns=SessionFactor.data_columns(self._graphs),
            kwargs={"graphs": self._graphs, "classes": self._classes},
        ))
        daily = factor.resample("1d", "last").compute(days[0].strftime("%Y-%m-%d"), days[-1].strftime("%Y-%m-%d"))
        rows = pd.DatetimeIndex(daily["timestamp"].values).normalize().get_indexer(days)
        columns = pd.Index([str(s) for s in daily["symbol"].values]).get_indexer([str(s) for s in symbols])
        found = np.ix_(rows >= 0, columns >= 0)
        read = np.ix_(rows[rows >= 0], columns[columns >= 0])
        filled = {}
        for name in self.columns:
            values = np.full((len(days), len(symbols)), np.nan, dtype=np.float32)
            values[found] = daily[name].transpose("timestamp", "symbol").values[read]
            filled[name] = values
        return filled
