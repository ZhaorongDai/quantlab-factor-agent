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

Each aggregation is expanded into KunQuant accumulators: running values
that restart at each session's first bar and skip a missing value, sums
(the count, the sum, sums of powers) or folds (the first, last, least and
greatest value present). Moments are sums of the values less the session's
first value, as quantlab's session statistics compute them, so that in
float32 a price's deviations are not lost to its level. An accumulator is
named after what it accumulates, the hash of what the aggregation's input
subgraph computes (``computation_hashes``; of each input's, joined, for
two) and the kind
(``<hash>_count``, ``<hash>_first``, ``<hash>_shifted_sumsq``...): one
function builds one accumulator per name, and KunQuant's expression folding
merges two accumulators only when their names are the same, so two
aggregations of one subgraph in one function, in one graph or in two, share
their sums (``SessionMean(x)`` and ``SessionStd(x)``: the count, the first
value and the shifted sums of ``x`` and of its square).

All the graphs of one factor are one KunQuant function, and an operator
that computes what an earlier one computes (in the same graph or another:
the same ``computation_hashes``) is that operator, so a subtree two
candidates share is computed once, and a graph's values are the same
whichever graphs it is computed with.

``SessionColumns`` is the provider of session columns to a daily factor
(``factor_agent.derived``): it computes the intraday graphs of a stage split
(``factor_agent.gate.stages``) with ``SessionFactor``, in chunks of sessions,
and puts each session's value on its day of the daily panel.
"""

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
import xarray as xr
from KunQuant.Driver import compileit
from KunQuant.Op import Builder, ConstantOp, Input, OpBase, Output
from KunQuant.ops import (
    Accumulator, And, Equals, LessEqual, LessThan, Max, Min, Not, Select, SetAccumulator, SetInfOrNanToValue, Sqrt,
)
from KunQuant.Stage import Function
from quantlab.factor.chunking import ChunkedBuilder
from quantlab.factor.config import FactorConfig
from quantlab.factor.kunquant import BATCH_OPTIONS, FactorKunQuant
from quantlab.factor.predefined.intraday import LAST_PRICE, SESSION_BARS_LEFT, SESSION_POSITION, session_inputs

from factor_agent.aggregations import (
    SessionAggregation, SessionCorr, SessionCount, SessionFirst, SessionLast, SessionMax, SessionMean, SessionMin,
    SessionSkew, SessionStd, SessionSum, SessionWeightedMean,
)
from factor_agent.derived import Provider
from factor_agent.gate._canonical import computation_hashes
from factor_agent.gate.graph import Graph, Node
from factor_agent.gate.vocabulary import resolve_classes

#: The inputs the factor adds to the bars' own columns, computed from the bar timestamps and prices.
SESSION_INPUTS = (SESSION_POSITION, SESSION_BARS_LEFT, LAST_PRICE)
#: The bars' column the last trade price is carried from: the bar's last trade, missing on a bar without one.
PRICE = "close"
#: The bytes per bar and symbol of the arrays ``session_inputs`` computes the session inputs through: the float64
#: price and closing-auction price it is given, its index, mask and price arrays, and the six float32 inputs it
#: returns, counted from its code (about 150 bytes) and rounded up.
SESSION_INPUT_WORK = 160


def _zero_if_missing(value: OpBase) -> OpBase:
    """Return ``value`` with NaN and infinities replaced by 0."""
    return SetInfOrNanToValue(value, 0.0)


def _present(value: OpBase) -> OpBase:
    """Return 1 where ``value`` is finite, else 0."""
    return _zero_if_missing(value * 0.0 + 1.0)


def _missing(value: OpBase) -> OpBase:
    """Whether ``value`` is NaN or infinite."""
    return LessEqual(_present(value), ConstantOp(0.0))


def _beside(column: OpBase, value: OpBase) -> OpBase:
    """``column``, computed in the stage of ``value``.

    KunQuant keeps a comparison in the stage of what it compares and cannot
    pass its result to another stage, where it is a mask, not a number: a
    mask of the session position must be made beside what it selects.
    """
    return column + _zero_if_missing(value) * 0.0


def _nan_unless(condition: OpBase, value: OpBase) -> OpBase:
    """Return ``value`` where ``condition`` holds, else NaN."""
    return Select(condition, value, ConstantOp("nan"))


class _Expansion:
    """What an aggregation expands into: its running values, named after what they hold, over one function's inputs."""

    def __init__(self, inputs: dict[str, OpBase]):
        self._inputs = inputs
        self._running: dict[str, OpBase] = {}

    def input(self, name: str) -> OpBase:
        """The function's ``Input`` of ``name``, added when the graphs did not read it."""
        if name not in self._inputs:
            self._inputs[name] = Input(name)
        return self._inputs[name]

    def running(
        self, name: str, step: Callable[[OpBase], OpBase], reads: OpBase, start: float | str = 0.0,
    ) -> OpBase:
        """The running value ``name`` over the session's bars up to the current one: ``step`` of its value before.

        ``step`` reads ``reads`` besides the value before, which is
        ``start`` before the session's first bar. It is the KunQuant
        accumulator ``name``, which must say what it holds: a second call
        under the same name in one function gives the first's value,
        whatever its ``step``.
        """
        if name not in self._running:
            # KunQuant needs an accumulator and its update in one stage, and places the accumulator with its
            # input: on the position alone, it would come before a cross-sectional operator whose values it sums.
            total = Accumulator(reads, name, is_whole_time_required=True)
            position = _beside(self.input(SESSION_POSITION), reads)
            first = LessEqual(position, ConstantOp(0.0))
            always = Equals(position, position)  # the position is finite
            self._running[name] = SetAccumulator(total, always, step(Select(first, ConstantOp(start), total)))
        return self._running[name]

    def session_sum(self, term: OpBase, name: str) -> OpBase:
        """The running sum of ``term`` over the session's bars up to the current one, a missing term adding 0."""
        value = _zero_if_missing(term)
        return self.running(name, lambda before: before + value, value)

    def session_fold(self, x: OpBase, name: str, combine: Callable[[OpBase, OpBase], OpBase]) -> OpBase:
        """``combine(held, x)`` over the values of ``x`` present in the session so far; NaN before the first."""
        def step(held: OpBase) -> OpBase:
            return Select(_missing(x), held, Select(_missing(held), x, combine(held, x)))
        return self.running(name, step, x, "nan")

    def first(self, x: OpBase, key: str) -> OpBase:
        """The first value of ``x`` present in the session so far, NaN before it: ``SessionFirst``'s and the shift."""
        return self.session_fold(x, f"{key}_first", lambda held, value: held)


def _count(expansion: _Expansion, inputs: list[OpBase], keys: list[str]) -> OpBase:
    [x], [key] = inputs, keys
    return expansion.session_sum(_present(x), f"{key}_count")


def _sum(expansion: _Expansion, inputs: list[OpBase], keys: list[str]) -> OpBase:
    [x], [key] = inputs, keys
    return expansion.session_sum(x, f"{key}_sum")


def _shifted_sums(expansion: _Expansion, x: OpBase, key: str, powers: int) -> list[OpBase]:
    """The session sums of the first ``powers`` powers of ``x`` less its session's first value.

    A price's deviations in a session are orders below its level: float32
    sums of the prices themselves would lose them to cancellation, sums of
    the values shifted by the session's first value keep them (as
    quantlab's session statistics do).
    """
    shifted = x - expansion.first(x, key)
    terms = {
        "shifted_sum": shifted, "shifted_sumsq": shifted * shifted, "shifted_sumcube": shifted * shifted * shifted,
    }
    return [expansion.session_sum(term, f"{key}_{kind}") for kind, term in list(terms.items())[:powers]]


def _mean(expansion: _Expansion, inputs: list[OpBase], keys: list[str]) -> OpBase:
    [x], [key] = inputs, keys
    count = _count(expansion, inputs, keys)
    [s1] = _shifted_sums(expansion, x, key, 1)
    return _nan_unless(LessThan(ConstantOp(0.0), count), expansion.first(x, key) + s1 / count)


def _std(expansion: _Expansion, inputs: list[OpBase], keys: list[str]) -> OpBase:
    [x], [key] = inputs, keys
    n = _count(expansion, inputs, keys)
    s1, s2 = _shifted_sums(expansion, x, key, 2)
    m2 = Max(s2 - s1 * s1 / n, ConstantOp(0.0))
    # The unbiased standard deviation, as pandas and KunQuant's windows give it.
    return _nan_unless(LessThan(ConstantOp(1.0), n), Sqrt(m2 / (n - 1.0)))


def _skew(expansion: _Expansion, inputs: list[OpBase], keys: list[str]) -> OpBase:
    [x], [key] = inputs, keys
    n = _count(expansion, inputs, keys)
    s1, s2, s3 = _shifted_sums(expansion, x, key, 3)
    mean = s1 / n
    m2 = s2 - s1 * mean
    m3 = s3 - mean * s2 * 3.0 + mean * mean * s1 * 2.0
    # The unbiased skewness, as pandas and KunQuant's windows give it.
    skew = Sqrt(n * (n - 1.0)) / (n - 2.0) * Sqrt(n) * m3 / (m2 * Sqrt(m2))
    return _nan_unless(And(LessThan(ConstantOp(2.0), n), LessThan(ConstantOp(0.0), m2)), skew)


def _first(expansion: _Expansion, inputs: list[OpBase], keys: list[str]) -> OpBase:
    [x], [key] = inputs, keys
    return expansion.first(x, key)


def _last(expansion: _Expansion, inputs: list[OpBase], keys: list[str]) -> OpBase:
    [x], [key] = inputs, keys
    return expansion.session_fold(x, f"{key}_last", lambda held, value: value)


def _min(expansion: _Expansion, inputs: list[OpBase], keys: list[str]) -> OpBase:
    [x], [key] = inputs, keys
    return expansion.session_fold(x, f"{key}_min", Min)


def _max(expansion: _Expansion, inputs: list[OpBase], keys: list[str]) -> OpBase:
    [x], [key] = inputs, keys
    return expansion.session_fold(x, f"{key}_max", Max)


def _weighted_mean(expansion: _Expansion, inputs: list[OpBase], keys: list[str]) -> OpBase:
    (x, w), key = inputs, "_".join(keys)
    both = _present(x) * _present(w)
    weight = expansion.session_sum(w * both, f"{key}_weight")
    shifted = expansion.session_sum((x - expansion.first(x, keys[0])) * w * both, f"{key}_shifted_weighted_sum")
    return _nan_unless(Not(Equals(weight, ConstantOp(0.0))), expansion.first(x, keys[0]) + shifted / weight)


def _corr(expansion: _Expansion, inputs: list[OpBase], keys: list[str]) -> OpBase:
    key = "_".join(keys)
    both = _present(inputs[0]) * _present(inputs[1])
    # Each input less its first value present in the session, which comes no later than the pair's first bar.
    x, y = ((value - expansion.first(value, k)) * both for value, k in zip(inputs, keys))
    n = expansion.session_sum(both, f"{key}_count")
    sx, sy = (expansion.session_sum(v, f"{key}_shifted_sum_{i}") for i, v in enumerate((x, y)))
    sxx, syy = (expansion.session_sum(v * v, f"{key}_shifted_sumsq_{i}") for i, v in enumerate((x, y)))
    sxy = expansion.session_sum(x * y, f"{key}_shifted_cross")
    xx, yy, xy = sxx - sx * sx / n, syy - sy * sy / n, sxy - sx * sy / n
    return _nan_unless(And(LessThan(ConstantOp(0.0), xx), LessThan(ConstantOp(0.0), yy)), xy / Sqrt(xx * yy))


#: How each aggregation expands: from its input operators and the keys naming their sums (the computation hash of
#: each input's subgraph), its value at each bar.
_EXPANSIONS: dict[type, Callable[[_Expansion, list[OpBase], list[str]], OpBase]] = {
    SessionMean: _mean, SessionSum: _sum, SessionCount: _count, SessionStd: _std, SessionSkew: _skew,
    SessionFirst: _first, SessionLast: _last, SessionMin: _min, SessionMax: _max,
    SessionWeightedMean: _weighted_mean, SessionCorr: _corr,
}


def session_function(graphs: Mapping[str, Graph]) -> Function:
    """One KunQuant function with an output per graph, named after it: its value at each session's last bar.

    Every aggregation is expanded into its session sums; the bars before a
    session's last are NaN. A subtree is computed once however often the
    graphs hold it: an operator computing what an earlier one of any graph
    computes is that operator (``Mul(close, volume)`` and
    ``Mul(volume, close)`` are one, ``x + 1e8 - 1e8`` and ``x`` two), so an
    aggregation of one subgraph in two graphs sums it once. Operators in or after a loop are kept as
    built, a loop's body being its own.

    Raises
    ------
    ValueError
        If a graph does not end in a session aggregation.
    """
    inputs: dict[str, OpBase] = {}
    expansion = _Expansion(inputs)
    ops: list[OpBase] = []
    # The operator computing each computation hash, over every graph so far.
    built: dict[str, OpBase] = {}
    for name, graph in graphs.items():
        if not _is_aggregation(graph, graph.nodes[-1]):
            raise ValueError(f"SessionFactor: graph {name} does not end in a session aggregation")
        hashes = computation_hashes(graph)
        shared = _shareable(graph)
        # Graph.build makes an operator per node other than an Input, in node order.
        nodes = iter([i for i, node in enumerate(graph.nodes) if node.op != "Input"])
        output, new = graph.build(inputs)
        replaced: dict[OpBase, OpBase] = {}
        for op in new:
            if isinstance(op, Input):
                ops.append(op)
                continue
            i = next(nodes)
            if shared[i] and hashes[i] in built:
                replaced[op] = built[hashes[i]]
                continue
            op.replace_inputs(replaced)
            if isinstance(op, SessionAggregation):
                keys = [hashes[j] for j in graph.nodes[i].inputs]
                builder = Builder()
                with builder:
                    replaced[op] = _EXPANSIONS[type(op)](expansion, op.inputs, keys)
                ops += builder.ops
            else:
                ops.append(op)
            if shared[i]:
                built[hashes[i]] = replaced.get(op, op)
        value = replaced.get(output, output)
        builder = Builder()
        with builder:
            last = LessEqual(_beside(expansion.input(SESSION_BARS_LEFT), value), ConstantOp(0.0))
            Output(Select(last, value, ConstantOp("nan")), name)
        ops += builder.ops
    return Function(ops)


def _shareable(graph: Graph) -> list[bool]:
    """Whether each node of ``graph`` may be taken from another graph: neither a loop, in one nor after one."""
    loops = {node.parent for node in graph.nodes if node.parent is not None}
    shareable: list[bool] = []
    for i, node in enumerate(graph.nodes):
        shareable.append(i not in loops and node.parent is None and all(shareable[j] for j in node.inputs))
    return shareable


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

    def cell_bytes(self) -> tuple[int, int]:
        """Return the bytes one bar of one symbol takes: at peak, and as output; see ``Factor.cell_bytes``.

        The peak counts, at 8 bytes each as ``FactorKunQuant`` does, the
        columns read, the session inputs, every array KunQuant holds for
        each bar between two of its stages (a cross-sectional operator's
        values, say: the ``TEMP`` buffers of the generated module) and the
        outputs, and besides them the arrays ``session_inputs`` computes
        through (``SESSION_INPUT_WORK``), as if all were held at once.
        """
        output = 8 * self.num_factors
        source = "\n".join(compileit(
            self._get_factor_func(), self.class_name, input_layout="TS", output_layout="TS",
            options=dict(BATCH_OPTIONS),
        ))
        held = len(self._input_variables()) + len(SESSION_INPUTS) + source.count("BufferKind::TEMP")
        return 8 * held + output + SESSION_INPUT_WORK, output

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

    The graphs run as one ``SessionFactor``, so a subtree several of them
    hold is computed once, and in chunks of whole sessions through
    quantlab's ``ChunkedBuilder``: only one chunk's bars are in memory at a
    time, beside the daily values. ``granularity`` is the chunks' calendar
    period (``"month"``, ``"day"``...); ``None`` computes the days whole
    when they fit in the memory available (``SessionFactor.cell_bytes``)
    and otherwise in the coarsest period whose chunks do. The values do not depend on it. The dataset must
    read a store (``stored_symbols``), as a campaign's intraday dataset
    does.
    """

    def __init__(
        self, dataset, graphs: Mapping[str, list], classes: Mapping[str, str], njobs: int = 1,
        granularity: str | None = None,
    ):
        self._dataset = dataset
        self._graphs = {name: list(graph) for name, graph in graphs.items()}
        self._classes = dict(classes)
        self._njobs = njobs
        self._granularity = granularity
        self.columns = tuple(self._graphs)

    def fill(self, dataset, panel, arrays, symbols, timestamps) -> dict[str, np.ndarray]:
        days = pd.DatetimeIndex(timestamps).normalize()
        factor = SessionFactor(FactorConfig(
            warmup_bars=0, dataset=self._dataset, mode="batch", njobs=self._njobs,
            data_columns=SessionFactor.data_columns(self._graphs),
            kwargs={"graphs": self._graphs, "classes": self._classes},
        ))
        builder = ChunkedBuilder(factor.resample("1d", "last"), granularity=self._granularity)
        daily = builder.compute(days[0].strftime("%Y-%m-%d"), days[-1].strftime("%Y-%m-%d"))
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
