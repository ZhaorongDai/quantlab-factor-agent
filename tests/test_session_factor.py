"""Session columns from intraday bars: ``SessionFactor`` on the synthetic minute dataset of ``synthetic_intraday``.

Candidate code is recorded through a real Gate, under a vocabulary of the
intraday bars' columns and session inputs and the example operator file, so
an aggregation is the one node the Gate records; the factor computes each
graph on the minute bars and reads one value per stock and session.
Expected values come from a few lines of numpy per session.
"""

import re

import numpy as np
import pandas as pd
import pytest
import yaml
from KunQuant.Driver import compileit
from KunQuant.Op import Builder, ConstantOp, Input, Output
from KunQuant.ops import Accumulator, Equals, LessEqual, Select, SetAccumulator, SetInfOrNanToValue
from KunQuant.Stage import Function
from quantlab.factor.config import FactorConfig
from quantlab.factor.kunquant import BATCH_OPTIONS, run_batch_graph
from quantlab.factor.kunquant_compile import CACHE_ENV

from factor_agent.candidate import Candidate
from factor_agent.config import GateConfig
from factor_agent.gate import Gate, Passed, Rejected
from factor_agent.gate.graph import Graph
from factor_agent.gate.vocabulary import Vocabulary
from factor_agent.session import SESSION_INPUTS, SessionFactor
from factor_agent.zoo import Zoo
from tests.vocabulary_files import EXAMPLE_OPERATORS
from tests.synthetic_intraday import (
    DAYS, EMPTY_DAY, EMPTY_SYMBOL, HALF_DAY, LATE_SYMBOL, SYMBOLS, intraday_dataset, minute_bars,
    session_bars, session_of,
)

FIELDS = {
    "family": "minute_bars",
    "description": "one-minute trade bars",
    "intraday": {"windows": [5]},
    "fields": [
        {"name": "close", "column": "close", "unit": "dollars/shares", "description": "last trade price of the bar"},
        {"name": "volume", "column": "volume", "unit": "shares", "description": "shares traded in the bar"},
        {"name": "session_position", "column": "session_position", "unit": "1",
         "description": "the bar's position in its session, 0 for the first"},
        {"name": "session_bars_left", "column": "session_bars_left", "unit": "1",
         "description": "bars from this bar to the session's last, 0 for the last"},
        {"name": "last_price", "column": "last_price", "unit": "dollars/shares",
         "description": "the last trade price at or before the bar, within the session"},
    ],
}

#: Each aggregation of one input and what it measures of a price, by the example operator file's dimension rules.
AGGREGATIONS = {
    "SessionMean": "price", "SessionSum": "price", "SessionStd": "price", "SessionMin": "price",
    "SessionMax": "price", "SessionFirst": "price", "SessionLast": "price", "SessionSkew": "pure number",
    "SessionCount": "pure number",
}


@pytest.fixture(scope="module")
def vocabulary(tmp_path_factory):
    root = tmp_path_factory.mktemp("session_factor")
    (root / "fields.yaml").write_text(yaml.safe_dump(FIELDS))
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(CACHE_ENV, str(root / "kunquant_cache"))
        yield Vocabulary.load([root / "fields.yaml"], EXAMPLE_OPERATORS)


@pytest.fixture(scope="module")
def panel():
    return minute_bars()


def _check(vocabulary, code, gate=None):
    return (gate or Gate(GateConfig(timeout_seconds=10.0), vocabulary)).check(
        Candidate(hypothesis="Intraday prices.", mechanism_type="behavioural_bias",
                  horizon_rationale="A week.", description="Test.", code=code),
        Zoo(),
    )


def _graph(vocabulary, code):
    result = _check(vocabulary, code)
    assert isinstance(result, Passed), getattr(result, "reasons", None)
    return result.graph


def _factor(vocabulary, codes: dict[str, str], panel=None) -> SessionFactor:
    graphs = {name: _graph(vocabulary, code).to_json() for name, code in codes.items()}
    return SessionFactor(FactorConfig(
        warmup_bars=0, dataset=intraday_dataset(panel), mode="batch", njobs=1,
        data_columns=SessionFactor.data_columns(graphs),
        kwargs={"graphs": graphs, "classes": vocabulary.class_paths(graphs.values())},
    ))


def _sessions(vocabulary, codes: dict[str, str], panel=None):
    """Each graph of ``codes`` per stock and session."""
    return _factor(vocabulary, codes, panel).resample("1d", "last").compute(DAYS[0], DAYS[-1])


def _per_session(panel, reduce) -> np.ndarray:
    """``reduce(bars of one session and symbol, as a dict of columns)`` on ``[session, symbol]``."""
    out = np.full((len(DAYS), len(SYMBOLS)), np.nan)
    for i, day in enumerate(DAYS):
        rows = session_of(panel, day)
        for j in range(len(SYMBOLS)):
            out[i, j] = reduce({name: panel[name].values[rows, j] for name in panel.data_vars})
    return out


def _session_mean(values: np.ndarray) -> float:
    present = values[np.isfinite(values)]
    return present.mean() if len(present) else np.nan


def _of_present(reduce, empty=np.nan):
    """``reduce`` of the values present in a session, ``empty`` when there is none."""
    def reduced(values: np.ndarray) -> float:
        present = values[np.isfinite(values)]
        return reduce(present) if len(present) else empty
    return reduced


#: A numpy reference of each aggregation of one input, on one session's values.
REFERENCES = {
    "SessionMean": _of_present(np.mean),
    "SessionSum": _of_present(np.sum, 0.0),
    "SessionCount": _of_present(len, 0.0),
    "SessionMin": _of_present(np.min),
    "SessionMax": _of_present(np.max),
    "SessionFirst": _of_present(lambda present: present[0]),
    "SessionLast": _of_present(lambda present: present[-1]),
    # As pandas and KunQuant's windows: the unbiased standard deviation and skewness.
    "SessionStd": _of_present(lambda present: pd.Series(present).std()),
    "SessionSkew": _of_present(lambda present: pd.Series(present).skew()),
}


def _values(daily, name) -> np.ndarray:
    assert pd.DatetimeIndex(daily["timestamp"].values).strftime("%Y-%m-%d").tolist() == list(DAYS)
    return daily[name].sel(symbol=list(SYMBOLS)).transpose("timestamp", "symbol").values


def test_a_session_mean_is_one_node_of_the_recorded_graph(vocabulary):
    graph = _graph(vocabulary, "alpha = SessionMean(close * volume)")
    assert [node.op for node in graph.nodes] == ["Input", "Input", "Mul", "SessionMean"]


def _dimension(vocabulary, code):
    return _check(vocabulary, code).stats["dimension"]


def test_each_aggregation_measures_what_its_dimension_rule_says(vocabulary):
    for aggregation, dimension in AGGREGATIONS.items():
        assert _dimension(vocabulary, f"alpha = {aggregation}(close)") == dimension, aggregation
    assert _dimension(vocabulary, "alpha = SessionSum(volume)") == "volume"
    # A weighted mean measures what x does, whatever the weight; a correlation is a pure number.
    assert _dimension(vocabulary, "alpha = SessionWeightedMean(close, volume)") == "price"
    assert _dimension(vocabulary, "alpha = SessionCorr(close, volume)") == "pure number"


def test_a_correlation_with_its_inputs_swapped_is_a_rewritten_duplicate(vocabulary):
    gate = Gate(GateConfig(timeout_seconds=10.0), vocabulary)
    assert isinstance(_check(vocabulary, "alpha = SessionCorr(close, volume)", gate), Passed)
    swapped = _check(vocabulary, "alpha = SessionCorr(volume, close)", gate)
    assert isinstance(swapped, Rejected) and any("duplicate" in r for r in swapped.reasons), swapped


def test_the_session_mean_of_a_field_is_the_mean_of_its_values_present_in_the_session(vocabulary, panel):
    daily = _sessions(vocabulary, {"mean_close": "alpha = SessionMean(close)"})
    expected = _per_session(panel, lambda bars: _session_mean(bars["close"]))
    np.testing.assert_allclose(_values(daily, "mean_close"), expected, rtol=2e-6)
    assert np.isnan(expected[DAYS.index(EMPTY_DAY), SYMBOLS.index(EMPTY_SYMBOL)])


@pytest.mark.parametrize("aggregation", [*AGGREGATIONS, "SessionWeightedMean", "SessionCorr"])
def test_a_sessions_value_uses_that_sessions_bars_only(vocabulary, panel, aggregation):
    middle = DAYS.index(DAYS[1])
    changed = panel.copy(deep=True)
    others = ~session_of(panel, DAYS[middle])
    for name in ("close", "volume"):
        changed[name].values[others] *= 1.5 + 0.1 * np.arange(others.sum())[:, None]
    if aggregation in AGGREGATIONS:
        codes = {"dollars": f"alpha = {aggregation}(close * volume)", "price": f"alpha = {aggregation}(last_price)"}
    else:
        codes = {"dollars": f"alpha = {aggregation}(close, volume)",
                 "price": f"alpha = {aggregation}(last_price, volume)"}
    before, after = _sessions(vocabulary, codes), _sessions(vocabulary, codes, changed)
    for name in codes:
        np.testing.assert_array_equal(_values(after, name)[middle], _values(before, name)[middle])
        if aggregation != "SessionCount":  # which counts the bars with a value, unchanged
            assert not np.allclose(_values(after, name)[0], _values(before, name)[0], equal_nan=True)


def _last_price(close: np.ndarray) -> np.ndarray:
    """The last trade price at or before each bar of one session: missing before the session's first trade."""
    return pd.Series(close).ffill().to_numpy()


def test_the_session_inputs_are_fields_of_the_intraday_graph(vocabulary, panel):
    daily = _sessions(vocabulary, {
        "position": "alpha = SessionMean(session_position * close)",
        "left": "alpha = SessionMean(session_bars_left * close)",
        "mean_price": "alpha = SessionMean(last_price)",
    })

    def position(bars):
        return _session_mean(np.arange(len(bars["close"])) * bars["close"])

    def left(bars):
        return _session_mean(np.arange(len(bars["close"]))[::-1] * bars["close"])

    np.testing.assert_allclose(_values(daily, "position"), _per_session(panel, position), rtol=2e-6)
    np.testing.assert_allclose(_values(daily, "left"), _per_session(panel, left), rtol=2e-6)
    expected = _per_session(panel, lambda bars: _session_mean(_last_price(bars["close"])))
    np.testing.assert_allclose(_values(daily, "mean_price"), expected, rtol=2e-6)
    # The late symbol's first bars carry no price over from the session before.
    late, rows = SYMBOLS.index(LATE_SYMBOL), session_of(panel, DAYS[1])
    carried = _session_mean(pd.Series(panel["close"].values[:, late]).ffill().to_numpy()[rows])
    assert _values(daily, "mean_price")[1, late] != pytest.approx(carried, rel=1e-7)
    assert np.isnan(_values(daily, "mean_price")[DAYS.index(EMPTY_DAY), SYMBOLS.index(EMPTY_SYMBOL)])


def test_each_session_is_its_own_bars_whatever_their_number(vocabulary):
    daily = _sessions(vocabulary, {"length": "alpha = SessionMean(session_bars_left + session_position)"})
    for i, day in enumerate(DAYS):
        np.testing.assert_array_equal(_values(daily, "length")[i], session_bars(day) - 1)
    assert session_bars(HALF_DAY) < session_bars(DAYS[0])


def _accumulators(function: Function) -> int:
    """The accumulators KunQuant compiles ``function`` to."""
    source = "\n".join(compileit(
        function, "accumulators", input_layout="TS", output_layout="TS", options=dict(BATCH_OPTIONS),
    ))
    return len(set(re.findall(r"\baccu_\d+ \{", source)))


def _kunquant_session_sums(names: tuple[str, str]) -> Function:
    """Two running sums of ``2 x``, restarting where ``first`` is 1, as KunQuant accumulators named ``names``."""
    builder = Builder()
    with builder:
        x, first = Input("x"), Input("first")
        for i, name in enumerate(names):
            value = SetInfOrNanToValue(x * 2.0, 0.0)
            total = Accumulator(value, name, is_whole_time_required=True)
            restart = LessEqual(ConstantOp(1.0), first)
            Output(SetAccumulator(total, Equals(value, value), Select(restart, value, total + value)), f"sum_{i}")
    return Function(builder.ops)


def test_kunquant_merges_two_equal_accumulators_only_under_one_name():
    rng = np.random.default_rng(1)
    arrays = {
        "x": rng.normal(size=(12, 8)).astype(np.float32),
        "first": np.tile((np.arange(12) % 4 == 0)[:, None], (1, 8)).astype(np.float32),
    }
    values = {}
    for names in (("sum_a", "sum_b"), ("sum_a", "sum_a")):
        assert _accumulators(_kunquant_session_sums(names)) == len(set(names))
        out = run_batch_graph("SessionSums", _kunquant_session_sums(names), arrays, np.arange(12), np.arange(8), 1)
        values[names] = (out["sum_0"].values, out["sum_1"].values)
    for distinct, merged in zip(*values.values()):
        np.testing.assert_array_equal(distinct, merged)
    np.testing.assert_allclose(values["sum_a", "sum_a"][0][3], 2 * arrays["x"][:4].sum(axis=0), rtol=1e-6)


def test_two_session_means_of_one_subgraph_share_their_accumulators(vocabulary):
    one = _factor(vocabulary, {"a": "alpha = SessionMean(close * volume)"})
    two = _factor(vocabulary, {"a": "alpha = SessionMean(close * volume)", "b": "alpha = SessionMean(close * volume)"})
    other = _factor(vocabulary, {"a": "alpha = SessionMean(close * volume)", "b": "alpha = SessionMean(close)"})
    # A mean is its count, its first value and the sum of its values less the first.
    assert _accumulators(two.graph()) == _accumulators(one.graph()) == 3
    assert _accumulators(other.graph()) == 6
    daily = two.resample("1d", "last").compute(DAYS[0], DAYS[-1])
    np.testing.assert_array_equal(daily["a"].values, daily["b"].values)


def test_aggregations_of_one_subgraph_share_their_sums(vocabulary):
    std = _factor(vocabulary, {"std": "alpha = SessionStd(close * volume)"})
    both = _factor(vocabulary, {
        "mean": "alpha = SessionMean(close * volume)", "std": "alpha = SessionStd(volume * close)",
        "first": "alpha = SessionFirst(close * volume)", "count": "alpha = SessionCount(close * volume)",
    })
    # The standard deviation adds the sum of squares to the mean's count, first value and sum.
    assert _accumulators(both.graph()) == _accumulators(std.graph()) == 4


def test_a_subtree_two_graphs_share_is_computed_once(vocabulary):
    codes = {
        "mean": "alpha = SessionMean(Rank(WindowedAvg(close * volume, 5)))",
        "std": "alpha = SessionStd(Rank(WindowedAvg(volume * close, 5)))",
    }
    both = _factor(vocabulary, codes)
    # The two Rank subtrees are written apart but compute the same values (one canonical hash): one of each operator.
    ops = [type(op).__name__ for op in both.graph().ops]
    assert ops.count("Rank") == ops.count("WindowedAvg") == 1
    daily = both.resample("1d", "last").compute(DAYS[0], DAYS[-1])
    for name, code in codes.items():
        alone = _sessions(vocabulary, {name: code})
        np.testing.assert_allclose(daily[name].values, alone[name].values, rtol=1e-6)
        assert np.isfinite(alone[name].values).any()


def test_a_graph_that_does_not_end_in_an_aggregation_is_refused(vocabulary):
    # The Gate refuses such a graph (an intraday field reaching the daily stage), so it is recorded without it.
    graphs = {"a": Graph.record(vocabulary.execute("alpha = SessionMean(close) * volume")["alpha"], vocabulary).to_json()}
    factor = SessionFactor(FactorConfig(
        warmup_bars=0, dataset=intraday_dataset(), mode="batch", njobs=1,
        data_columns=SessionFactor.data_columns(graphs),
        kwargs={"graphs": graphs, "classes": vocabulary.class_paths(graphs.values())},
    ))
    with pytest.raises(ValueError, match="graph a does not end in a session aggregation"):
        factor.graph()


def test_the_session_mean_is_expanded_only_when_the_function_is_built(vocabulary):
    factor = _factor(vocabulary, {"a": "alpha = SessionMean(close)"})
    ops = {type(op).__name__ for op in factor.graph().ops}
    assert "Accumulator" in ops and "SessionMean" not in ops


def _of_pairs(reduce):
    """``reduce(x, y)`` of the bars of a session where both have a value, NaN when there is none."""
    def reduced(x: np.ndarray, y: np.ndarray) -> float:
        both = np.isfinite(x) & np.isfinite(y)
        return reduce(x[both], y[both]) if both.any() else np.nan
    return reduced


#: A numpy reference of each aggregation of two inputs.
PAIR_REFERENCES = {
    "SessionWeightedMean": _of_pairs(lambda x, w: np.sum(x * w) / np.sum(w) if np.sum(w) != 0 else np.nan),
    "SessionCorr": _of_pairs(lambda x, y: np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan),
}

#: The absolute error a pure number of float32 sums keeps, near 0.
ABSOLUTE = {"SessionSkew": 5e-5}


@pytest.mark.parametrize("aggregation", REFERENCES)
def test_each_aggregation_of_one_input_is_its_numpy_reference_on_the_values_present(vocabulary, panel, aggregation):
    daily = _sessions(vocabulary, {
        "of_close": f"alpha = {aggregation}(close)",
        "of_dollars": f"alpha = {aggregation}(close * volume)",
        "of_price": f"alpha = {aggregation}(last_price)",
        "of_rank": f"alpha = {aggregation}(Rank(close * volume))",
    })
    reference = REFERENCES[aggregation]
    # A cross-sectional operator ranks across the symbols at each bar.
    ranked = panel.assign(rank=(panel["close"] * panel["volume"]).to_pandas().rank(axis=1, pct=True))
    for name, values in {
        "of_close": lambda bars: bars["close"],
        "of_dollars": lambda bars: bars["close"] * bars["volume"],
        "of_price": lambda bars: _last_price(bars["close"]),
        "of_rank": lambda bars: bars["rank"],
    }.items():
        expected = _per_session(ranked, lambda bars: reference(values(bars)))
        np.testing.assert_allclose(_values(daily, name), expected, rtol=2e-5, atol=ABSOLUTE.get(aggregation, 1e-6),
                                   err_msg=name)
    empty = _values(daily, "of_close")[DAYS.index(EMPTY_DAY), SYMBOLS.index(EMPTY_SYMBOL)]
    if aggregation in ("SessionCount", "SessionSum"):
        assert empty == 0
    else:
        assert np.isnan(empty)


def _price_levels(panel):
    """``panel`` with closes around 125 to 1000 dollars that move a tenth as much: deviations far below the level."""
    levels = panel.copy(deep=True)
    close = levels["close"].values
    level = 125.0 * np.arange(1, len(SYMBOLS) + 1)
    close[:] = level * (1 + 0.1 * np.log(close / np.nanmean(close, axis=0)))
    return levels


def test_moments_of_price_levels_keep_their_relative_accuracy_in_float32(vocabulary, panel):
    levels = _price_levels(panel)
    daily = _sessions(vocabulary, {"std": "alpha = SessionStd(close)", "skew": "alpha = SessionSkew(close)"}, levels)
    std = _per_session(levels, lambda bars: REFERENCES["SessionStd"](bars["close"]))
    # Sums of the prices themselves in float32 would lose the deviations; numpy's in float64 keep them.
    assert np.nanmax(std / _per_session(levels, lambda bars: _session_mean(bars["close"]))) < 2e-3
    np.testing.assert_allclose(_values(daily, "std"), std, rtol=1e-4)
    skew = _per_session(levels, lambda bars: REFERENCES["SessionSkew"](bars["close"]))
    np.testing.assert_allclose(_values(daily, "skew"), skew, rtol=1e-3, atol=1e-3)


def _position(bars):
    return np.arange(len(bars["close"]), dtype=float)


@pytest.mark.parametrize("aggregation", PAIR_REFERENCES)
def test_each_aggregation_of_two_inputs_is_its_numpy_reference_on_the_bars_where_both_have_a_value(
    vocabulary, panel, aggregation,
):
    daily = _sessions(vocabulary, {
        "by_volume": f"alpha = {aggregation}(close, volume)",
        "by_position": f"alpha = {aggregation}(last_price, session_position)",
        "by_dollars": f"alpha = {aggregation}(volume, close * volume)",
    })
    reference = PAIR_REFERENCES[aggregation]
    for name, (x, y) in {
        "by_volume": (lambda bars: bars["close"], lambda bars: bars["volume"]),
        "by_position": (lambda bars: _last_price(bars["close"]), _position),
        "by_dollars": (lambda bars: bars["volume"], lambda bars: bars["close"] * bars["volume"]),
    }.items():
        expected = _per_session(panel, lambda bars: reference(x(bars), y(bars)))
        np.testing.assert_allclose(_values(daily, name), expected, rtol=2e-5, atol=1e-5, err_msg=name)
    assert np.isnan(_values(daily, "by_volume")[DAYS.index(EMPTY_DAY), SYMBOLS.index(EMPTY_SYMBOL)])


def test_a_correlation_of_price_levels_keeps_its_relative_accuracy_in_float32(vocabulary, panel):
    levels = _price_levels(panel)
    daily = _sessions(vocabulary, {"trend": "alpha = SessionCorr(close, session_position)"}, levels)
    expected = _per_session(levels, lambda bars: PAIR_REFERENCES["SessionCorr"](bars["close"], _position(bars)))
    np.testing.assert_allclose(_values(daily, "trend"), expected, rtol=1e-4, atol=1e-4)


def test_the_memory_a_bar_takes_counts_the_session_inputs_and_grows_with_what_is_computed(vocabulary):
    mean = _factor(vocabulary, {"a": "alpha = SessionMean(volume)"})
    peak, output = mean.cell_bytes()
    assert output == 8
    # The columns read, the three session inputs, the float64 copy of the price they come from and the output.
    assert peak >= 8 * (len(mean.config.data_columns) + len(SESSION_INPUTS) + 1 + 1)
    # A cross-sectional rank of each bar ends a KunQuant stage: its values are held for every bar.
    ranked = _factor(vocabulary, {"a": "alpha = SessionMean(Rank(volume))"})
    assert ranked.config.data_columns == mean.config.data_columns
    assert ranked.cell_bytes()[0] > peak
    both = _factor(vocabulary, {"a": "alpha = SessionMean(volume)", "b": "alpha = SessionMax(volume)"})
    assert both.cell_bytes()[0] > peak
