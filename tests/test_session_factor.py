"""Session columns from intraday bars: ``SessionFactor`` on the synthetic minute dataset of ``synthetic_intraday``.

Candidate code is recorded through a real Gate, under a vocabulary of the
intraday bars' columns and session inputs and a few operators, so a
``SessionMean`` is the one node the Gate records; the factor computes each
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
from factor_agent.gate import Gate, Passed
from factor_agent.gate.vocabulary import Vocabulary
from factor_agent.session import SessionFactor
from factor_agent.zoo import Zoo
from tests.synthetic_intraday import (
    DAYS, EMPTY_DAY, EMPTY_SYMBOL, HALF_DAY, LATE_SYMBOL, SYMBOLS, intraday_dataset, minute_bars,
    session_bars, session_of,
)

FIELDS = {
    "family": "minute_bars",
    "description": "one-minute trade bars",
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

OPERATORS = {
    "operators": [
        {"name": "SessionMean", "class": "factor_agent.aggregations.SessionMean", "signature": "SessionMean(x)",
         "dimension": "same", "description": "mean of x over the session's bars with a value"},
        {"name": "Mul", "class": "KunQuant.ops.Mul", "signature": "Mul(x, y)", "inputs": ["series", "series"],
         "dimension": "product"},
        {"name": "Add", "class": "KunQuant.ops.Add", "signature": "Add(x, y)", "inputs": ["series", "series"],
         "dimension": "match"},
    ],
}


@pytest.fixture(scope="module")
def vocabulary(tmp_path_factory):
    root = tmp_path_factory.mktemp("session_factor")
    (root / "fields.yaml").write_text(yaml.safe_dump(FIELDS))
    (root / "operators.yaml").write_text(yaml.safe_dump(OPERATORS))
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(CACHE_ENV, str(root / "kunquant_cache"))
        yield Vocabulary.load([root / "fields.yaml"], root / "operators.yaml")


@pytest.fixture(scope="module")
def panel():
    return minute_bars()


def _graph(vocabulary, code):
    result = Gate(GateConfig(timeout_seconds=10.0), vocabulary).check(
        Candidate(hypothesis="Intraday prices.", mechanism_type="behavioural_bias",
                  horizon_rationale="A week.", description="Test.", code=code),
        Zoo(),
    )
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


def _values(daily, name) -> np.ndarray:
    assert pd.DatetimeIndex(daily["timestamp"].values).strftime("%Y-%m-%d").tolist() == list(DAYS)
    return daily[name].sel(symbol=list(SYMBOLS)).transpose("timestamp", "symbol").values


def test_a_session_mean_is_one_node_of_the_recorded_graph(vocabulary):
    graph = _graph(vocabulary, "alpha = SessionMean(close * volume)")
    assert [node.op for node in graph.nodes] == ["Input", "Input", "Mul", "SessionMean"]


def test_the_session_mean_of_a_field_is_the_mean_of_its_values_present_in_the_session(vocabulary, panel):
    daily = _sessions(vocabulary, {"mean_close": "alpha = SessionMean(close)"})
    expected = _per_session(panel, lambda bars: _session_mean(bars["close"]))
    np.testing.assert_allclose(_values(daily, "mean_close"), expected, rtol=2e-6)
    assert np.isnan(expected[DAYS.index(EMPTY_DAY), SYMBOLS.index(EMPTY_SYMBOL)])


def test_a_sessions_value_uses_that_sessions_bars_only(vocabulary, panel):
    middle = DAYS.index(DAYS[1])
    changed = panel.copy(deep=True)
    others = ~session_of(panel, DAYS[middle])
    for name in ("close", "volume"):
        changed[name].values[others] *= 1.5
    codes = {"mean_dollars": "alpha = SessionMean(close * volume)", "mean_price": "alpha = SessionMean(last_price)"}
    before, after = _sessions(vocabulary, codes), _sessions(vocabulary, codes, changed)
    for name in codes:
        np.testing.assert_array_equal(_values(after, name)[middle], _values(before, name)[middle])
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
    assert _accumulators(two.graph()) == _accumulators(one.graph()) == 2
    assert _accumulators(other.graph()) == 4
    daily = two.resample("1d", "last").compute(DAYS[0], DAYS[-1])
    np.testing.assert_array_equal(daily["a"].values, daily["b"].values)


def test_a_graph_that_does_not_end_in_an_aggregation_is_refused(vocabulary):
    factor = _factor(vocabulary, {"a": "alpha = SessionMean(close) * volume"})
    with pytest.raises(ValueError, match="graph a does not end in a session aggregation"):
        factor.graph()


def test_the_session_mean_is_expanded_only_when_the_function_is_built(vocabulary):
    factor = _factor(vocabulary, {"a": "alpha = SessionMean(close)"})
    ops = {type(op).__name__ for op in factor.graph().ops}
    assert "Accumulator" in ops and "SessionMean" not in ops
