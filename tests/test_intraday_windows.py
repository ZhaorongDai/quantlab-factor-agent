"""Windows of the intraday stage stay within the session: missing on its first bars until they fit, blind to the session before.

Candidate code is split into its stages (``stages.split``) under the
price-volume and minute-bar field files and the example operator file, and
its intraday graphs are computed by ``SessionColumns`` on the synthetic
minute dataset of ``synthetic_intraday``: three sessions of eight symbols,
one a half day.
"""

import itertools

import numpy as np
import pandas as pd
import pytest
from quantlab.factor.kunquant_compile import CACHE_ENV

from factor_agent.gate.graph import Graph
from factor_agent.gate.stages import session_column, split
from factor_agent.session import SessionColumns
from tests.synthetic_intraday import DAYS, SYMBOLS, intraday_dataset, minute_bars, session_bars, session_of
from tests.vocabulary_files import intraday


#: Where each dataset the tests compute on is stored: the session columns read a store.
_STORES = {}


@pytest.fixture(scope="module", autouse=True)
def _cache(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(CACHE_ENV, str(tmp_path_factory.mktemp("kunquant_cache")))
        _STORES["root"], _STORES["count"] = tmp_path_factory.mktemp("minutes"), itertools.count()
        yield


def _staged(code: str):
    vocabulary = intraday()
    staged = split(Graph.record(vocabulary.execute(code)["alpha"], vocabulary), vocabulary)
    assert staged.reasons == ()
    return staged


def _sessions(codes: dict[str, str], panel=None, granularity=None) -> dict[str, np.ndarray]:
    """The session column of each candidate of ``codes``, one aggregation each, on ``[day, symbol]``.

    ``granularity`` is how the session columns are computed in chunks (``SessionColumns``); None for at once.
    """
    graphs = {}
    for key, code in codes.items():
        [(name, graph)] = _staged(code).intraday.items()
        graphs[key] = (name, graph.to_json())
    recorded = {name: graph for name, graph in graphs.values()}
    store = intraday_dataset(panel).to_zarr(_STORES["root"] / f"minutes_{next(_STORES['count'])}.zarr")
    provider = SessionColumns(
        store, recorded, intraday().class_paths(recorded.values()), njobs=1, granularity=granularity,
    )
    filled = provider.fill(None, None, {}, np.array(SYMBOLS), pd.DatetimeIndex(DAYS).values)
    return {key: filled[name] for key, (name, _) in graphs.items()}


def test_an_intraday_graph_is_named_after_what_it_computes_once_its_windows_are_rewritten():
    staged = _staged("alpha = SessionMean(WindowedAvg(bar_close, 5))")
    [(name, graph)] = staged.intraday.items()
    assert name == session_column(graph.fingerprint())
    assert "session_position" in graph.input_names
    unwindowed = _staged("alpha = SessionMean(bar_close)")
    assert "session_position" not in next(iter(unwindowed.intraday.values())).input_names


#: Candidates counting the bars where a window of the session position has a value, and the bars it reaches back.
SPANS = {
    "window": ("alpha = SessionCount(WindowedAvg(session_position, 5))", 4),
    "lag": ("alpha = SessionCount(BackRef(session_position, 1))", 1),
    "nested": ("alpha = SessionCount(BackRef(WindowedAvg(session_position, 5), 10))", 14),
    "exponential window": ("alpha = SessionCount(EWMean(session_position, 10, 5))", 9),
    "loop": (
        "loop = ForeachBackWindow(session_position, 5)\n"
        "with Builder(loop):\n"
        "    item = IterValue(loop, session_position)\n"
        "alpha = SessionCount(ReduceAdd(item) + session_position)",
        4,
    ),
}


def test_an_intraday_window_is_missing_on_a_sessions_first_bars_until_it_fits():
    counts = _sessions({key: code for key, (code, _) in SPANS.items()})
    for key, (_, span) in SPANS.items():
        for i, day in enumerate(DAYS):
            np.testing.assert_array_equal(counts[key][i], session_bars(day) - span, err_msg=f"{key} on {day}")


#: Intraday windows of a price, missing on a bar without a trade.
WINDOWED = {
    "mean": "alpha = SessionMean(WindowedAvg(bar_close, 5))",
    "std": "alpha = SessionMean(WindowedStddev(bar_close, 10))",
    "lag": "alpha = SessionMean(bar_close / BackRef(bar_close, 1))",
    "max": "alpha = SessionMean(WindowedMax(bar_close, 15))",
    "rank": "alpha = SessionMean(TsRank(bar_close, 10))",
    "quantile": "alpha = SessionMean(WindowedQuantile(bar_close, 10, 0.5))",
    "decay": "alpha = SessionMean(DecayLinear(bar_close, 5))",
    "exponential": "alpha = SessionMean(EWMean(bar_close, 10, 5))",
    "nested": "alpha = SessionMean(WindowedAvg(bar_close / BackRef(bar_close, 5), 10))",
    "volume": "alpha = SessionSum(WindowedSum(bar_volume, 30))",
}


def test_an_intraday_windows_values_do_not_depend_on_the_session_before():
    panel = minute_bars()
    changed = panel.copy(deep=True)
    before = session_of(panel, DAYS[0])
    for name in ("close", "volume"):
        changed[name].values[before] *= 1.5 + 0.1 * np.arange(before.sum())[:, None]
    first, second = _sessions(WINDOWED), _sessions(WINDOWED, changed)
    for key in WINDOWED:
        np.testing.assert_array_equal(second[key][1], first[key][1], err_msg=key)
        assert not np.allclose(second[key][0], first[key][0], equal_nan=True), key
        assert np.isfinite(first[key][1]).sum() >= len(SYMBOLS) - 1, key


def test_intraday_windows_computed_a_session_at_a_time_equal_those_computed_at_once():
    whole, daily = _sessions(WINDOWED), _sessions(WINDOWED, granularity="day")
    for key in WINDOWED:
        np.testing.assert_array_equal(daily[key], whole[key], err_msg=key)


def test_a_cross_sectional_rank_in_the_intraday_stage_ranks_across_all_symbols_of_the_bar():
    panel = minute_bars()
    [ranks] = _sessions({"rank": "alpha = SessionMean(Rank(bar_close))"}).values()
    # KunQuant's Rank: the share of the bar's symbols with a value at most this one's, NaN where it has none.
    close = panel["close"].transpose("timestamp", "symbol").values
    per_bar = pd.DataFrame(close).rank(axis=1, pct=True, method="max").to_numpy()
    for i, day in enumerate(DAYS):
        bars = per_bar[session_of(panel, day)]
        expected = np.array([np.nanmean(bars[:, j]) if np.isfinite(bars[:, j]).any() else np.nan for j in range(len(SYMBOLS))])
        np.testing.assert_allclose(ranks[i], expected, rtol=1e-5)


def test_the_rewrite_reads_the_session_position_the_session_factor_adds():
    from quantlab.factor.predefined import intraday as quantlab_intraday

    from factor_agent.aggregations import SESSION_POSITION

    assert SESSION_POSITION == quantlab_intraday.SESSION_POSITION
