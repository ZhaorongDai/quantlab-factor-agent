"""Session columns as a derived-columns provider: an intraday graph's daily values, on the daily panel's axes.

``SessionColumns`` computes the intraday graphs of a stage split on the
synthetic minute dataset of ``synthetic_intraday`` and fills them on the
symbols and dates a daily factor runs on: in their order, NaN for a symbol
the minute bars do not hold and for a day without a session. Expected values
are a few lines of numpy per session.
"""

import numpy as np
import pandas as pd
import pytest
from quantlab.factor.kunquant_compile import CACHE_ENV

from factor_agent.gate.graph import Graph
from factor_agent.gate.stages import split
from factor_agent.session import SessionColumns
from tests.synthetic_intraday import DAYS, SYMBOLS, intraday_dataset, minute_bars, session_of
from tests.vocabulary_files import intraday


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "kunquant_cache"))


def _intraday_graphs(code: str) -> dict[str, list]:
    """The intraday graphs of ``code``'s stage split, as ``Graph.to_json`` records them."""
    vocabulary = intraday()
    staged = split(Graph.record(vocabulary.execute(code)["alpha"], vocabulary), vocabulary)
    assert staged.reasons == ()
    return {name: graph.to_json() for name, graph in staged.intraday.items()}


def test_session_columns_are_filled_on_the_daily_symbols_and_dates():
    graphs = _intraday_graphs("alpha = SessionMean(bar_close * bar_volume)")
    [name] = graphs
    provider = SessionColumns(intraday_dataset(), graphs, intraday().class_paths(graphs.values()), njobs=1)
    assert provider.columns == (name,)
    # A day before the minute bars begin, Thanksgiving without a session, and a symbol they do not hold.
    timestamps = pd.DatetimeIndex(["2024-11-25", *DAYS[:2], "2024-11-28", DAYS[2]]).values
    symbols = np.array(["S5", "S0", "S9", "S2"])
    filled = provider.fill(None, None, {}, symbols, timestamps)

    panel = minute_bars()
    expected = np.full((len(timestamps), len(symbols)), np.nan)
    for row, day in [(1, DAYS[0]), (2, DAYS[1]), (4, DAYS[2])]:
        bars = session_of(panel, day)
        for col, symbol in enumerate(symbols):
            if symbol in SYMBOLS:
                values = (panel["close"] * panel["volume"]).sel(symbol=symbol).values[bars]
                present = values[np.isfinite(values)]
                expected[row, col] = present.mean() if len(present) else np.nan
    assert filled[name].shape == expected.shape
    np.testing.assert_allclose(filled[name], expected, rtol=1e-5)
