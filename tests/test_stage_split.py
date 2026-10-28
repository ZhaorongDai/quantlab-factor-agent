"""The stage split: a recorded graph as its daily graph and its intraday graphs, and the stage rules it breaks.

Graphs are recorded from candidate code under the vocabulary of the
price-volume and minute-bar field files and the example operator file. The
expected daily and intraday graphs are written with KunQuant's operators
over the columns themselves, a session column as the ``Input`` it is, and
compared by fingerprint.
"""

import pytest
from KunQuant.Op import Input, Rank
from KunQuant.ops import Div, Mul, WindowedAvg

from factor_agent.aggregations import SessionMean
from factor_agent.gate._canonical import computation_hashes
from factor_agent.gate.graph import Graph
from factor_agent.gate.stages import split
from tests.vocabulary_files import intraday, price_volume


def _recorded(code: str, vocabulary=None) -> Graph:
    vocabulary = vocabulary or intraday()
    return Graph.record(vocabulary.execute(code)["alpha"], vocabulary)


def _graph(op) -> Graph:
    return Graph.record(op, intraday())


def _session_column(op) -> str:
    """The name of the session column of the intraday graph ``op``: the hash of what it computes."""
    return f"session_{computation_hashes(_graph(op))[-1]}"


def test_a_daily_graph_is_its_own_daily_graph_with_no_intraday_graph():
    graph = _recorded("alpha = WindowedAvg(close / open, 5)", price_volume())
    staged = split(graph, price_volume())
    assert staged.daily is graph
    assert staged.intraday == {}
    assert staged.reasons == ()


def test_an_aggregation_subtree_becomes_a_session_column_the_daily_graph_reads():
    staged = split(_recorded("alpha = WindowedAvg(SessionMean(bar_close * bar_volume), 5) / close"), intraday())
    session = SessionMean(Mul(Input("close"), Input("volume")))
    name = _session_column(session)
    assert staged.reasons == ()
    assert list(staged.intraday) == [name]
    assert staged.intraday[name].fingerprint() == _graph(session).fingerprint()
    assert staged.daily.fingerprint() == _graph(Div(WindowedAvg(Input(name), 5), Input("adjClose"))).fingerprint()
    assert staged.daily.input_names == tuple(sorted([name, "adjClose"]))


def test_each_intraday_branch_is_an_intraday_graph_of_its_own():
    staged = split(_recorded("alpha = SessionMean(bar_buy_volume) / SessionMean(bar_volume)"), intraday())
    buy, volume = SessionMean(Input("buy_volume")), SessionMean(Input("volume"))
    assert staged.reasons == ()
    assert set(staged.intraday) == {_session_column(buy), _session_column(volume)}
    expected = Div(Input(_session_column(buy)), Input(_session_column(volume)))
    assert staged.daily.fingerprint() == _graph(expected).fingerprint()


def test_a_subtree_two_graphs_share_is_one_session_column_whatever_the_operand_order():
    first = split(_recorded("alpha = Rank(SessionMean(bar_close * bar_volume))"), intraday())
    second = split(_recorded("alpha = WindowedAvg(SessionMean(bar_volume * bar_close), 20)"), intraday())
    assert list(first.intraday) == list(second.intraday)
    [name] = first.intraday
    assert first.daily.fingerprint() == _graph(Rank(Input(name))).fingerprint()


def test_a_cross_sectional_operator_may_rank_each_bar_in_the_intraday_stage():
    staged = split(_recorded("alpha = SessionMean(Rank(bar_volume))"), intraday())
    assert staged.reasons == ()
    assert len(staged.intraday) == 1


@pytest.mark.parametrize("code, reason", [
    (
        "alpha = bar_close / close",
        "the intraday field bar_close reaches the daily stage without a session aggregation",
    ),
    (
        "alpha = WindowedAvg(bar_volume, 5)",
        "the intraday field bar_volume reaches the daily stage without a session aggregation",
    ),
    (
        "alpha = SessionMean(bar_close / close)",
        "the daily field close is read inside the intraday stage, under SessionMean",
    ),
    (
        "alpha = SessionMean(ExpMovingAvg(bar_close, 5))",
        "ExpMovingAvg is used in the intraday stage, but may be used in the daily stage only",
    ),
])
def test_a_graph_breaking_a_stage_rule_is_given_the_reason(code, reason):
    staged = split(_recorded(code), intraday())
    assert reason in staged.reasons, staged.reasons


def test_an_operator_of_the_intraday_stage_only_is_refused_in_the_daily_stage(tmp_path):
    from factor_agent.gate.vocabulary import Vocabulary
    from tests.vocabulary_files import EXAMPLE_OPERATORS, MINUTE_BARS, PRICE_VOLUME

    text = EXAMPLE_OPERATORS.read_text().replace(
        "    dimension: transcendental\n    stages: [intraday, daily]\n    description: natural logarithm",
        "    dimension: transcendental\n    stages: [intraday]\n    description: natural logarithm",
    )
    assert text != EXAMPLE_OPERATORS.read_text()
    (tmp_path / "operators.yaml").write_text(text)
    vocabulary = Vocabulary.load([PRICE_VOLUME, MINUTE_BARS], tmp_path / "operators.yaml")
    assert split(_recorded("alpha = SessionMean(Log(bar_close))", vocabulary), vocabulary).reasons == ()
    assert split(_recorded("alpha = Log(SessionMean(bar_close))", vocabulary), vocabulary).reasons == (
        "Log is used in the daily stage, but may be used in the intraday stage only",
    )


def test_the_daily_bar_mask_of_a_nan_preserving_operator_stays_out_of_the_intraday_stage():
    from KunQuant.Op import ConstantOp
    from KunQuant.ops import LessThanConst, Select

    # Select is quantlab's NaN-preserving stand-in, which adds the daily family's bar mask.
    staged = split(_recorded("alpha = SessionMean(Select(session_bars_left < 20, bar_volume, ConstantOp(0)))"), intraday())
    assert staged.reasons == ()
    [graph] = staged.intraday.values()
    session = SessionMean(Select(LessThanConst(Input("session_bars_left"), 20), Input("volume"), ConstantOp(0)))
    assert graph.fingerprint() == _graph(session).fingerprint()
    assert list(staged.intraday) == [_session_column(session)]


@pytest.mark.parametrize("code, reason", [
    (
        "alpha = SessionMean(close)",
        "SessionMean aggregates a daily value: an aggregation reduces the intraday bars of each session",
    ),
    (
        "alpha = SessionStd(SessionMean(bar_close))",
        "SessionMean is nested inside SessionStd: an aggregation's input is a value of the intraday bars",
    ),
])
def test_an_aggregation_of_a_daily_value_or_of_an_aggregation_is_refused(code, reason):
    staged = split(_recorded(code), intraday())
    assert staged.reasons == (reason,)
    assert staged.intraday == {}
