"""Evaluator.evaluate() and join(): neutralisation, compile-failure bisection, the pool, admission order, the store.

Graphs come from the real Gate and are evaluated on the synthetic store of
``tests/synthetic_market.py``; the reference set is Alpha158's KMID, which
is the planted intraday signal ``close / open`` up to a constant.
"""

import json

import numpy as np
import pandas as pd
import polars as pl
import pytest
from KunQuant.Op import Builder, Input, Output, Rank
from KunQuant.Stage import Function
from KunQuant.ops import Log, WindowedAvg
from quantlab.analysis.factor_report import newey_west_lags, newey_west_t_stat
from quantlab.core.component import rebuild
from quantlab.factor.config import FactorConfig, NeutralizedConfig
from quantlab.factor.kunquant import FactorKunQuant
from quantlab.factor.polars import FactorPolars
from quantlab.factor.kunquant_compile import CACHE_ENV
from quantlab.factor.predefined.alpha158 import Alpha158Stock
from quantlab.factor.predefined.neutralized import NeutralizedFactor
from quantlab.model.evaluation import cross_sectional_rank_ic_series

from factor_agent.candidate import Candidate
from factor_agent.config import GateConfig
from factor_agent.evaluator import CompileFailure, Evaluator, Metrics
from factor_agent.exported import GraphFactor
from factor_agent.gate import Gate
from factor_agent.gate.graph import Graph
from factor_agent.scoring import Neutralization, Scoring
from factor_agent.segments import Segment
from factor_agent.zoo import Zoo
from tests.vocabulary_files import price_volume
from tests import synthetic_market

WARMUP = 60
INPUTS = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")


@pytest.fixture(scope="module")
def market(tmp_path_factory):
    root = tmp_path_factory.mktemp("evaluator")
    components = synthetic_market.build(root / "data")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(CACHE_ENV, str(root / "kunquant_cache"))
        yield {name: rebuild(component) for name, component in components.items()}


def _scoring(market, neutralised=True, membership=True) -> Scoring:
    """The campaign's scoring, or one without neutralisation or without membership."""
    neutralization = Neutralization(market["exposures"], "marketcap", "industry") if neutralised else None
    return Scoring(market["membership"] if membership else None, neutralization, njobs=4)


def _evaluator(market, reference=(), scoring=None, label=None) -> Evaluator:
    return Evaluator.for_segment(
        Segment("training", *synthetic_market.TRAINING),
        prices=market["prices"], label=label or market["label"], scoring=scoring or _scoring(market),
        reference=reference, warmup_bars=WARMUP, njobs=4,
    )


def _kmid(market) -> Alpha158Stock:
    return Alpha158Stock(FactorConfig(
        warmup_bars=WARMUP, dataset=market["prices"], mode="batch", data_columns=INPUTS,
        factor_names=("KMID",), njobs=4,
    ))


def _graph(code: str) -> Graph:
    candidate = Candidate.from_dict({
        "hypothesis": "A test hypothesis.", "mechanism_type": "behavioural_bias",
        "horizon_rationale": "A week.", "description": "test", "code": code,
    }, ["behavioural_bias"])
    checked = Gate(GateConfig(), price_volume()).check(candidate, Zoo())
    assert not hasattr(checked, "reasons"), checked
    return checked.graph


#: A graph that passes from_json but that KunQuant cannot compile: its window is not a number.
UNCOMPILABLE = Graph.from_json(
    [["Input", [["name", "adjClose"]], [], None], ["WindowedAvg", [["window", "abc"]], [0], None]], price_volume().classes,
)


class _RankedIntraday(FactorKunQuant):
    """``(close - open) / open`` ranked across the market, as a ``ranked=True`` quantlab alpha outputs it."""

    def _get_factor_names(self) -> tuple[str, ...]:
        return ("intraday",)

    def _get_factor_func(self) -> Function:
        builder = Builder()
        with builder:
            close, open_ = Input("adjClose"), Input("adjOpen")
            Output(Rank((close - open_) / open_), "intraday")
        return Function(builder.ops)


def _same(a: Metrics, b: Metrics) -> bool:
    """Equal metrics, NaN equal to NaN."""
    return json.dumps(a.to_dict()) == json.dumps(b.to_dict())


def _ic_of(market, panel, label_factor=None, masked=True) -> tuple[float, float]:
    """Mean daily rank IC and its NW t of ``panel``, masked to members unless not ``masked``, on the scored dates."""
    label_factor = label_factor or market["label"]
    label = label_factor.read(*synthetic_market.TRAINING)[label_factor.get_factor_names()[0]]
    label = label.isel(timestamp=slice(0, label.sizes["timestamp"] - label_factor.lookahead_bars()))
    panel = market["membership"].mask(panel) if masked else panel
    values = panel["intraday"].reindex(timestamp=label["timestamp"], symbol=label["symbol"])
    ic = cross_sectional_rank_ic_series(values.values, label.values)
    series = pd.Series(ic[np.isfinite(ic)])
    t_stat, _ = newey_west_t_stat(series, newey_west_lags(len(series), label_factor.span_bars()))
    return float(series.mean()), t_stat


def test_neutralisation_matches_quantlabs_neutralised_factor(market):
    ranked = _RankedIntraday(FactorConfig(
        warmup_bars=WARMUP, dataset=market["prices"], mode="batch", data_columns=("adjOpen", "adjClose"), njobs=4,
    ))
    neutral = NeutralizedFactor(NeutralizedConfig(factor=ranked, dataset=market["exposures"], njobs=4))
    expected = _ic_of(market, neutral.compute(*synthetic_market.TRAINING))
    unneutralised = _ic_of(market, ranked.compute(*synthetic_market.TRAINING))

    [metrics] = _evaluator(market).evaluate([_graph("alpha = (close - open) / open")])

    assert (metrics.ic_mean, metrics.ic_t) == pytest.approx(expected, rel=1e-9)
    assert abs(metrics.ic_mean - unneutralised[0]) > 1e-4


def _ranked_intraday(market) -> "_RankedIntraday":
    return _RankedIntraday(FactorConfig(
        warmup_bars=WARMUP, dataset=market["prices"], mode="batch", data_columns=("adjOpen", "adjClose"), njobs=4,
    ))


def test_without_neutralisation_the_ranked_values_are_scored(market):
    expected = _ic_of(market, _ranked_intraday(market).compute(*synthetic_market.TRAINING))

    [metrics] = _evaluator(market, scoring=_scoring(market, neutralised=False)).evaluate(
        [_graph("alpha = (close - open) / open")]
    )

    assert (metrics.ic_mean, metrics.ic_t) == pytest.approx(expected, rel=1e-9)


def test_without_membership_every_symbol_is_scored(market):
    # The planted intraday signal has the stated sign on the members and the reverse on the rest.
    whole_market = market["market_label"]
    expected = _ic_of(market, _ranked_intraday(market).compute(*synthetic_market.TRAINING), whole_market, masked=False)
    on_members = _ic_of(market, _ranked_intraday(market).compute(*synthetic_market.TRAINING), whole_market)

    [metrics] = _evaluator(
        market, scoring=_scoring(market, neutralised=False, membership=False), label=whole_market,
    ).evaluate([_graph("alpha = (close - open) / open")])

    assert (metrics.ic_mean, metrics.ic_t) == pytest.approx(expected, rel=1e-9)
    assert on_members[0] > 0.1 > abs(expected[0])


def test_a_batch_with_a_graph_that_fails_to_compile_returns_metrics_for_every_other_graph(market):
    volume, intraday = _graph("alpha = volume / WindowedAvg(volume, 20)"), _graph("alpha = close / open")
    evaluator = _evaluator(market)

    results = evaluator.evaluate([volume, UNCOMPILABLE, intraday])

    assert isinstance(results[1], CompileFailure)
    assert "TypeError" in results[1].error
    assert _same(results[0], evaluator.evaluate([volume])[0])
    assert _same(results[2], evaluator.evaluate([intraday])[0])
    assert all(isinstance(r, Metrics) and r.ic_t > 3 for r in (results[0], results[2]))


def test_a_lone_graph_that_fails_to_compile_is_a_compile_failure(market):
    assert isinstance(_evaluator(market).evaluate([UNCOMPILABLE])[0], CompileFailure)


@pytest.fixture(scope="module")
def against_kmid(market):
    codes = {
        "noisy copy": "alpha = (close - open) / (high - low)",
        "averaged copy": "alpha = WindowedAvg(close / open, 5)",
        "volume": "alpha = volume / WindowedAvg(volume, 20)",
    }
    results = _evaluator(market, [_kmid(market)]).evaluate([_graph(code) for code in codes.values()])
    return dict(zip(codes, results))


def test_a_noisy_copy_of_a_reference_factor_is_nearest_to_it(against_kmid):
    noisy = against_kmid["noisy copy"]
    assert noisy.nearest == "KMID (Alpha158Stock)"
    assert noisy.nearest_correlation > 0.9


def test_the_partial_ic_removes_what_the_reference_factor_explains(against_kmid):
    averaged = against_kmid["averaged copy"]
    assert abs(averaged.nearest_correlation) < 0.7
    assert averaged.ic_t > 10
    assert averaged.partial_ic_t < averaged.ic_t / 3


def test_information_the_reference_lacks_keeps_its_partial_ic(against_kmid):
    volume = against_kmid["volume"]
    assert abs(volume.nearest_correlation) < 0.1
    assert volume.partial_ic_t > 10
    assert volume.partial_ic_t == pytest.approx(volume.ic_t, rel=0.2)


def test_a_reference_store_that_starts_inside_the_segment_is_read_from_its_first_bar(market, tmp_path):
    kmid = Alpha158Stock(FactorConfig(
        warmup_bars=WARMUP, dataset=market["prices"], mode="batch", data_columns=INPUTS,
        factor_names=("KMID",), njobs=4, file_path=str(tmp_path / "kmid.zarr"),
    ))
    kmid.build("2011-01-03", synthetic_market.END)

    [noisy] = _evaluator(market, [kmid]).evaluate([_graph("alpha = (close - open) / (high - low)")])

    assert noisy.nearest == "KMID (Alpha158Stock)"
    assert noisy.nearest_correlation > 0.4


def test_a_reference_store_that_starts_after_the_segment_adds_nothing(market, tmp_path):
    kmid = Alpha158Stock(FactorConfig(
        warmup_bars=WARMUP, dataset=market["prices"], mode="batch", data_columns=INPUTS,
        factor_names=("KMID",), njobs=4, file_path=str(tmp_path / "kmid.zarr"),
    ))
    kmid.build("2012-06-01", synthetic_market.END)
    graph = _graph("alpha = (close - open) / (high - low)")

    [alone] = _evaluator(market).evaluate([graph])
    [against_late] = _evaluator(market, [kmid]).evaluate([graph])

    assert against_late.nearest == ""
    assert against_late.partial_ic_t == pytest.approx(alone.partial_ic_t)


def test_without_a_pool_there_is_no_nearest_factor_and_the_partial_ic_is_the_ic(market):
    [metrics] = _evaluator(market).evaluate([_graph("alpha = volume / WindowedAvg(volume, 20)")])
    assert metrics.nearest == ""
    assert np.isnan(metrics.nearest_correlation)
    assert metrics.partial_ic_t == pytest.approx(metrics.ic_t, rel=1e-6)


def test_a_candidate_admitted_earlier_in_the_batch_is_in_the_pool_of_the_next(market):
    graphs = [_graph("alpha = volume / WindowedAvg(volume, 20)"), _graph("alpha = volume / WindowedAvg(volume, 10)")]
    seen = []

    def admit(position, metrics, ran):
        seen.append((position, ran))
        return "first volume" if position == 0 else None

    alone = _evaluator(market).evaluate(graphs)
    admitted = _evaluator(market).evaluate(graphs, admit)

    assert seen == [(0, 2), (1, 2)]
    assert _same(admitted[0], alone[0])
    assert admitted[1].nearest == "first volume"
    assert admitted[1].nearest_correlation > 0.9
    assert alone[1].partial_ic_t > 10
    assert abs(admitted[1].partial_ic_t) < 3


def test_a_one_tail_signal_has_flat_inner_deciles(market):
    [metrics] = _evaluator(market).evaluate([_graph("alpha = Min(open, close) / low")])
    returns = metrics.decile_returns
    assert len(returns) == 10
    assert returns[-1] - max(returns[:-1]) > 0.5 * (returns[-1] - min(returns))


def test_the_scoring_says_in_one_sentence_what_is_done(market):
    assert _scoring(market).describe() == (
        "A factor is ranked across the whole market each day, neutralised against the industry (industry) and "
        "the logarithm of the market cap (marketcap) and z-scored, and scored on the members of the research "
        "universe that day."
    )
    assert _scoring(market, neutralised=False, membership=False).describe() == (
        "A factor is ranked across the whole market each day, not neutralised, and scored on every symbol."
    )


def test_coverage_is_the_mean_daily_share_of_the_members_with_a_value(market):
    # close / open = exp(0.01 z): Log(100 Log(close / open) - 1) = Log(z - 1) has a value only
    # where z > 1, on about one member in six.
    full, sparse = _evaluator(market).evaluate([
        _graph("alpha = close / open"), _graph("alpha = Log(Log(close / open) * 100 - 1)"),
    ])
    assert full.coverage == pytest.approx(1.0)
    assert 0.1 < sparse.coverage < 0.25


def test_a_fill_does_not_raise_coverage_above_the_data_it_fills(market):
    sparse = "Log(Log(close / open) * 100 - 1)"
    unfilled, filled, nested = _evaluator(market).evaluate([
        _graph(f"alpha = {sparse}"),
        _graph(f"alpha = SetInfOrNanToValue({sparse}, 0)"),
        _graph(f"alpha = SetInfOrNanToValue(SetInfOrNanToValue({sparse}) + close / open)"),
    ])
    assert 0.1 < unfilled.coverage < 0.25
    assert filled.coverage == pytest.approx(unfilled.coverage)
    assert nested.coverage == pytest.approx(unfilled.coverage)


def test_a_fill_still_changes_the_scored_values(market):
    sparse = "Log(Log(close / open) * 100 - 1)"
    unfilled, filled = _evaluator(market).evaluate([
        _graph(f"alpha = {sparse}"), _graph(f"alpha = SetInfOrNanToValue({sparse}, 0)"),
    ])
    assert filled.ic_mean != pytest.approx(unfilled.ic_mean)


#: Coverage, IC mean and IC t of the sparse graph, recorded before coverage was measured without fills.
BEFORE_FILLS = (0.15613592233009707, 0.2830496958728959, 75.08359082679765)


def test_a_graph_without_fills_scores_as_before(market):
    [metrics] = _evaluator(market).evaluate([_graph("alpha = Log(Log(close / open) * 100 - 1)")])
    assert (metrics.coverage, metrics.ic_mean, metrics.ic_t) == pytest.approx(BEFORE_FILLS)


def _known(market, names) -> Alpha158Stock:
    return Alpha158Stock(FactorConfig(
        warmup_bars=WARMUP, dataset=market["prices"], mode="batch", data_columns=INPUTS,
        factor_names=names, njobs=4,
    ))


def _against_known(market, known, reference=()) -> Evaluator:
    return Evaluator.for_segment(
        Segment("training", *synthetic_market.TRAINING),
        prices=market["prices"], label=market["label"], scoring=_scoring(market),
        reference=reference, known=known, warmup_bars=WARMUP, njobs=4,
    )


def test_a_rewrite_of_a_known_factor_has_a_sampled_correlation_near_one_and_is_named(market):
    evaluator = _against_known(market, [_known(market, ("KMID", "STD20"))])

    [rewrite] = evaluator.evaluate([_graph("alpha = close / open - 1")])

    assert rewrite.nearest_known == "KMID (Alpha158Stock, a known factor)"
    assert rewrite.nearest_known_correlation > 0.95
    # A known factor is not in the reference pool: nothing to be nearest to, nothing residualised.
    assert rewrite.nearest == ""
    assert rewrite.partial_ic_t == pytest.approx(rewrite.ic_t, rel=1e-6)


#: Every fifth scored date of the training segment, from the first: the synthetic market trades every
#: business day and every date before the purge is scored.
SAMPLED = pd.bdate_range(*synthetic_market.TRAINING)[::5]


class _SampleProbe(FactorPolars):
    """The intraday move only on ``SAMPLED`` (``on_sample``) or only off it (``off_sample``), noise elsewhere."""

    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        move = pl.col("adjClose") / pl.col("adjOpen") - 1.0
        noise = (pl.col("adjVolume") * 7919.0).mod(1.0)  # the fractional digits of the volume
        sampled = pl.col("timestamp").is_in(pl.Series(SAMPLED.to_numpy()).cast(pl.Datetime("ns")))
        return lf.with_columns(
            pl.when(sampled).then(move).otherwise(noise).alias("on_sample"),
            pl.when(sampled).then(noise).otherwise(move).alias("off_sample"),
        ).select(["timestamp", "symbol", "on_sample", "off_sample"])


def _probe(market, name) -> _SampleProbe:
    return _SampleProbe(FactorConfig(
        warmup_bars=0, dataset=market["prices"], mode="batch", data_columns=INPUTS, factor_names=(name,),
    ))


def test_the_known_factors_are_compared_on_every_fifth_scored_date_only_the_same_on_a_second_evaluator(market):
    graph = _graph("alpha = close / open - 1")

    def against(name):
        return Evaluator.for_segment(
            Segment("training", *synthetic_market.TRAINING),
            prices=market["prices"], label=market["label"], scoring=_scoring(market, neutralised=False),
            known=[_probe(market, name)], warmup_bars=WARMUP, njobs=4,
        )

    [on_sample], [again], [off_sample] = (against(n).evaluate([graph]) for n in ("on_sample", "on_sample", "off_sample"))

    assert on_sample.nearest_known_correlation == pytest.approx(1.0, abs=1e-3)
    assert abs(off_sample.nearest_known_correlation) < 0.1
    assert _same(on_sample, again)


def test_known_factors_change_neither_the_partial_ic_nor_the_correlation_with_the_reference_set(market, against_kmid):
    codes = ["alpha = (close - open) / (high - low)", "alpha = WindowedAvg(close / open, 5)",
             "alpha = volume / WindowedAvg(volume, 20)"]
    evaluator = _against_known(market, [_known(market, ("KMID", "STD20", "VSTD5"))], reference=[_kmid(market)])

    results = evaluator.evaluate([_graph(code) for code in codes])

    for result, before in zip(results, against_kmid.values()):
        assert result.nearest_known.endswith("(Alpha158Stock, a known factor)")
        assert json.dumps(_without_known(result)) == json.dumps(_without_known(before))


class _VolumeTilted(FactorKunQuant):
    """The intraday move tilted by the volume shock, ranked: a component (no store) neutralisation changes."""

    def _get_factor_names(self) -> tuple[str, ...]:
        return ("tilted",)

    def _get_factor_func(self) -> Function:
        builder = Builder()
        with builder:
            close, open_, volume = Input("adjClose"), Input("adjOpen"), Input("adjVolume")
            Output(Rank((close - open_) / open_ + Log(volume / WindowedAvg(volume, 20)) * 0.1), "tilted")
        return Function(builder.ops)


def test_a_known_factor_given_as_a_component_is_neutralised_as_its_rewrite_is(market):
    # Neutralised against the log volume, the scoring takes the tilt out of a candidate; a known factor
    # given as a component (computed, not read from a store) must be scored the same way to match it.
    neutralization = Neutralization([market["prices"], market["exposures"]], "adjVolume", "industry")
    tilted = _VolumeTilted(FactorConfig(
        warmup_bars=WARMUP, dataset=market["prices"], mode="batch", data_columns=INPUTS, njobs=4,
    ))
    evaluator = Evaluator.for_segment(
        Segment("training", *synthetic_market.TRAINING),
        prices=market["prices"], label=market["label"],
        scoring=Scoring(market["membership"], neutralization, njobs=4),
        known=[tilted], warmup_bars=WARMUP, njobs=4,
    )

    [rewrite] = evaluator.evaluate([_graph("alpha = close / open - 1 + Log(volume / WindowedAvg(volume, 20)) / 10")])

    assert rewrite.nearest_known == "tilted (_VolumeTilted, a known factor)"
    assert rewrite.nearest_known_correlation > 0.95


def _without_known(metrics: Metrics) -> dict:
    return {k: v for k, v in metrics.to_dict().items() if not k.startswith("nearest_known")}


def test_without_known_factors_there_is_no_nearest_known_factor(against_kmid):
    noisy = against_kmid["noisy copy"]
    assert noisy.nearest_known == ""
    assert np.isnan(noisy.nearest_known_correlation)


def _as_reference(market, codes: dict[str, str]) -> GraphFactor:
    """The graphs of ``codes`` (name to code) as one reference factor, ranked as a candidate is."""
    graphs = {name: _graph(code).to_json() for name, code in codes.items()}
    return GraphFactor(FactorConfig(
        warmup_bars=WARMUP, dataset=market["prices"], mode="batch", data_columns=GraphFactor.data_columns(graphs),
        njobs=4, kwargs={"graphs": graphs, "classes": price_volume().class_paths(list(graphs.values()))},
    ))


def _numbers(metrics: Metrics) -> np.ndarray:
    """The metrics' statistics, without the factor names."""
    return np.array([
        metrics.ic_mean, metrics.ic_t, metrics.partial_ic_mean, metrics.partial_ic_t, metrics.nearest_correlation,
        *metrics.decile_returns, metrics.coverage, metrics.n_days,
    ])


def test_admissions_across_and_within_batches_score_as_a_fresh_evaluator_with_them_in_its_reference_set(market):
    admitted = {
        "first_volume": "alpha = volume / WindowedAvg(volume, 20)",
        "range": "alpha = (high - low) / close",
        "averaged_intraday": "alpha = WindowedAvg(close / open, 5)",
    }
    later = [
        _graph("alpha = volume / WindowedAvg(volume, 10)"),
        _graph("alpha = (close - open) / (high - low)"),
        _graph("alpha = WindowedAvg((high - low) / close, 10)"),
    ]
    names = list(admitted)
    evaluator = _evaluator(market, [_kmid(market)])
    evaluator.evaluate([_graph(admitted["first_volume"])], lambda position, metrics, ran: "first_volume")
    # The second batch admits both graphs, so the second is scored against the first within the batch.
    evaluator.evaluate([_graph(admitted[name]) for name in names[1:]], lambda position, metrics, ran: names[1 + position])
    grown = evaluator.evaluate(later)

    fresh = _evaluator(market, [_kmid(market), _as_reference(market, admitted)]).evaluate(later)

    for after, expected in zip(grown, fresh):
        np.testing.assert_allclose(_numbers(after), _numbers(expected), rtol=0, atol=1e-6)
        assert after.nearest == expected.nearest.removesuffix(" (GraphFactor)")



def test_the_coverage_trace_shows_an_infinite_input_emptying_a_cross_sectional_operator(market):
    # Min(open, close) - open is 0 wherever the close is at or above the open, about half the
    # member days, and dividing by it is infinite there; the z-score of a day with one infinite
    # value has no value on any member.
    graph = _graph("alpha = CrossSectionalZScore(close / (Min(open, close) - open))")
    [metrics] = _evaluator(market).evaluate([graph])
    daily, trace = _evaluator(market).trace_coverage(graph)
    by_op = {daily.nodes[t.node].op: t for t in trace}
    assert metrics.coverage == 0.0
    assert by_op["CrossSectionalZScore"].coverage == 0.0
    assert by_op["CrossSectionalZScore"].inputs == (by_op["Div"].node,)
    assert 0.4 < by_op["Div"].infinite < 0.6
    assert by_op["Div"].coverage == pytest.approx(1 - by_op["Div"].infinite)
    assert by_op["Div"].infinite_days == 1.0
    assert all(t.coverage == 1.0 and t.infinite == 0.0 for t in trace if t.node < by_op["Div"].node)


def test_the_coverage_trace_leaves_out_constants_and_nodes_of_a_loop(market):
    graph = _graph("alpha = WindowedLinearRegressionSlope(close, 10) / ConstantOp(2.0)")
    daily, trace = _evaluator(market).trace_coverage(graph)
    traced = {t.node for t in trace}
    assert len(daily.nodes) - 1 in traced
    assert all(daily.nodes[i].op != "ConstantOp" and daily.nodes[i].parent is None for i in traced)


def test_a_graph_measuring_what_a_tried_trial_measured_is_nearest_to_it(market):
    evaluator = _evaluator(market)
    evaluator.evaluate([_graph("alpha = close / open"), _graph("alpha = volume / WindowedAvg(volume, 20)")],
                       tried=["T1-01", "T1-02"])
    [again, other, broken] = evaluator.nearest_tried([
        _graph("alpha = Log(close / open) * 2"), _graph("alpha = high / Max(open, close)"), UNCOMPILABLE,
    ])
    assert again[0] == "T1-01" and again[1] == pytest.approx(1.0, abs=1e-3)
    assert abs(other[1]) < 0.5
    assert isinstance(broken, CompileFailure)


def test_the_tried_matrix_holds_every_pair_once_whichever_way_the_trials_joined(market):
    graphs = [_graph("alpha = close / open"), _graph("alpha = volume / WindowedAvg(volume, 20)"),
              _graph("alpha = high / Max(open, close)")]
    one = _evaluator(market)
    one.evaluate(graphs[:2], tried=["a", "b"])
    one.evaluate(graphs[2:], tried=["c"])
    other = _evaluator(market)
    assert other.join(tried=list(zip(["a", "x", "b", "c"], [graphs[0], UNCOMPILABLE, *graphs[1:]]))) == ["x"]
    names, matrix = one.tried()
    assert names == other.tried()[0] == ["a", "b", "c"]
    assert matrix == pytest.approx(other.tried()[1], abs=1e-6)
    assert matrix == pytest.approx(matrix.T)
    assert np.diag(matrix) == pytest.approx(1.0, abs=1e-6)


def test_a_graph_that_did_not_compute_does_not_join_the_tried_trials(market):
    evaluator = _evaluator(market)
    evaluator.evaluate([UNCOMPILABLE, _graph("alpha = close / open")], tried=["x", "T1-02"])
    assert evaluator.tried()[0] == ["T1-02"]


def test_without_tried_trials_there_is_no_nearest_one(market):
    [(name, correlation)] = _evaluator(market).nearest_tried([_graph("alpha = close / open")])
    assert name == "" and np.isnan(correlation)


def _stored(market, store, reference=(), prices=None, store_key=None) -> Evaluator:
    """An evaluator keeping its columns in ``store``; with ``prices``, reading those."""
    return Evaluator.for_segment(
        Segment("training", *synthetic_market.TRAINING),
        prices=prices or market["prices"], label=market["label"], scoring=_scoring(market),
        reference=reference, warmup_bars=WARMUP, njobs=4, store=store, store_key=store_key or {"data": "synthetic"},
    )


class _Unreadable:
    """A price dataset that raises on every read while ``readable`` is false, as one would if the store were gone."""

    def __init__(self, prices, readable=False):
        self._prices = prices
        self.readable = readable

    def __getattr__(self, name):
        if not self.readable and name in ("panel", "to_kunquant", "head", "own_names", "to_shared_names"):
            raise AssertionError(f"the prices were read ({name})")
        return getattr(self._prices, name)


def test_a_factor_joined_from_the_store_reads_no_prices(market, tmp_path):
    graph = _graph("alpha = volume / WindowedAvg(volume, 20)")
    first = _stored(market, tmp_path / "columns")
    first.evaluate([graph], lambda position, metrics, ran: "volume")

    again = _stored(market, tmp_path / "columns", prices=_Unreadable(market["prices"]))
    assert again.join(pool=[("volume", graph)]) == []
    assert again.contributions(["volume"]) == first.contributions(["volume"])


def test_factors_joined_from_the_store_score_the_next_batch_as_computed_ones(market, tmp_path):
    accepted = [_graph("alpha = volume / WindowedAvg(volume, 20)"), _graph("alpha = (high - low) / close")]
    trials = [*accepted, _graph("alpha = close / open")]
    later = [_graph("alpha = volume / WindowedAvg(volume, 10)"), _graph("alpha = WindowedAvg((high - low) / close, 5)")]
    names = ["volume", "range"]
    first = _stored(market, tmp_path / "columns", [_kmid(market)])
    first.evaluate(trials, lambda position, metrics, ran: names[position] if position < 2 else None, ["a", "b", "c"])

    prices = _Unreadable(market["prices"])
    resumed = _stored(market, tmp_path / "columns", [_kmid(market)], prices=prices)
    assert resumed.join(pool=list(zip(names, accepted)), tried=list(zip("abc", trials))) == []
    computed = _evaluator(market, [_kmid(market)])
    assert computed.join(pool=list(zip(names, accepted)), tried=list(zip("abc", trials))) == []

    assert resumed.tried()[0] == computed.tried()[0] == ["a", "b", "c"]
    np.testing.assert_array_equal(resumed.tried()[1], computed.tried()[1])
    np.testing.assert_array_equal(resumed.tried()[1], first.tried()[1])
    prices.readable = True  # the later batch is computed
    again = resumed.evaluate(later)
    for metrics in (first.evaluate(later), computed.evaluate(later)):
        assert all(_same(a, b) for a, b in zip(metrics, again, strict=True))


def test_a_store_written_under_other_settings_is_emptied_and_written_again(market, tmp_path):
    graph = _graph("alpha = volume / WindowedAvg(volume, 20)")
    unreadable = _Unreadable(market["prices"])
    _stored(market, tmp_path / "columns", store_key={"data": "one"}).evaluate([graph], lambda *_: "volume")

    assert _stored(market, tmp_path / "columns", store_key={"data": "two"}, prices=unreadable).join(
        pool=[("volume", graph)],
    ) == ["volume"]
    assert _stored(market, tmp_path / "columns", store_key={"data": "two"}).join(pool=[("volume", graph)]) == []
    assert _stored(market, tmp_path / "columns", store_key={"data": "two"}, prices=unreadable).join(
        pool=[("volume", graph)],
    ) == []
    assert _stored(market, tmp_path / "columns", store_key={"data": "one"}, prices=unreadable).join(
        pool=[("volume", graph)],
    ) == ["volume"]


def test_a_graph_of_the_last_batch_joins_without_being_computed_again_and_is_stored(market, tmp_path):
    graphs = [_graph("alpha = close / open"), _graph("alpha = volume / WindowedAvg(volume, 20)")]
    prices = _Unreadable(market["prices"], readable=True)
    evaluator = _stored(market, tmp_path / "columns", prices=prices)
    evaluator.evaluate(graphs)
    prices.readable = False
    assert evaluator.join(pool=[("volume", graphs[1])]) == []
    assert evaluator.contributions(["volume"])["volume"] is not None

    unreadable = _stored(market, tmp_path / "columns", prices=_Unreadable(market["prices"]))
    assert unreadable.join(pool=[("volume", graphs[1])]) == []
    assert unreadable.join(pool=[("never admitted", graphs[0])]) == ["never admitted"]


def test_a_tried_trial_joining_the_pool_later_is_stored_as_a_pool_factor_once_computed(market, tmp_path):
    graph = _graph("alpha = volume / WindowedAvg(volume, 20)")
    _stored(market, tmp_path / "columns").evaluate([graph], tried=["T1-01"])

    assert _stored(market, tmp_path / "columns").join(pool=[("volume", graph)], tried=[("T1-01", graph)]) == []
    unreadable = _stored(market, tmp_path / "columns", prices=_Unreadable(market["prices"]))
    assert unreadable.join(pool=[("volume", graph)], tried=[("T1-01", graph)]) == []
