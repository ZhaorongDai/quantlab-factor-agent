"""Evaluator.evaluate(): neutralisation, compile-failure bisection, the reference pool and admission order.

Graphs come from the real Gate and are evaluated on the synthetic store of
``tests/synthetic_market.py``; the reference set is Alpha158's KMID, which
is the planted intraday signal ``close / open`` up to a constant.
"""

import json

import numpy as np
import pandas as pd
import pytest
from KunQuant.Op import Builder, Input, Output, Rank
from KunQuant.Stage import Function
from quantlab.analysis.factor_report import newey_west_lags, newey_west_t_stat
from quantlab.core.component import rebuild
from quantlab.factor.config import FactorConfig, NeutralizedConfig
from quantlab.factor.kunquant import FactorKunQuant
from quantlab.factor.kunquant_compile import CACHE_ENV
from quantlab.factor.predefined.alpha158 import Alpha158Stock
from quantlab.factor.predefined.neutralized import NeutralizedFactor
from quantlab.model.evaluation import cross_sectional_rank_ic_series

from factor_agent.candidate import Candidate
from factor_agent.config import GateConfig
from factor_agent.evaluator import CompileFailure, Evaluator, Metrics
from factor_agent.gate import Gate
from factor_agent.gate.graph import Graph
from factor_agent.segments import Segment
from factor_agent.zoo import Zoo
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


def _evaluator(market, reference=()) -> Evaluator:
    return Evaluator.for_segment(
        Segment("training", *synthetic_market.TRAINING),
        prices=market["prices"], label=market["label"], membership=market["membership"],
        exposures=market["exposures"], reference=reference, warmup_bars=WARMUP, njobs=4,
    )


def _kmid(market) -> Alpha158Stock:
    return Alpha158Stock(FactorConfig(
        warmup_bars=WARMUP, dataset=market["prices"], mode="batch", data_columns=INPUTS,
        factor_names=("KMID",), njobs=4,
    ))


def _graph(code: str) -> Graph:
    candidate = Candidate.from_dict({
        "hypothesis": "A test hypothesis.", "mechanism_type": "behavioural_bias", "expected_sign": 1,
        "horizon_rationale": "A week.", "reference": "Synthetic", "description": "test", "code": code,
    })
    checked = Gate(GateConfig()).check(candidate, Zoo())
    assert not hasattr(checked, "reasons"), checked
    return checked.graph


#: A graph that passes from_json but that KunQuant cannot compile: its window is not a number.
UNCOMPILABLE = Graph.from_json([["Input", [["name", "adjClose"]], [], None], ["WindowedAvg", [["window", "abc"]], [0], None]])


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


def _ic_of(market, panel) -> tuple[float, float]:
    """Mean daily rank IC and its NW t of ``panel`` masked to members, on the dates the evaluator scores."""
    label_factor = market["label"]
    label = label_factor.read(*synthetic_market.TRAINING)[label_factor.get_factor_names()[0]]
    label = label.isel(timestamp=slice(0, label.sizes["timestamp"] - label_factor.lookahead_bars()))
    values = market["membership"].mask(panel)["intraday"].reindex(timestamp=label["timestamp"], symbol=label["symbol"])
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
