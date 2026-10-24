"""Gate.check: every bad candidate comes back Rejected with a readable reason.

The graph is built in a real subprocess; nothing is faked.
"""

import importlib
import pkgutil
import sys

import KunQuant
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from KunQuant.Op import Input, OpBase

from factor_agent.candidate import Candidate
from factor_agent.config import GateConfig
from factor_agent.gate import Gate, Passed, Rejected
from factor_agent.zoo import Zoo
from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.stock import StockDataset
from quantlab.factor.config import FactorConfig, NeutralizedConfig, PolarsFactorConfig
from quantlab.factor.predefined.alpha158 import Alpha158Stock
from quantlab.factor.predefined.momentum import Momentum
from quantlab.factor.predefined.neutralized import NeutralizedFactor
from tests.vocabulary_files import price_volume


def _candidate(code: str, hypothesis: str = "Intraday strength carries into the next week.") -> Candidate:
    return Candidate(
        hypothesis=hypothesis,
        mechanism_type="behavioural_bias",
        expected_sign=1,
        horizon_rationale="Under-reaction resolves over a few days.",
        description="Close over open.",
        code=code,
    )


def _check(code: str, **limits):
    return Gate(GateConfig(timeout_seconds=10.0, **limits), price_volume()).check(_candidate(code), Zoo())


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        ("alpha = close /", "syntax error"),
        ("import os\nalpha = close", "import"),
        ("from math import log\nalpha = close", "import"),
        ("alpha = SecretOp(close)", "SecretOp"),
        ("alpha = Input('adjClose')", "Input"),
        ("alpha = open(close)", "graph build failed"),
        ("alpha = close.__class__", "attribute access"),
        ('alpha = "{0.__class__}".format(close)', "attribute access"),
        ("f = lambda x: x\nalpha = f(close)", "Lambda"),
        ("class A:\n    pass\nalpha = close", "class"),
        ("alpha = (close / open, volume)", "exactly one output"),
        ("beta = close", "alpha"),
        ("alpha = 1.0", "operator"),
        ("alpha = Div(close, 0)", "graph build failed"),
    ],
)
def test_a_malformed_candidate_is_rejected_with_a_readable_reason(code, reason):
    result = _check(code)
    assert isinstance(result, Rejected)
    assert any(reason in r for r in result.reasons), result.reasons


# Each static constraint with a candidate that passes it and one that breaks
# it: (passing code, failing code, the failing reason, the gate's limits).
CONSTRAINTS = {
    "window": ("alpha = WindowedAvg(close, 20) / close", "alpha = WindowedAvg(close, 21) / close",
               "window 21 is not one of the allowed values", {}),
    "loop window": (
        "loop = ForeachBackWindow(close, 10)\nwith Builder(loop):\n    item = IterValue(loop, close)\n"
        "alpha = ReduceMax(item) / close",
        "loop = ForeachBackWindow(close, 7)\nwith Builder(loop):\n    item = IterValue(loop, close)\n"
        "alpha = ReduceMax(item) / close",
        "window 7 is not one of the allowed values", {}),
    "lag": ("alpha = close / BackRef(close, 1)", "alpha = close / BackRef(close, 3)",
            "lag 3 is not one of the allowed values", {}),
    "quantile": ("alpha = WindowedQuantile(close, 20, 0.75) / close",
                 "alpha = WindowedQuantile(close, 20, 0.7) / close",
                 "quantile 0.7 is not one of the allowed values", {}),
    "half-life": ("alpha = EWMean(close / open, 60, 20)", "alpha = EWMean(close / open, 60, 17.5)",
                  "half_life 17.5 is not one of the allowed values", {}),
    "winsorize bound": ("alpha = CrossSectionalWinsorize(close / open, 0.05, 0.95)",
                        "alpha = CrossSectionalWinsorize(close / open, 0.07, 0.95)",
                        "lower 0.07 is not one of the allowed values", {}),
    "constant": ("alpha = Rank(close / open) - 0.5", "alpha = Rank(close / open) - 0.37",
                 "constant 0.37 is not one of the allowed values", {}),
    "chained constants": ("alpha = Rank(close / open) * 0.25", "alpha = Rank(close / open) * 0.25 * 1.5",
                          "the constants combine to MulConst scale 0.375", {}),
    "chained lags": ("alpha = close / BackRef(BackRef(close, 5), 5)",
                     "alpha = close / BackRef(BackRef(close, 5), 10)",
                     "the constants combine to BackRef lag 15", {}),
    "node count": ("x = close\nfor i in range(150):\n    x = x + open\nalpha = x / close",
                   "x = close\nfor i in range(250):\n    x = x + open\nalpha = x / close",
                   "more than the limit of 200", {}),
    "free-parameter ratio": ("alpha = (close - open) / (high - low + 1e-12)",
                             "alpha = (close * 2 + open * 3) / (high * 5 + low * 4)",
                             "free-parameter ratio", {}),
    "base fields": ("alpha = (close - open) / (high - low)", "alpha = (close - open) / (high - low) * volume",
                    "reads 5 input fields", {"max_base_fields": 4}),
    "dimension": ("alpha = close / open + volume / WindowedAvg(volume, 20)", "alpha = close + volume",
                  "Add combines a price with a volume", {}),
    "comparison output": ("alpha = Select(close > open, ConstantOp(1.0), ConstantOp(-1.0))",
                          "alpha = close > open", "alpha is a comparison", {}),
    "comparison as number": ("alpha = Select(close > open, close, open) / vwap",
                             "alpha = (close > open) * close / open", "takes a number but gets a comparison", {}),
    "constant output": ("alpha = Rank(close / open)", "alpha = close * 0 + ConstantOp(1.0)",
                        "alpha is a constant", {}),
}


@pytest.mark.parametrize("rule", CONSTRAINTS)
def test_each_static_constraint_passes_one_candidate_and_rejects_another(rule):
    good, bad, reason, limits = CONSTRAINTS[rule]
    passed = _check(good, **limits)
    assert isinstance(passed, Passed), getattr(passed, "reasons", None)
    rejected = _check(bad, **limits)
    assert isinstance(rejected, Rejected)
    assert any(reason in r for r in rejected.reasons), rejected.reasons


def test_code_that_never_finishes_is_stopped_by_the_timeout():
    gate = Gate(GateConfig(timeout_seconds=1.0), price_volume())
    result = gate.check(_candidate("while True:\n    pass\nalpha = close"), Zoo())
    assert isinstance(result, Rejected)
    assert any("timed out" in r for r in result.reasons), result.reasons


def test_code_that_allocates_past_the_memory_limit_is_rejected():
    gate = Gate(GateConfig(timeout_seconds=30.0, memory_mb=256), price_volume())
    result = gate.check(_candidate("x = list(range(10 ** 8))\nalpha = close / open"), Zoo())
    assert isinstance(result, Rejected)
    assert any("256 MB memory limit" in r for r in result.reasons), result.reasons
    # The gate still works afterwards.
    assert isinstance(gate.check(_candidate("alpha = close / open"), Zoo()), Passed)


def test_a_well_formed_candidate_passes_with_its_graph_and_statistics():
    code = (
        "def strength(x, y):\n"
        "    return x / y\n"
        "ratio = strength(close, open)\n"
        "alpha = Rank(WindowedAvg(ratio, 5))\n"
    )
    result = _check(code)
    assert isinstance(result, Passed), getattr(result, "reasons", None)
    assert result.graph.input_names == ("adjClose", "adjOpen")
    assert result.stats == {
        "nodes": 5, "operators": 3, "free_parameters": 0, "free_parameter_ratio": 0.0,
        "base_fields": ["close", "open"], "dimension": "pure number",
    }
    assert len(result.fingerprint) == 64


def test_an_unknown_dimension_is_passed_and_recorded():
    result = _check("alpha = Log(close) - Log(BackRef(close, 5)) + Log(volume)")
    assert isinstance(result, Passed), getattr(result, "reasons", None)
    assert result.stats["dimension"] == "unknown"


def test_derived_inputs_count_as_one_field_each():
    result = _check("alpha = vwap / close + amount / WindowedAvg(amount, 20)")
    assert isinstance(result, Passed), getattr(result, "reasons", None)
    assert result.stats["base_fields"] == ["amount", "close", "vwap"]
    assert result.stats["dimension"] == "pure number"


def test_a_nan_sensitive_operator_keeps_a_missing_bar_missing():
    result = _check("alpha = Max(close / open, ConstantOp(1.0))")
    assert isinstance(result, Passed), getattr(result, "reasons", None)
    # quantlab's stand-in adds 0 * every input, so a bar with no data stays NaN.
    assert result.graph.nodes[-1].op == "Add"
    assert set(result.graph.input_names) == {"adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"}
    assert result.stats["base_fields"] == ["close", "open"]


def test_the_same_graph_gets_the_same_fingerprint():
    one = _check("alpha = close / open")
    two = _check("x = close\ny = open\nalpha = x / y")
    other = _check("alpha = open / close")
    assert one.fingerprint == two.fingerprint
    assert one.fingerprint != other.fingerprint


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("alpha = Rank(WindowedAvg(close, 5) + WindowedAvg(open, 20))",
         "alpha = Rank(WindowedAvg(open, 20) + WindowedAvg(close, 5))"),
        ("alpha = WindowedCorrelation(close, 20, volume)", "alpha = WindowedCorrelation(volume, 20, close)"),
        ("alpha = Rank(close / open) * 4", "alpha = Rank(close / open) * (ConstantOp(2.0) * 2)"),
        ("alpha = (close - open) / 2", "alpha = (close - open) * 0.5"),
        ("alpha = Rank(close / open)", "alpha = Rank(close / open + 0)"),
        ("a = WindowedAvg(close, 5)\nalpha = a / a", "alpha = WindowedAvg(close, 5) / WindowedAvg(close, 5)"),
    ],
)
def test_a_rewritten_duplicate_is_rejected_by_its_canonical_fingerprint(first, second):
    gate = Gate(GateConfig(timeout_seconds=10.0), price_volume())
    one = gate.check(_candidate(first), Zoo())
    assert isinstance(one, Passed), getattr(one, "reasons", None)
    two = gate.check(_candidate(second), Zoo())
    assert isinstance(two, Rejected)
    assert any("duplicate" in r for r in two.reasons), two.reasons



@pytest.mark.parametrize(
    "second",
    [
        "alpha = MulConst(WindowedAvg(close / open, 20), -1)",
        "alpha = 0 - WindowedAvg(close / open, 20)",
        "alpha = Rank(WindowedAvg(close / open, 20)) * 3 + 1",
    ],
)
def test_a_candidate_that_only_keeps_or_reverses_an_earlier_ranking_is_a_duplicate(second):
    first = Gate(GateConfig(timeout_seconds=10.0), price_volume()).check(
        _candidate("alpha = WindowedAvg(close / open, 20)"), Zoo(),
    )
    assert isinstance(first, Passed), getattr(first, "reasons", None)
    gate = Gate(GateConfig(timeout_seconds=10.0), price_volume(), seen=[first.graph])
    result = gate.check(_candidate(second), Zoo())
    assert isinstance(result, Rejected)
    assert any("keeps or reverses the ranking of an earlier candidate" in r for r in result.reasons), result.reasons

def test_a_loop_body_is_kept_in_the_graph():
    code = (
        "ratio = close / open\n"
        "loop = ForeachBackWindow(ratio, 5)\n"
        "with Builder(loop):\n"
        "    item = IterValue(loop, ratio)\n"
        "alpha = ReduceAdd(item)\n"
    )
    result = _check(code)
    assert isinstance(result, Passed), getattr(result, "reasons", None)
    assert result.stats["nodes"] == 6


@pytest.fixture(scope="module")
def alpha158(tmp_path_factory) -> Zoo:
    """A zoo of every Alpha158Stock feature and a Polars momentum factor.

    Only the Polars factor reads the store, to learn its column names.
    """
    root = tmp_path_factory.mktemp("alpha158")
    xr.Dataset(
        {"Close": (["timestamp", "symbol"], np.ones((3, 2)))},
        coords={"timestamp": pd.bdate_range("2024-01-01", periods=3), "symbol": ["A", "B"]},
    ).to_zarr(root / "prices.zarr")
    dataset = StockDataset(DatasetConfig(
        zarr_file_path=str(root / "prices.zarr"), raw_data_dir_path=str(root), market="us_equity", frequency="1d",
    ))
    factor = Alpha158Stock(FactorConfig(
        warmup_bars=60, dataset=dataset, mode="batch",
        data_columns=("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"),
    ))
    momentum = Momentum(PolarsFactorConfig(warmup_bars=20, dataset=dataset, kwargs={"n": 20}))
    return Zoo.from_reference([factor, momentum], price_volume())


@pytest.mark.parametrize(
    ("code", "feature", "subtree"),
    [
        ("alpha = (close - open) / open", "KMID", "Div(Sub(close, open), open)"),
        ("alpha = Rank((close - open) / open) * 2", "KMID", "Div(Sub(close, open), open)"),
        ("alpha = WindowedStddev(close, 20) / close", "STD20", "Div(WindowedStddev(close, 20), close)"),
        ("alpha = 1 - (close - open) / open", "KMID", "Div(Sub(close, open), open)"),
        ("alpha = WindowedStddev(close, 20) / close * -1", "STD20", "Div(WindowedStddev(close, 20), close)"),
    ],
)
def test_a_candidate_that_rewrites_an_alpha158_feature_is_rejected(alpha158, code, feature, subtree):
    result = Gate(GateConfig(), price_volume()).check(_candidate(code), alpha158)
    assert isinstance(result, Rejected)
    assert any(f"{feature} (Alpha158Stock)" in r and subtree in r for r in result.reasons), result.reasons


def test_a_candidate_sharing_a_large_subtree_with_alpha158_is_rejected_with_the_subtree(alpha158):
    code = (
        "co = WindowedCorrelation(close / BackRef(close, 1), 20, Log(volume / (BackRef(volume, 1) + 1e-12) + 1))\n"
        "alpha = co * Sign(close - open)\n"
    )
    result = Gate(GateConfig(), price_volume()).check(_candidate(code), alpha158)
    assert isinstance(result, Rejected)
    reason = next(r for r in result.reasons if "subtree" in r)
    assert "CORD20 (Alpha158Stock)" in reason
    assert "WindowedCorrelation(" in reason and "BackRef(volume, 1)" in reason


@pytest.mark.parametrize("code", ["alpha = Rank(WindowedAvg(close / open, 5))", "alpha = 1 - WindowedAvg(close / open, 5)"])
def test_a_candidate_unlike_any_alpha158_feature_passes(alpha158, code):
    result = Gate(GateConfig(), price_volume()).check(_candidate(code), alpha158)
    assert isinstance(result, Passed), getattr(result, "reasons", None)


def test_the_parameter_values_of_an_alpha158_feature_are_not_a_new_factor(alpha158):
    result = Gate(GateConfig(), price_volume()).check(_candidate("alpha = WindowedStddev(close, 40) / close"), alpha158)
    assert isinstance(result, Rejected)
    assert any("STD" in r and "parameter" in r for r in result.reasons), result.reasons


def test_reference_factors_without_a_kunquant_graph_are_listed_as_not_covered(alpha158):
    assert alpha158.not_covered == ("momentum_20 (Momentum)",)
    assert "KMID (Alpha158Stock)" in alpha158.covered


def test_a_neutralised_store_is_compared_by_the_graph_of_the_factor_it_neutralises(tmp_path):
    xr.Dataset(
        {"Close": (["timestamp", "symbol"], np.ones((3, 2)))},
        coords={"timestamp": pd.bdate_range("2024-01-01", periods=3), "symbol": ["A", "B"]},
    ).to_zarr(tmp_path / "prices.zarr")
    dataset = StockDataset(DatasetConfig(
        zarr_file_path=str(tmp_path / "prices.zarr"), raw_data_dir_path=str(tmp_path), market="us_equity", frequency="1d",
    ))
    ranked = Alpha158Stock(FactorConfig(
        warmup_bars=60, dataset=dataset, mode="batch",
        data_columns=("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"),
    ))
    neutral = NeutralizedFactor(NeutralizedConfig(factor=ranked, dataset=dataset, factor_names=("KMID", "STD20")))

    zoo = Zoo.from_reference([neutral], price_volume())

    assert zoo.covered == ("KMID (Alpha158Stock)", "STD20 (Alpha158Stock)")
    result = Gate(GateConfig(), price_volume()).check(_candidate("alpha = (close - open) / open"), zoo)
    assert isinstance(result, Rejected)
    assert any("KMID (Alpha158Stock)" in r for r in result.reasons), result.reasons


def _accepted(code: str) -> Zoo:
    passed = Gate(GateConfig(), price_volume()).check(_candidate(code), Zoo())
    assert isinstance(passed, Passed), getattr(passed, "reasons", None)
    zoo = Zoo()
    zoo.add("strength", "accepted in round 1", passed.graph)
    return zoo


def test_a_refinement_that_only_changes_windows_is_rejected_as_parameter_tuning():
    zoo = _accepted("alpha = WindowedAvg(close / open, 5) / WindowedStddev(close / open, 20)")
    result = Gate(GateConfig(), price_volume()).check(
        _candidate("alpha = WindowedAvg(close / open, 10) / WindowedStddev(close / open, 60)"), zoo,
    )
    assert isinstance(result, Rejected)
    assert any("strength (accepted in round 1)" in r and "parameter" in r for r in result.reasons), result.reasons


def test_a_refinement_that_changes_the_structure_passes():
    zoo = _accepted("alpha = WindowedAvg(close / open, 5) / WindowedStddev(close / open, 20)")
    result = Gate(GateConfig(), price_volume()).check(
        _candidate("alpha = DecayLinear(close / open, 5) / WindowedStddev(close / open, 20)"), zoo,
    )
    assert isinstance(result, Passed), getattr(result, "reasons", None)


@pytest.mark.parametrize(
    ("hypothesis", "code", "reason"),
    [
        ("Heavy trading volume after a strong day signals attention.", "alpha = close / open", "volume"),
        ("Share turnover proxies for disagreement.", "alpha = close / open", "turnover"),
        ("The closing price drifts toward the opening price.", "alpha = WindowedAvg(close, 5) / close", "opening price"),
        ("Return volatility is priced as risk.", "alpha = close / BackRef(close, 20)", "volatility"),
        ("The correlation of returns with volume reveals informed trading.",
         "alpha = (close / open) * volume / WindowedAvg(volume, 20)", "correlation"),
    ],
)
def test_a_hypothesis_naming_what_the_graph_never_uses_fails_mechanism_binding(hypothesis, code, reason):
    result = Gate(GateConfig(), price_volume()).check(_candidate(code, hypothesis), Zoo())
    assert isinstance(result, Rejected)
    assert any("mechanism binding" in r and reason in r for r in result.reasons), result.reasons


@pytest.mark.parametrize(
    ("hypothesis", "code"),
    [
        ("Heavy trading volume after a strong day signals attention.",
         "alpha = (close / open) * volume / WindowedAvg(volume, 20)"),
        ("Dollar volume, not share volume, measures attention.", "alpha = amount / WindowedAvg(amount, 20)"),
        ("Return volatility is priced as risk.", "alpha = EWVar(close / BackRef(close, 1), 60, 20)"),
        ("Stocks with a high price relative to peers attract lottery demand.", "alpha = close / open"),
        ("Price far above the volume-weighted average price reverts.", "alpha = vwap / close"),
        ("Stocks at the low end of the return distribution rebound.", "alpha = close / open"),
    ],
)
def test_a_hypothesis_bound_to_its_graph_passes(hypothesis, code):
    result = Gate(GateConfig(), price_volume()).check(_candidate(code, hypothesis), Zoo())
    assert isinstance(result, Passed), getattr(result, "reasons", None)


def _op_classes() -> set[type]:
    """Every OpBase subclass KunQuant and quantlab declare in a module."""
    for module in pkgutil.walk_packages(KunQuant.__path__, "KunQuant."):
        try:
            importlib.import_module(module.name)
        except ImportError:  # optional backends (MLIR, CUDA) that are not installed
            pass
    for name in ("kunquant_ts", "kunquant_cs", "kunquant_nan"):
        importlib.import_module(f"quantlab.factor.{name}")
    found, stack = set(), [OpBase]
    while stack:
        for sub in stack.pop().__subclasses__():
            stack.append(sub)
            # Classes made at run time (quantlab's per-bounds variants) are not declared
            # operators, and an underscore marks a library's private base class.
            declared = getattr(sys.modules.get(sub.__module__), sub.__name__, None) is sub
            if declared and not sub.__name__.startswith("_"):
                found.add(sub)
    return found


def _unlisted() -> set[type]:
    vocabulary = price_volume()
    return _op_classes() - set(vocabulary.operator_classes) - set(vocabulary.excluded) - {Input}


def test_the_example_operator_file_registers_or_excludes_every_operator_class():
    missing = sorted(f"{c.__module__}.{c.__name__}" for c in _unlisted())
    assert not missing, f"operators the example operator file neither registers nor excludes: {missing}"


def test_the_guard_notices_an_operator_in_neither_list():
    class NewOperator(OpBase):
        pass

    module = sys.modules[__name__]
    module.NewOperator = NewOperator
    try:
        assert NewOperator in _unlisted()
    finally:
        del module.NewOperator


def test_the_bar_mask_inside_a_reference_graph_is_no_shared_subtree():
    from KunQuant.Op import Builder
    from quantlab.factor import kunquant_nan
    from factor_agent.gate.graph import Graph

    vocabulary = price_volume()
    with Builder():
        o, h, l, c, v = (Input(x) for x in ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"))
        with kunquant_nan.missing_bars_only([o, h, l, c, v]):
            reference = kunquant_nan.SetInfOrNanToValue(c / o)
    zoo = Zoo()
    zoo.add("masked", "Reference", Graph.record(reference, vocabulary, strict=False))
    code = (
        "positive_slope = GreaterThanConst(WindowedLinearRegressionSlope(close, 60), 0)\n"
        "alpha = Select(positive_slope, close, ConstantOp(0))"
    )
    result = Gate(GateConfig(), vocabulary).check(_candidate(code), zoo)
    assert isinstance(result, Passed), getattr(result, "reasons", None)
