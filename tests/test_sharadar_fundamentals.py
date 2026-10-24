"""The repository's Sharadar fundamentals field file, with the price-volume one and the example operators.

Nothing reads server data: the field file's columns are checked against the
column names quantlab's Sharadar tables declare, the graphs are built
through a real Gate, and the zoo holds the real ``FundamentalAlpha`` graphs
over a store that only gives the factor a dataset to be built on.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.sharadar.tables import DAILY_INDICATORS, SF1_INDICATORS
from quantlab.dataset.stock import StockDataset
from quantlab.factor.config import FactorConfig
from quantlab.factor.predefined.fundamental_alpha import (
    OUTPUT_COLUMNS,
    FundamentalAlpha,
    FundamentalAlphaParameters,
)

from KunQuant.Op import Builder, ConstantOp, Input, OpBase
from KunQuant.ops import Abs, Select

from factor_agent.candidate import Candidate
from factor_agent.config import GateConfig
from factor_agent.gate import Gate, Passed, Rejected
from factor_agent.gate.graph import Graph
from factor_agent.gate.vocabulary import Vocabulary
from factor_agent.zoo import Zoo
from tests.vocabulary_files import CONFIGS, EXAMPLE_OPERATORS, PRICE_VOLUME

FUNDAMENTALS = CONFIGS / "fields" / "sharadar_fundamentals.yaml"

#: The columns of the merged SEP + DAILY + SF1 ART panel the campaign reads (SF1's own marketcap and ev
#: go by sf1_marketcap and sf1_ev there, so the plain names are DAILY's).
SEP_COLUMNS = ("open", "high", "low", "close", "volume", "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")
PANEL_COLUMNS = (*SEP_COLUMNS, *DAILY_INDICATORS, *SF1_INDICATORS, "reportperiod", "release_date")

#: SF1's per-share items and SEP's unadjusted prices: none may be a field (design Q47).
PER_SHARE = ("bvps", "dps", "eps", "epsdil", "epsusd", "fcfps", "price", "sps", "tbvps", "divyield", "sharefactor")
UNADJUSTED = ("open", "high", "low", "close", "volume")


@pytest.fixture(scope="module")
def vocabulary() -> Vocabulary:
    return Vocabulary.load([PRICE_VOLUME, FUNDAMENTALS], EXAMPLE_OPERATORS, columns=PANEL_COLUMNS)


def _candidate(code: str, hypothesis: str = "Cheap firms earn a premium.") -> Candidate:
    return Candidate(
        hypothesis=hypothesis,
        mechanism_type="risk_premium",
        expected_sign=1,
        horizon_rationale="Valuation spreads close slowly.",
        reference="Test reference",
        description="A fundamental ratio.",
        code=code,
    )


def _check(vocabulary: Vocabulary, code: str, hypothesis: str = "Cheap firms earn a premium.", zoo=None):
    return Gate(GateConfig(timeout_seconds=20.0), vocabulary).check(_candidate(code, hypothesis), zoo or Zoo())


def test_the_field_file_loads_with_the_price_volume_file_on_vendor_columns(vocabulary):
    fundamentals = [f for f in vocabulary.fields.values() if f.family == "sharadar_fundamentals"]
    assert 35 <= len(fundamentals) <= 45
    # Named after their vendor columns (design Q51), each with a unit and a description.
    assert all(f.name == f.column and f.dimension is not None and f.description for f in fundamentals)
    assert {"netinc", "revenue", "assets", "marketcap", "ev", "pe"} <= {f.name for f in fundamentals}
    read = {f.column for f in fundamentals}
    assert not read & set(PER_SHARE + UNADJUSTED)


@pytest.mark.parametrize(
    ("code", "dimension"),
    [
        ("alpha = netinc / marketcap", "pure number"),
        ("alpha = amount / marketcap", "pure number"),
        ("alpha = (netinc - PeriodLag(netinc, 4)) / assetsavg", "pure number"),
        ("alpha = Log(pe) - Log(evebit)", "pure number"),
        ("alpha = close / netinc", "1/shares"),
    ],
)
def test_dimensions_are_inferred_over_mixed_fields(vocabulary, code, dimension):
    result = _check(vocabulary, code)
    assert isinstance(result, Passed), getattr(result, "reasons", None)
    assert result.stats["dimension"] == dimension


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        ("alpha = close + netinc", "Add combines a price with a money"),
        ("alpha = amount / marketcap + marketcap", "Add combines a pure number with a money"),
    ],
)
def test_a_sum_of_dimensioned_fields_that_differ_is_refused(vocabulary, code, reason):
    result = _check(vocabulary, code)
    assert isinstance(result, Rejected)
    assert any(reason in r for r in result.reasons), result.reasons


EARNINGS = "Investors under-react to earnings, so high earnings relative to price predict returns."


def test_a_hypothesis_naming_earnings_needs_a_graph_reading_an_earnings_field(vocabulary):
    unbound = _check(vocabulary, "alpha = revenue / marketcap", EARNINGS)
    assert isinstance(unbound, Rejected)
    assert any("names earnings" in r and "netinc" in r for r in unbound.reasons), unbound.reasons
    bound = _check(vocabulary, "alpha = netinc / marketcap", EARNINGS)
    assert isinstance(bound, Passed), getattr(bound, "reasons", None)


@pytest.mark.parametrize(
    ("hypothesis", "code"),
    [
        ("Firms with high retained earnings are mature and safe.", "alpha = retearn / assets"),
        ("High asset turnover marks efficient firms.", "alpha = assetturnover"),
        ("Working capital tied up in operations predicts low returns.", "alpha = (assetsc - liabilitiesc) / assets"),
        ("Book-to-market captures distress risk.", "alpha = 1 / pb"),
    ],
)
def test_a_fundamental_hypothesis_bound_to_its_graph_passes(vocabulary, hypothesis, code):
    result = _check(vocabulary, code, hypothesis)
    assert isinstance(result, Passed), getattr(result, "reasons", None)


@pytest.mark.parametrize(
    ("hypothesis", "code", "reason"),
    [
        ("Accruals reverse as earnings quality is revealed.", "alpha = netinc / assetsavg", "accruals"),
        ("Working capital tied up in operations predicts low returns.", "alpha = assetsc / assets", "working capital"),
        ("High research and development spending is under-priced.", "alpha = sgna / marketcap", "R&D"),
    ],
)
def test_a_fundamental_hypothesis_naming_what_the_graph_never_reads_is_refused(vocabulary, hypothesis, code, reason):
    result = _check(vocabulary, code, hypothesis)
    assert isinstance(result, Rejected)
    assert any("mechanism binding" in r and reason in r for r in result.reasons), result.reasons


@pytest.fixture(scope="module")
def fundamental_alpha(tmp_path_factory, vocabulary) -> Zoo:
    """A zoo of every FundamentalAlpha output, recorded from its real KunQuant graph.

    The store only gives the factor a dataset; building the graph reads no data.
    """
    root = tmp_path_factory.mktemp("fundamental_alpha")
    xr.Dataset(
        {"adjClose": (["timestamp", "symbol"], np.ones((3, 2)))},
        coords={"timestamp": pd.bdate_range("2024-01-01", periods=3), "symbol": ["A", "B"]},
    ).to_zarr(root / "panel.zarr")
    dataset = StockDataset(DatasetConfig(
        zarr_file_path=str(root / "panel.zarr"), raw_data_dir_path=str(root), market="us_equity", frequency="1d",
    ))
    names = tuple(OUTPUT_COLUMNS)
    factor = FundamentalAlpha(FactorConfig(
        warmup_bars=600, dataset=dataset, mode="batch", factor_names=names,
        data_columns=FundamentalAlphaParameters().required_panel_columns(names),
    ))
    return Zoo.from_reference([factor], vocabulary)


@pytest.mark.parametrize(
    ("code", "factor", "subtree"),
    [
        ("alpha = (revenue - cor - sgna - intexp) / equityavg", "operating_profitability",
         "Sub(Sub(Sub(revenue, cor), sgna), intexp)"),
        ("alpha = ((ppnenet - PeriodLag(ppnenet, 4)) + (inventory - PeriodLag(inventory, 4))) / assets",
         "investment_to_assets", "PeriodLag(ppnenet, 4)"),
    ],
)
def test_a_candidate_rewriting_a_fundamental_alpha_factor_is_refused_by_name(vocabulary, fundamental_alpha, code,
                                                                            factor, subtree):
    result = _check(vocabulary, code, zoo=fundamental_alpha)
    assert isinstance(result, Rejected)
    assert any(f"{factor} (FundamentalAlpha)" in r and subtree in r for r in result.reasons), result.reasons


def test_a_fundamental_candidate_unlike_any_fundamental_alpha_factor_passes(vocabulary, fundamental_alpha):
    result = _check(vocabulary, "alpha = gp / assets", "Gross profitability predicts returns.", zoo=fundamental_alpha)
    assert isinstance(result, Passed), getattr(result, "reasons", None)



@pytest.mark.parametrize(
    ("code", "factor"),
    [
        ("alpha = fcf / marketcap", "fcf_yield"),
        ("alpha = Rank(rnd / marketcap)", "rd_to_market"),
        ("alpha = ncfo / assetsavg", "cash_profitability"),
        ("alpha = 0 - (netinc - ncfo) / assetsavg", "accruals"),
    ],
)
def test_a_plain_ratio_is_refused_by_name_as_a_rewrite_of_a_guarded_fundamental_alpha_ratio(
    vocabulary, fundamental_alpha, code, factor,
):
    result = _check(vocabulary, code, zoo=fundamental_alpha)
    assert isinstance(result, Rejected)
    assert any(f"rewrites {factor} (FundamentalAlpha)" in r for r in result.reasons), result.reasons


def _guarded(epsilon: float, absolute: bool, inner: bool, flipped: bool) -> OpBase:
    """``fcf / marketcap`` under a division guard on ``d > epsilon`` (or ``Abs(d)``) with a NaN fill.

    ``inner`` also guards the denominator itself to 1, as ``FundamentalAlpha``
    writes it; ``flipped`` tests ``d <= epsilon`` and puts the NaN first.
    """
    with Builder():
        n, d = Input("fcf"), Input("marketcap")
        guarded = Abs(d) if absolute else d
        valid = guarded > ConstantOp(epsilon)
        denominator = Select(valid, d, ConstantOp(1.0)) if inner else d
        if flipped:
            return Select(guarded <= ConstantOp(epsilon), ConstantOp("nan"), n / denominator)
        return Select(valid, n / denominator, ConstantOp("nan"))


@pytest.mark.parametrize(
    ("epsilon", "absolute", "inner", "flipped"),
    [(1e-9, False, True, False), (1e-6, True, True, False), (1e-12, False, False, False), (1e-9, True, False, True)],
)
def test_any_small_division_guard_is_seen_through(vocabulary, epsilon, absolute, inner, flipped):
    zoo = Zoo()
    zoo.add("guarded", "test", Graph.record(_guarded(epsilon, absolute, inner, flipped), vocabulary, strict=False))
    result = _check(vocabulary, "alpha = fcf / marketcap", zoo=zoo)
    assert isinstance(result, Rejected)
    assert any("rewrites guarded (test)" in r for r in result.reasons), result.reasons
