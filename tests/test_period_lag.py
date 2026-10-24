"""Period lags read from data: ``PeriodLag(x, k)`` on a synthetic periodic dataset.

Each symbol shows one quarterly report at a time: report r covers the period
ending 91 r days after a base date, is shown for 63 bars, and carries the
value 100 + r. A field ``k`` reports ago is the value of the k-th earlier
report shown, as quantlab's ``FundamentalAlpha`` reads it: missing where no
report is shown, and where that report's period does not end k periods
before the current one (a skipped or missing report in between). The lags
are declared in a field file, built through a real Gate and computed by
``GraphFactor``, the class of accepted factors, without its rank.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from KunQuant.Op import Output
from KunQuant.Stage import Function
from quantlab.dataset.config import DatasetConfig
from quantlab.dataset.stock import StockDataset
from quantlab.factor.config import FactorConfig
from quantlab.factor.kunquant_compile import CACHE_ENV
from quantlab.label.predefined.fret import Return

from factor_agent.candidate import Candidate
from factor_agent.config import GateConfig
from factor_agent.evaluator import Evaluator, Metrics
from factor_agent.exported import GraphFactor
from factor_agent.gate import Gate, Passed
from factor_agent.gate.graph import Graph
from factor_agent.gate.vocabulary import Vocabulary
from factor_agent.scoring import Scoring
from factor_agent.segments import Segment
from factor_agent.zoo import Zoo

BARS_PER_REPORT = 63
FIRST_REPORT_BAR = 20
#: Symbol A shows every report; B skips report 3; C shows nothing after report 5 for one report, then resumes.
REPORTS = {"A": list(range(12)), "B": [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11], "C": [0, 1, 2, 3, 4, 5, None, 7, 8, 9]}
BASE = pd.Timestamp("2009-12-31")

#: The store starts with the first bar, so a lag's warm-up before it is short, as the tests intend.
pytestmark = pytest.mark.filterwarnings("ignore:.*warm-up bar:UserWarning")

FIELDS = {
    "family": "statements",
    "description": "statement items",
    "period_lag": {
        "column": "{column}__q{k}",
        "k": [1, 4],
        "period": {"description": "one quarterly report", "key": "reportperiod", "days": 91,
                   "tolerance_days": 45, "bars": BARS_PER_REPORT},
    },
    "fields": [
        {"name": "netinc", "column": "netinc", "unit": "dollars", "description": "net income"},
        {"name": "price", "column": "adjClose", "unit": "dollars/shares", "description": "price"},
    ],
}

OPERATORS = {
    "parameters": {"constant": [0, 1]},
    "operators": [
        {"name": "Sub", "class": "KunQuant.ops.Sub", "signature": "Sub(x, y)", "inputs": ["series", "series"],
         "dimension": "match"},
        {"name": "Add", "class": "KunQuant.ops.Add", "signature": "Add(x, y)", "inputs": ["series", "series"],
         "dimension": "match"},
    ],
}


def _panel(timestamps: pd.DatetimeIndex) -> xr.Dataset:
    n = len(timestamps)
    rng = np.random.default_rng(3)
    netinc = np.full((n, len(REPORTS)), np.nan)
    key = np.full((n, len(REPORTS)), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    for j, reports in enumerate(REPORTS.values()):
        for slot, report in enumerate(reports):
            if report is None:
                continue
            bars = slice(FIRST_REPORT_BAR + slot * BARS_PER_REPORT, FIRST_REPORT_BAR + (slot + 1) * BARS_PER_REPORT)
            netinc[bars, j] = 100.0 + report
            key[bars, j] = (BASE + pd.Timedelta(days=91 * report)).to_datetime64()
    return xr.Dataset(
        {
            "netinc": (["timestamp", "symbol"], netinc),
            "reportperiod": (["timestamp", "symbol"], key),
            "adjClose": (["timestamp", "symbol"], np.full((n, len(REPORTS)), 10.0)),
            "adjOpen": (["timestamp", "symbol"], 10.0 * np.exp(np.cumsum(0.01 * rng.normal(size=netinc.shape), axis=0))),
        },
        coords={"timestamp": timestamps, "symbol": list(REPORTS)},
    )


@pytest.fixture(scope="module")
def periodic(tmp_path_factory):
    root = tmp_path_factory.mktemp("period_lag")
    timestamps = pd.bdate_range("2010-01-04", periods=FIRST_REPORT_BAR + 12 * BARS_PER_REPORT + 10)
    path = root / "market" / "periodic" / "periodic.zarr"
    path.parent.mkdir(parents=True)
    _panel(timestamps).to_zarr(path, mode="w")
    dataset = StockDataset(DatasetConfig(
        zarr_file_path=str(path), raw_data_dir_path=str(root / "downloads"), market="us_equity", frequency="1d",
    ))
    (root / "fields.yaml").write_text(yaml.safe_dump(FIELDS))
    (root / "operators.yaml").write_text(yaml.safe_dump(OPERATORS))
    vocabulary = Vocabulary.load([root / "fields.yaml"], root / "operators.yaml", columns=dataset_columns(path))
    label = Return(FactorConfig(
        warmup_bars=15, dataset=dataset, mode="batch", data_columns=("adjOpen",), kwargs={"n_forward_periods": 5},
        file_path=str(root / "labels" / "ret_5" / "ret_5.zarr"), njobs=1,
    ))
    label.build(str(timestamps[0].date()), str(timestamps[-1].date()))
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(CACHE_ENV, str(root / "kunquant_cache"))
        yield {"dataset": dataset, "vocabulary": vocabulary, "timestamps": timestamps, "label": label}


def dataset_columns(path) -> list[str]:
    return list(xr.open_zarr(path).data_vars)


def _graph(vocabulary, code):
    result = Gate(GateConfig(timeout_seconds=10.0), vocabulary).check(
        Candidate(hypothesis="Earnings growth.", mechanism_type="behavioural_bias", expected_sign=1,
                  horizon_rationale="A week.", reference="Test", description="Test.", code=code),
        Zoo(),
    )
    assert isinstance(result, Passed), getattr(result, "reasons", None)
    return result.graph


class Unranked(GraphFactor):
    """``GraphFactor`` without the rank, so the values read are the graph's own."""

    def _get_factor_func(self) -> Function:
        classes = self._classes()
        ops = []
        for name, recorded in self._recorded().items():
            output, new = Graph.from_json(recorded, classes).build({})
            ops += [*new, Output(output, name)]
        return Function(ops)


def _values(periodic, code: str, start=None) -> xr.DataArray:
    """The graph of ``code`` computed from ``start``, with no warm-up of its own."""
    vocabulary = periodic["vocabulary"]
    graph = _graph(vocabulary, code)
    graphs = {"lagged": graph.to_json()}
    lags = GraphFactor.period_lags(graphs, vocabulary.lag_columns)
    factor = Unranked(FactorConfig(
        warmup_bars=0, dataset=periodic["dataset"], mode="batch", data_columns=GraphFactor.data_columns(graphs, lags),
        njobs=1, kwargs={"graphs": graphs, "classes": vocabulary.class_paths([graphs["lagged"]]), "period_lags": lags},
    ))
    timestamps = periodic["timestamps"]
    return factor.compute(str((start or timestamps[0]).date()), str(timestamps[-1].date()))["lagged"]


def _bar(slot: int) -> int:
    """A bar in the middle of the ``slot``-th report a symbol shows."""
    return FIRST_REPORT_BAR + slot * BARS_PER_REPORT + BARS_PER_REPORT // 2


@pytest.mark.parametrize(
    ("symbol", "slot", "k", "expected"),
    [
        ("A", 0, 1, np.nan),  # no earlier report
        ("A", 1, 1, 100.0),
        ("A", 5, 1, 104.0),
        ("A", 3, 4, np.nan),
        ("A", 4, 4, 100.0),
        ("A", 11, 4, 107.0),
        ("B", 2, 1, 101.0),
        ("B", 3, 1, np.nan),  # report 4 follows report 2: two periods away, not one
        ("B", 4, 1, 104.0),
        ("B", 4, 4, np.nan),  # five periods before report 5 is report 0
        ("B", 6, 4, np.nan),  # report 3 was skipped
        ("B", 7, 4, 104.0),
        ("C", 6, 1, np.nan),  # the report is missing: nothing is shown
        ("C", 7, 1, np.nan),  # report 6 is missing, so report 5 is two periods away
        ("C", 7, 4, np.nan),  # the fourth report shown before report 7 is report 2
        ("C", 9, 4, np.nan),
        ("C", 8, 1, 107.0),
    ],
)
def test_period_lag_reads_the_kth_earlier_report_and_is_missing_when_it_is_missing_or_not_k_periods_away(
    periodic, symbol, slot, k, expected,
):
    values = _values(periodic, f"alpha = PeriodLag(netinc, {k})")
    value = float(values.sel(symbol=symbol).isel(timestamp=_bar(slot)))
    np.testing.assert_equal(value, expected)


def test_a_period_lag_reads_its_earlier_reports_from_before_the_start(periodic):
    start = periodic["timestamps"][_bar(9)]
    values = _values(periodic, "alpha = PeriodLag(netinc, 4)", start=start)
    assert float(values.sel(symbol="A", timestamp=start)) == 105.0


def test_the_evaluator_scores_a_period_lag_where_the_earlier_report_is_k_periods_away(periodic):
    # Over the fifth and sixth reports, A and C have the report four periods before; B skipped report 3.
    timestamps = periodic["timestamps"]
    first, last = FIRST_REPORT_BAR + 4 * BARS_PER_REPORT, FIRST_REPORT_BAR + 6 * BARS_PER_REPORT - 1
    evaluator = Evaluator.for_segment(
        Segment("training", str(timestamps[first].date()), str(timestamps[last].date())),
        prices=periodic["dataset"], label=periodic["label"], scoring=Scoring(None, None, njobs=1),
        warmup_bars=0, njobs=1, purge_bars=0, period_lags=periodic["vocabulary"].lag_columns,
    )
    graph = _graph(periodic["vocabulary"], "alpha = PeriodLag(netinc, 4)")
    [metrics] = evaluator.evaluate([graph])
    assert isinstance(metrics, Metrics), metrics
    assert metrics.coverage == pytest.approx(2 / 3)
