"""Derived columns: a daily factor whose graph reads columns its providers fill, on the periodic dataset.

Period lags are one provider; a second one, filling a column of its own,
plugs in beside them, and the factor reads the longest warm-up any of them
needs. The dataset and its reports are those of ``test_period_lag``.
"""

import pytest
from KunQuant.Op import Input, Output
from KunQuant.Stage import Function
from quantlab.factor.config import FactorConfig

from factor_agent import derived, period_lag
from tests.test_period_lag import _bar, periodic  # noqa: F401 (the fixture)

pytestmark = pytest.mark.filterwarnings("ignore:.*warm-up bar:UserWarning")


class _Doubled(derived.Provider):
    """``doubled``, twice the close, read after ``warmup_bars`` bars of warm-up."""

    columns = ("doubled",)
    data_columns = ("adjClose",)

    def __init__(self, warmup_bars: int):
        self.warmup_bars = warmup_bars

    def fill(self, dataset, panel, arrays, symbols, timestamps):
        return {"doubled": 2.0 * arrays["adjClose"]}


class _LaggedAndDoubled(derived.DerivedColumnsFactor):
    """``PeriodLag(netinc, 4)`` and ``doubled`` as they are filled."""

    def _providers(self):
        return self.config.kwargs["providers"]

    def _get_factor_names(self):
        return ("lagged", "twice")

    def _get_factor_func(self):
        lagged, doubled = Input("netinc__q4"), Input("doubled")
        return Function([lagged, doubled, Output(lagged, "lagged"), Output(doubled, "twice")])


def _factor(periodic, doubled_warmup: int) -> _LaggedAndDoubled:
    lags = period_lag.read({"netinc__q4"}, periodic["vocabulary"].lag_columns)
    providers = [period_lag.PeriodLags(lags), _Doubled(doubled_warmup)]
    return _LaggedAndDoubled(FactorConfig(
        warmup_bars=10, dataset=periodic["dataset"], mode="batch",
        data_columns=derived.data_columns({"netinc__q4", "doubled"}, providers), njobs=1,
        kwargs={"providers": providers},
    ))


def test_a_second_provider_fills_its_column_beside_the_period_lags(periodic):
    factor = _factor(periodic, doubled_warmup=0)
    timestamps = periodic["timestamps"]
    values = factor.compute(str(timestamps[0].date()), str(timestamps[-1].date()))
    assert float(values["twice"].sel(symbol="A").isel(timestamp=_bar(3))) == 20.0
    assert float(values["lagged"].sel(symbol="A").isel(timestamp=_bar(11))) == 107.0


@pytest.mark.parametrize(("doubled_warmup", "expected"), [(0, 6 * 63), (500, 500)])
def test_the_warm_up_is_the_longest_any_provider_needs(periodic, doubled_warmup, expected):
    # A lag of 4 quarterly reports of 63 bars reads (4 + 2) * 63 bars.
    assert _factor(periodic, doubled_warmup).warmup_bars == expected


def test_the_dataset_columns_are_the_graph_inputs_read_as_they_are_and_the_providers_columns():
    lags = {"netinc__q4": {"column": "netinc", "k": 4, "key": "reportperiod", "days": 91,
                           "tolerance_days": 45, "bars": 63}}
    providers = [period_lag.PeriodLags(lags), _Doubled(0)]
    assert derived.data_columns({"netinc__q4", "doubled", "volume"}, providers) == (
        "adjClose", "netinc", "reportperiod", "volume",
    )
