"""KunQuant's max drawdown over a computed value, made to compile when importing factor_agent.

KunQuant 0.1.12's ``WindowedMaxDrawdown`` reads the values of its loop from its
input rather than from a windowed buffer of it, so it compiles on a data column
and fails with "Bad buffer" on a value computed from the columns, as the
price-volume field amount is (Menooker/KunQuant#84). factor_agent buffers the
input first, as KunQuant's other windowed operators do.
"""

import numpy as np
from KunQuant.Driver import KunCompilerConfig, compileit
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function
from KunQuant.jit import cfake
from KunQuant.ops import Mul, WindowedMaxDrawdown
from KunQuant.runner import KunRunner

import factor_agent  # noqa: F401  (the import makes the max drawdown of a computed value compile)
from factor_agent.candidate import Candidate
from factor_agent.config import GateConfig
from factor_agent.exported import ranked_function
from factor_agent.gate import Gate
from factor_agent.zoo import Zoo
from tests.vocabulary_files import price_volume

WINDOW = 5


def test_a_candidate_of_the_max_drawdown_of_amount_compiles():
    vocabulary = price_volume()
    checked = Gate(GateConfig(timeout_seconds=60), vocabulary).check(
        Candidate("h", "other", "r", "d", "alpha = WindowedMaxDrawdown(amount, 10)"), Zoo.from_existing((), vocabulary))
    assert checked.graph is not None, checked.reasons
    compileit(ranked_function({"x": checked.graph}), "m", output_layout="TS")


def _max_drawdown(x: np.ndarray) -> np.ndarray:
    """The largest fall of ``x`` from the peak of the last ``WINDOW`` values to the lowest value since that peak, as a share of the peak."""
    out = np.full(len(x), np.nan)
    for t in range(WINDOW - 1, len(x)):
        window = x[t - WINDOW + 1:t + 1].astype(float)
        peak = int(np.argmax(window))
        out[t] = (window[peak] - window[peak:].min()) / window[peak]
    return out


def test_the_max_drawdown_of_a_computed_value_is_that_of_its_values(tmp_path):
    builder = Builder()
    with builder:
        close, volume = Input("close"), Input("volume")
        Output(WindowedMaxDrawdown(Mul(close, volume), WINDOW), "drawdown")
    library = cfake.compileit(
        [("drawdown", Function(builder.ops), KunCompilerConfig(input_layout="TS", output_layout="TS"))],
        "drawdown", cfake.CppCompilerConfig(), tempdir=str(tmp_path))
    rng = np.random.default_rng(0)
    close, volume = (rng.random((40, 8), dtype="float32") + 0.5 for _ in range(2))
    out = KunRunner.runGraph(KunRunner.createSingleThreadExecutor(), library.getModule("drawdown"),
                             {"close": close, "volume": volume}, 0, 40)["drawdown"]
    expected = np.stack([_max_drawdown(close[:, i] * volume[:, i]) for i in range(close.shape[1])], axis=1)
    np.testing.assert_allclose(out[WINDOW - 1:], expected[WINDOW - 1:], rtol=1e-6, atol=1e-7)
