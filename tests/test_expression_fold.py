"""KunQuant's expression folding, made polynomial when importing factor_agent.

KunQuant 0.1.12 tells two equal-hashed operators apart by walking both as trees
(``OpBase.items``), which is exponential in a graph that reuses its values: an
EW beta of a cross-sectional ratio of a cross-sectional ratio never finished
compiling, and the round evaluating it stopped. factor_agent compares the two
as graphs instead, remembering each pair it has compared.
"""

import signal
import time

import pytest
from KunQuant.Driver import compileit
from KunQuant.Op import Builder, ForeachBackWindow, Input, IterValue, Output, WindowedTempOutput
from KunQuant.Stage import Function
from KunQuant.ops import Add, AddConst, ReduceAdd, WindowedAvg
from KunQuant.passes import ExprFold

import factor_agent  # noqa: F401  (the import makes the folding polynomial)
from factor_agent.candidate import Candidate
from factor_agent.config import GateConfig
from factor_agent.exported import ranked_function
from factor_agent.gate import Gate
from factor_agent.zoo import Zoo
from tests.vocabulary_files import price_volume

#: The candidate whose evaluation stopped a round of the daily learning test (arm A2, round 6).
REUSING = """
r = close / BackRef(close, 1) - 1
g = CrossSectionalWinsorize(r, 0.01, 0.99) + 1
u = g / g
rm = (g / Scale(g)) * Scale(u) - 1
b = EWBeta(r, rm, 250, 120)
a = EWAlpha(r, rm, 250, 120)
e = Abs(r - a - b * rm)
ew = CrossSectionalWinsorize(e, 0.01, 0.99) + 0.01
ue = ew / ew
civ = (ew / Scale(ew)) * Scale(ue)
dciv = Log(civ / BackRef(civ, 1))
hl = Log(high) - Log(low)
h1 = CrossSectionalWinsorize(hl, 0.025, 0.975) + 0.01
uh = h1 / h1
mr = (h1 / Scale(h1)) * Scale(uh)
dm = Log(mr / BackRef(mr, 1))
cb = EWBeta(dciv, dm, 250, 120)
ca = EWAlpha(dciv, dm, 250, 120)
dperp = dciv - ca - cb * dm
own = hl + 0.01
dov = Log(own / BackRef(own, 1))
alpha = EWBeta(dov, dperp, 250, 120)
"""

LIMIT_SECONDS = 20


@pytest.fixture
def limit():
    def stop(*_):
        raise TimeoutError(f"the compile took more than {LIMIT_SECONDS} s")

    previous = signal.signal(signal.SIGALRM, stop)
    signal.alarm(LIMIT_SECONDS)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous)


def test_a_graph_reusing_its_values_compiles_in_seconds(limit):
    vocabulary = price_volume()
    checked = Gate(GateConfig(timeout_seconds=60), vocabulary).check(
        Candidate("h", "other", "r", "d", REUSING), Zoo.from_existing((), vocabulary))
    started = time.monotonic()
    compileit(ranked_function({"x": checked.graph}), "m", output_layout="TS")
    assert time.monotonic() - started < LIMIT_SECONDS


def _folded(build) -> list[str]:
    builder = Builder()
    with builder:
        build()
    function = Function(builder.ops)
    ExprFold.expr_fold(function)
    return [type(op).__name__ for op in function.ops]


def test_equal_operators_over_distinct_equal_inputs_are_still_folded():
    def build():
        close, volume = Input("close"), Input("volume")
        left = WindowedAvg(Add(close, volume), 5)
        right = WindowedAvg(Add(close, volume), 5)  # the same as left, built twice
        Output(Add(left, right), "x")

    assert _folded(build).count("WindowedAvg") == 1


def _two_loops(body) -> list[str]:
    """Fold two equal loops over ``close``, each reducing what ``body`` makes of its value."""
    def build():
        close = Input("close")
        for name in ("x", "y"):
            window = WindowedTempOutput(close, 5)
            with ForeachBackWindow(window, 5) as loop:
                value = body(IterValue(loop, window))
            Output(ReduceAdd(value), name)

    return _folded(build)


def test_equal_loop_bodies_reusing_their_values_are_folded(limit):
    def body(value):
        for _ in range(40):
            value = Add(value, value)
        return value

    assert _two_loops(body).count("ReduceAdd") == 1


def test_a_deep_chain_in_equal_loop_bodies_is_folded():
    def body(value):
        for _ in range(3000):
            value = AddConst(value, 1.0)
        return value

    assert _two_loops(body).count("ReduceAdd") == 1


def test_operators_over_different_inputs_are_not_folded():
    def build():
        close, volume = Input("close"), Input("volume")
        Output(Add(WindowedAvg(close, 5), WindowedAvg(volume, 5)), "x")

    assert _folded(build).count("WindowedAvg") == 2
