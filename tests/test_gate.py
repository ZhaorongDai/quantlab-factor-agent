"""Gate.check: every bad candidate comes back Rejected with a readable reason.

The graph is built in a real subprocess; nothing is faked.
"""

import pytest

from factor_agent.candidate import Candidate
from factor_agent.gate import Gate, Passed, Rejected


def _candidate(code: str) -> Candidate:
    return Candidate(
        hypothesis="Intraday strength carries into the next week.",
        mechanism_type="behavioural_bias",
        expected_sign=1,
        horizon_rationale="Under-reaction resolves over a few days.",
        reference="Test reference",
        description="Close over open.",
        code=code,
    )


@pytest.fixture(scope="module")
def gate():
    return Gate(timeout_seconds=5.0)


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        ("alpha = close /", "syntax error"),
        ("import os\nalpha = close", "import"),
        ("from math import log\nalpha = close", "import"),
        ("alpha = SecretOp(close)", "SecretOp"),
        ("alpha = open(close)", "graph build failed"),
        ("alpha = close.__class__", "attribute access"),
        ('alpha = "{0.__class__}".format(close)', "attribute access"),
        ("f = lambda x: x\nalpha = f(close)", "Lambda"),
        ("class A:\n    pass\nalpha = close", "class"),
        ("alpha = (close / open, volume)", "exactly one output"),
        ("beta = close", "alpha"),
        ("alpha = 1.0", "operator"),
        ("alpha = close > open", "GreaterThan"),
        ("alpha = Div(close, 0)", "graph build failed"),
    ],
)
def test_a_bad_candidate_is_rejected_with_a_readable_reason(gate, code, reason):
    result = gate.check(_candidate(code))
    assert isinstance(result, Rejected)
    assert any(reason in r for r in result.reasons), result.reasons


def test_code_that_never_finishes_is_stopped_by_the_timeout():
    result = Gate(timeout_seconds=1.0).check(_candidate("while True:\n    pass\nalpha = close"))
    assert isinstance(result, Rejected)
    assert any("timed out" in r for r in result.reasons), result.reasons


def test_a_well_formed_candidate_passes_with_its_graph(gate):
    code = (
        "def strength(x, y):\n"
        "    return x / y\n"
        "ratio = strength(close, open)\n"
        "alpha = Rank(WindowedAvg(ratio, 5))\n"
    )
    result = gate.check(_candidate(code))
    assert isinstance(result, Passed), getattr(result, "reasons", None)
    assert result.graph.input_names == ("adjClose", "adjOpen")
    assert result.stats["nodes"] == 5
    assert len(result.fingerprint) == 64


def test_the_same_graph_gets_the_same_fingerprint(gate):
    one = gate.check(_candidate("alpha = close / open"))
    two = gate.check(_candidate("x = close\ny = open\nalpha = x / y"))
    other = gate.check(_candidate("alpha = open / close"))
    assert one.fingerprint == two.fingerprint
    assert one.fingerprint != other.fingerprint


def test_a_loop_body_is_kept_in_the_graph(gate):
    code = (
        "ratio = close / open\n"
        "loop = ForeachBackWindow(ratio, 5)\n"
        "with Builder(loop):\n"
        "    item = IterValue(loop, ratio)\n"
        "alpha = ReduceAdd(item)\n"
    )
    result = gate.check(_candidate(code))
    assert isinstance(result, Passed), getattr(result, "reasons", None)
    assert result.stats["nodes"] == 6
