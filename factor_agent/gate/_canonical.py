"""Canonical hashes of a graph's nodes: what the fingerprint and duplicate checks compare.

Each node's hash is a Merkle hash of its operator, attributes, inputs and
enclosing loop, computed after three rewrites:

- constant folding: an elementwise operator whose inputs are all constants
  becomes the constant it computes;
- constant arithmetic in one form: ``x + c``, ``Add(x, ConstantOp(c))`` and
  ``x - (-c)`` are all ``AddConst(x, c)``, ``x / c`` is ``MulConst(x, 1/c)``,
  nested constant offsets and scales merge, and ``x + 0`` and ``x * 1`` are ``x``;
- commutative ordering: the inputs of a commutative operator are sorted;
- division guards: ``Select(d > eps, n / d, nan)``, also on ``Abs(d)``, with
  ``d <= eps`` and the NaN first, and with the denominator itself guarded
  (``n / Select(d > eps, d, 1)``, as quantlab's ``FundamentalAlpha`` writes
  its ratios), is ``n / d`` for any positive ``eps`` up to ``GUARD_EPSILON``.

A repeated subexpression hashes the same wherever it appears, so the
fingerprint does not depend on how the code shared intermediate values.
Multiplying by zero is never folded: ``0 * x`` is NaN where ``x`` is, which
the NaN-preserving stand-ins rely on.
"""

import hashlib
import json
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from factor_agent.gate.graph import Graph

#: Operators whose inputs may be reordered.
COMMUTATIVE = frozenset({
    "Add", "Mul", "Max", "Min", "Equals", "And", "Or", "WindowedCorrelation", "WindowedCovariance", "EWCov",
    "SessionCorr",
})

#: Elementwise operators folded when every input is a constant.
_FOLD = {
    "Add": lambda a, b: a + b,
    "Sub": lambda a, b: a - b,
    "Mul": lambda a, b: a * b,
    "Div": lambda a, b: a / b,
    "Max": max,
    "Min": min,
    "Pow": lambda a, b: a ** b,
    "Abs": abs,
    "Sign": lambda a: float((a > 0) - (a < 0)),
    "Exp": math.exp,
    "Log": math.log,
    "Sqrt": math.sqrt,
}

#: Significant digits a constant keeps, so 1/3 * 3 and 1 hash alike.
_DIGITS = 12

#: The largest threshold a division guard may test the denominator against.
GUARD_EPSILON = 1e-6


class _Const:
    """A node that folded to a constant."""

    def __init__(self, value: float):
        self.value = value

    def key(self):
        return ["const", "nan" if math.isnan(self.value) else _rounded(self.value)]


class _Node:
    """A node's canonical form: operator, attributes, input terms, loop term."""

    def __init__(self, op: str, attrs: dict, inputs: list, parent):
        self.op = op
        self.attrs = attrs
        self.inputs = inputs
        self.parent = parent
        self.hash = _sha([op, sorted(attrs.items()), [_key(t) for t in inputs], None if parent is None else _key(parent)])

    def key(self):
        return self.hash


def _rounded(value: float) -> float:
    return float(f"{value:.{_DIGITS}g}")


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _key(term):
    return term.key()


def _same(one, other) -> bool:
    return (one is None and other is None) or (one is not None and other is not None and _key(one) == _key(other))


def _number(value):
    """Attribute values as JSON numbers; 5 and 5.0 are the same window."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return float(value)


def canonical_hashes(graph: "Graph") -> list[str]:
    """Return the canonical hash of every node of ``graph``, in node order; the last is the factor's."""
    return [_sha(["term", _key(t)]) for t in canonical_terms(graph)]


#: Operators whose result does not depend on the order of their inputs, to the last bit, NaN included.
#: ``Max`` and ``Min`` are not among them: a SIMD maximum returns its second input when one is NaN.
_EXACTLY_COMMUTATIVE = frozenset({"Add", "Mul", "Equals", "And", "Or", "SessionCorr"})


def computation_hashes(graph: "Graph") -> list[str]:
    """Return a hash of what every node of ``graph`` computes, in node order; the last is the factor's.

    Two nodes hash alike only when they compute the same values to the last
    bit: the hash is of the operator, attributes, inputs and loop as
    written, without the canonical hash's rewrites (``x + 1e8 - 1e8`` is not
    ``x`` in float32, nor ``x / 3`` ``x * (1/3)``), the inputs of an
    operator whose result does not depend on their order sorted. A repeated
    subexpression hashes the same wherever it appears.
    """
    hashes: list[str] = []
    for node in graph.nodes:
        inputs = [hashes[i] for i in node.inputs]
        if node.op in _EXACTLY_COMMUTATIVE:
            inputs.sort()
        parent = None if node.parent is None else hashes[node.parent]
        hashes.append(_sha(["computation", node.op, sorted(node.attrs), inputs, parent]))
    return hashes


def effective_parameters(graph: "Graph", lags_below: frozenset[str] = frozenset()) -> list[tuple[str, str, float]]:
    """Return ``(operator, kind, value)`` for every value the factor uses once constants are merged.

    Kinds are ``"constant"`` (a merged offset or a folded constant),
    ``"scale"`` (a merged multiplier; dividing by ``c`` is scaling by ``1/c``)
    and ``"lag"`` (a merged chain of ``BackRef``). Chaining allowed values
    (``x * 0.25 * 1.5``, ``BackRef(BackRef(x, 5), 10)``) shows up here as the
    value it really is. The lags below an operator of ``lags_below`` (a
    session aggregation, whose inputs count bars, not days) are left out.
    """
    found, seen, stack = [], set(), [(canonical_terms(graph)[-1], True)]
    while stack:
        term, lags = stack.pop()
        if isinstance(term, _Const):
            found.append(("ConstantOp", "constant", term.value))
            continue
        if (term.hash, lags) in seen:
            continue
        seen.add((term.hash, lags))
        if term.op in ("AddConst", "SubConstSwap", "DivConstSwap"):
            found.append((term.op, "constant", term.attrs["value"]))
        elif term.op == "MulConst":
            found.append((term.op, "scale", term.attrs["value"]))
        elif term.op == "BackRef" and lags:
            found.append((term.op, "lag", term.attrs["window"]))
        below = lags and term.op not in lags_below
        stack += [(t, below) for t in term.inputs + ([] if term.parent is None else [term.parent])]
    return found


def canonical_terms(graph: "Graph") -> list:
    """Return every node's canonical term, in node order; the last is the factor's."""
    terms = []
    for node in graph.nodes:
        attrs = dict(node.attrs)
        inputs = [terms[i] for i in node.inputs]
        parent = None if node.parent is None else terms[node.parent]
        terms.append(_canonical(node.op, attrs, inputs, parent))
    return terms


def _canonical(op: str, attrs: dict, inputs: list, parent):
    """Return one node's canonical term, given its inputs' and loop's terms."""
    if op == "ConstantOp":
        value = attrs["value"]
        return _Const(float("nan") if value == "nan" else float(value))
    swap = bool(attrs.pop("swap", False))
    attrs = {k: _number(v) for k, v in attrs.items()}

    if op in ("AddConst", "SubConst", "MulConst", "DivConst"):
        x, c = inputs[0], attrs["value"]
        return _with_constant(op, x, c, swap, parent)
    if op in _FOLD and inputs and all(isinstance(t, _Const) for t in inputs):
        try:
            return _Const(float(_FOLD[op](*(t.value for t in inputs))))
        except (ArithmeticError, ValueError):
            pass
    if op in ("Add", "Sub", "Mul", "Div") and len(inputs) == 2:
        left, right = inputs
        if isinstance(right, _Const) and not isinstance(left, _Const):
            return _with_constant(op + "Const", left, right.value, False, parent)
        if isinstance(left, _Const) and not isinstance(right, _Const):
            swapped = op in ("Sub", "Div")
            return _with_constant(op + "Const", right, left.value, swapped, parent)
    if op == "BackRef" and isinstance(inputs[0], _Node) and inputs[0].op == "BackRef" and _same(inputs[0].parent, parent):
        return _Node(op, {"window": inputs[0].attrs["window"] + attrs["window"]}, inputs[0].inputs, parent)
    if op == "Select":
        ratio = _unguarded(inputs, parent)
        if ratio is not None:
            return ratio
    if op in COMMUTATIVE:
        inputs = sorted(inputs, key=lambda t: json.dumps(_key(t)))
    return _Node(op, attrs, inputs, parent)


def _guard(condition) -> tuple[object, bool] | None:
    """``(d, holds)`` when ``condition`` tests ``d`` (or ``Abs(d)``) against a small positive threshold.

    ``holds`` is True for ``d > eps`` (the ratio is valid where it holds),
    False for ``d <= eps``; None for any other condition.
    """
    if not isinstance(condition, _Node):
        return None
    tested, threshold, holds = None, None, None
    if condition.op in ("GreaterThan", "LessEqual") and isinstance(condition.inputs[1], _Const):
        tested, threshold, holds = condition.inputs[0], condition.inputs[1].value, condition.op == "GreaterThan"
    elif condition.op in ("LessThan", "GreaterEqual") and isinstance(condition.inputs[0], _Const):
        tested, threshold, holds = condition.inputs[1], condition.inputs[0].value, condition.op == "LessThan"
    elif condition.op in ("GreaterThanConst", "LessEqualConst") and not condition.attrs.get("swap"):
        tested, threshold, holds = condition.inputs[0], condition.attrs["value"], condition.op == "GreaterThanConst"
    if tested is None or not 0 < threshold <= GUARD_EPSILON:
        return None
    if isinstance(tested, _Node) and tested.op == "Abs":
        tested = tested.inputs[0]
    return tested, holds


def _unguarded(inputs: list, parent):
    """``n / d`` for a guarded ratio ``Select(guard on d, n / d, nan)`` (either order); None for another Select."""
    if len(inputs) != 3:
        return None
    guard = _guard(inputs[0])
    if guard is None:
        return None
    d, holds = guard
    ratio, fill = (inputs[1], inputs[2]) if holds else (inputs[2], inputs[1])
    if not (isinstance(fill, _Const) and math.isnan(fill.value)) or not isinstance(ratio, _Node):
        return None
    if ratio.op == "Div":
        numerator, denominator = ratio.inputs
    elif ratio.op == "DivConstSwap":
        numerator, [denominator] = None, ratio.inputs
    else:
        return None
    inner = _guard(denominator.inputs[0]) if isinstance(denominator, _Node) and denominator.op == "Select" else None
    if inner is not None and inner[1] and _same(inner[0], d) and _same(denominator.inputs[1], d):
        denominator = d
    if not _same(denominator, d) or not _same(ratio.parent, parent):
        return None
    if numerator is None:
        return _Node("DivConstSwap", ratio.attrs, [d], parent)
    return _canonical("Div", {}, [numerator, d], parent)


def _with_constant(op: str, x, c: float, swap: bool, parent):
    """``x op c`` (``c op x`` when ``swap``) as ``AddConst``/``MulConst``, or a swapped form."""
    if isinstance(x, _Const):
        a, b = (c, x.value) if swap else (x.value, c)
        try:
            return _Const(float(_FOLD[op.removesuffix("Const")](a, b)))
        except (ArithmeticError, ValueError):
            pass
    if op == "SubConst" and not swap:
        op, c = "AddConst", -c
    elif op == "DivConst" and not swap and c != 0:
        op, c = "MulConst", 1.0 / c
    if op in ("SubConst", "DivConst"):  # c - x, c / x
        return _Node(op + "Swap", {"value": _number(c)}, [x], parent)
    if isinstance(x, _Node) and x.op == op and _same(x.parent, parent):
        inner = x.attrs["value"]
        c = inner + c if op == "AddConst" else inner * c
        x = x.inputs[0]
    if (op == "AddConst" and c == 0) or (op == "MulConst" and c == 1):
        return x
    return _Node(op, {"value": _rounded(c)}, [x], parent)


#: Root operators a cross-sectional rank IC cannot tell from their input.
_RANK_PRESERVING = frozenset({"Rank", "Scale", "CrossSectionalZScore"})


def ranked_root(term):
    """Return ``term`` without the rank-preserving operators at its root.

    A cross-sectional rank, scale or z-score, a constant offset and a
    nonzero scale keep or reverse the factor's cross-sectional order, so
    ``Rank(x) * 2 - 1`` and ``1 - x`` are ``x`` to the originality check
    (the sign is the factor's direction, read from its training IC, not a new factor).
    """
    while isinstance(term, _Node) and (
        term.op in _RANK_PRESERVING
        or term.op in ("AddConst", "SubConstSwap")
        or (term.op == "MulConst" and term.attrs["value"] != 0)
    ):
        term = term.inputs[0]
    return term


def term_key(term) -> str:
    """Return a canonical term's identity as a string."""
    return json.dumps(_key(term))


def shape_key(term) -> str:
    """Return the hash of a canonical term with every numeric attribute erased.

    Two terms with the same shape differ at most in their windows, lags and
    constants.
    """
    memo: dict[int, str] = {}

    def shape(t) -> str:
        if isinstance(t, _Const):
            return "const"
        if id(t) not in memo:
            attrs = sorted(
                (k, "#" if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in t.attrs.items()
            )
            inputs = [shape(i) for i in t.inputs]
            if t.op in COMMUTATIVE:
                inputs.sort()
            memo[id(t)] = _sha([t.op, attrs, inputs, None if t.parent is None else shape(t.parent)])
        return memo[id(t)]

    return shape(term)
