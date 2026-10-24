"""Dimension inference: what each node of a graph measures, from the registry's dimension rules.

A dimension is a pair of exponents ``(price, volume)``: price ``(1, 0)``,
volume ``(0, 1)``, money ``(1, 1)``, a pure number ``(0, 0)``, a variance of
price ``(2, 0)``. Besides those, a node may be a comparison (``BOOL``), a
constant that takes whatever dimension it is combined with (``ANY``), a window
loop (``LOOP``) or something the rules cannot say (``UNKNOWN``, e.g. the log
of a price). An invalid combination, such as price plus volume, is a reason
to reject; ``UNKNOWN`` is passed on.
"""

from fractions import Fraction

from factor_agent.gate.vocabulary import COLUMN_DIMENSIONS, OPERATORS

BOOL = "bool"
ANY = "any"
LOOP = "loop"
UNKNOWN = "unknown"
PURE = (Fraction(0), Fraction(0))

_NAMES = {PURE: "pure number", (1, 0): "price", (0, 1): "volume", (1, 1): "money"}


def name(dim) -> str:
    """Return a readable name: ``"price"``, ``"price^2"``, ``"price/volume"``, ``"unknown"``..."""
    if isinstance(dim, str):
        return {BOOL: "comparison", ANY: "constant"}.get(dim, dim)
    if dim in _NAMES:
        return _NAMES[dim]
    up = [f"{base}^{e}" if e != 1 else base for base, e in zip(("price", "volume"), dim) if e > 0]
    down = [f"{base}^{-e}" if e != -1 else base for base, e in zip(("price", "volume"), dim) if e < 0]
    return ("*".join(up) or "1") + ("/" + "*".join(down) if down else "")


class _Invalid(Exception):
    pass


def infer(graph) -> tuple[object, list[str]]:
    """Return the factor's dimension and the reasons the graph's dimensions are invalid (empty if none)."""
    dims: list = []
    reasons: list[str] = []
    for node in graph.nodes:
        spec = OPERATORS[node.op]
        attrs = dict(node.attrs)
        ins = [dims[i] for i in node.inputs]
        try:
            _check_kinds(node.op, spec.inputs, ins)
            dims.append(_RULES[spec.dimension](node.op, ins, attrs))
        except _Invalid as exc:
            reasons.append(str(exc))
            dims.append(UNKNOWN)
    output = dims[-1]
    if output == BOOL:
        reasons.append("alpha is a comparison (true or false), not a number; turn it into a number with Select")
    elif output == ANY:
        reasons.append("alpha is a constant; it must depend on the inputs")
    elif output == LOOP:
        reasons.append("alpha is a window loop; reduce it with a Reduce* operator")
    return output, list(dict.fromkeys(reasons))


def _check_kinds(op: str, kinds: tuple[str, ...], ins: list) -> None:
    """Every input must be what the operator expects: a number, a comparison or a loop."""
    for position, dim in enumerate(ins):
        kind = kinds[position] if position < len(kinds) else "series"
        if kind == "condition" and dim not in (BOOL, UNKNOWN):
            raise _Invalid(f"{op} needs a comparison as input {position + 1}, got a {name(dim)}")
        if kind == "loop" and dim != LOOP:
            raise _Invalid(f"{op} needs a ForeachBackWindow loop as input {position + 1}")
        if kind == "series" and dim == BOOL:
            raise _Invalid(f"{op} takes a number but gets a comparison; turn it into a number with Select")
        if kind == "series" and dim == LOOP:
            raise _Invalid(f"{op} takes a series but gets a loop; read the loop with IterValue")


def _known(*dims) -> bool:
    return UNKNOWN not in dims


def _unify(op: str, a, b):
    if a == ANY:
        return b
    if b == ANY or a == b:
        return a
    if not _known(a, b):
        return UNKNOWN
    raise _Invalid(f"{op} combines a {name(a)} with a {name(b)}")


def _numeric(d):
    """A constant in a product counts as a pure number."""
    return PURE if d == ANY else d


def _add(a, b, sign=1):
    a, b = _numeric(a), _numeric(b)
    if not _known(a, b):
        return UNKNOWN
    return (a[0] + sign * b[0], a[1] + sign * b[1])


def _times(d, factor: Fraction):
    if d in (UNKNOWN, ANY):
        return d
    return (d[0] * factor, d[1] * factor)


def _scale(op, ins, attrs):
    # Multiplying by zero gives 0 where x has data and NaN where it has none:
    # it combines with anything, as a constant does.
    if op == "MulConst" and attrs["value"] == 0:
        return ANY
    if op == "DivConst" and attrs.get("swap"):
        return _times(ins[0], Fraction(-1))
    return ins[0]


def _pure_only(op, ins, attrs):
    return PURE if all(_numeric(d) == PURE for d in ins) else UNKNOWN


def _power(op, ins, attrs):
    base, exponent = (_numeric(d) for d in ins)
    return PURE if base == PURE and exponent == PURE else UNKNOWN


def _select(op, ins, attrs):
    # A choice between two constants depends on its condition: a pure number.
    chosen = _unify(op, ins[1], ins[2])
    return PURE if chosen == ANY else chosen


def _compare(op, ins, attrs):
    _unify(op, ins[0], ins[1])
    return BOOL


def _input(op, ins, attrs):
    exponents = COLUMN_DIMENSIONS.get(attrs["name"])
    return UNKNOWN if exponents is None else tuple(Fraction(e) for e in exponents)


_RULES = {
    "input": _input,
    "constant": lambda op, ins, attrs: ANY,
    "same": lambda op, ins, attrs: ins[0],
    "first": lambda op, ins, attrs: ins[0],
    "second": lambda op, ins, attrs: ins[1],
    "pure": lambda op, ins, attrs: PURE,
    "match": lambda op, ins, attrs: _unify(op, ins[0], ins[1]),
    "product": lambda op, ins, attrs: ANY if ins[0] == ins[1] == ANY else _add(ins[0], ins[1]),
    "quotient": lambda op, ins, attrs: ANY if ins[0] == ins[1] == ANY else _add(ins[0], ins[1], -1),
    "scale": _scale,
    "square": lambda op, ins, attrs: _times(ins[0], Fraction(2)),
    "sqrt": lambda op, ins, attrs: _times(ins[0], Fraction(1, 2)),
    "transcendental": _pure_only,
    "repeated_product": _pure_only,
    "power": _power,
    "select": _select,
    "compare": _compare,
    "compare_constant": lambda op, ins, attrs: BOOL,
    "logic": lambda op, ins, attrs: BOOL,
    "loop": lambda op, ins, attrs: LOOP,
}
