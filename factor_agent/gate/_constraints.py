"""Checks on a built graph: size, parameter values, dimensions and complexity.

Complexity is measured on the candidate's own nodes: the derived inputs
``vwap`` and ``amount`` count as one field each, and the bar mask the
NaN-preserving operators add is not the candidate's.
"""

import math
from functools import cache
from typing import TYPE_CHECKING

from quantlab.factor import kunquant_nan

from factor_agent.gate import _dimension
from factor_agent.gate._canonical import canonical_hashes, effective_parameters
from factor_agent.gate.graph import Graph
from factor_agent.gate.vocabulary import (
    FLAG_PARAMETERS, INPUT_COLUMNS, OPERATORS, PARAMETER_VALUES, bar_mask_inputs, inputs,
)

if TYPE_CHECKING:
    from factor_agent.config import GateConfig

#: Magnitudes of constants that carry no tuning: zero and one (NaN is not counted either).
_STRUCTURAL = (0.0, 1.0)


def check(graph: Graph, config: "GateConfig") -> tuple[list[str], dict]:
    """Return the reasons ``graph`` is refused (empty if none) and its statistics under ``config``'s limits."""
    reasons: list[str] = []
    fields, nodes, operators, constants = _candidate_part(graph)
    if nodes > config.max_nodes:
        reasons.append(f"the graph has {nodes} nodes, more than the limit of {config.max_nodes}")
    reasons += _parameter_reasons(graph) or _effective_reasons(graph)
    dimension, dimension_reasons = _dimension.infer(graph)
    reasons += dimension_reasons
    ratio = constants / max(operators, 1)
    if ratio > config.max_free_parameter_ratio:
        reasons.append(
            f"{constants} hand-set constants over {operators} operators is a free-parameter ratio of "
            f"{ratio:.2f}, above {config.max_free_parameter_ratio:g}; express the mechanism with fewer constants"
        )
    if len(fields) > config.max_base_fields:
        reasons.append(
            f"the factor reads {len(fields)} input fields ({', '.join(fields)}), more than {config.max_base_fields}"
        )
    stats = {
        "nodes": nodes,
        "operators": operators,
        "free_parameters": constants,
        "free_parameter_ratio": round(ratio, 4),
        "base_fields": fields,
        "dimension": _dimension.name(dimension),
    }
    return reasons, stats


def _parameter_reasons(graph: Graph) -> list[str]:
    reasons = []
    for node in graph.nodes:
        params = OPERATORS[node.op].params
        for key, value in node.attrs:
            kind = params.get(key)
            if kind is None:
                reasons.append(f"{node.op} has an attribute {key!r} the gate does not know")
            elif kind not in FLAG_PARAMETERS and not _allowed(kind, value):
                allowed = ", ".join(f"{v:g}" for v in sorted(set(PARAMETER_VALUES[kind])))
                reasons.append(f"{node.op}: {kind} {value!r} is not one of the allowed values ({allowed})")
    return reasons


def _effective_reasons(graph: Graph) -> list[str]:
    """Refuse values reached by chaining allowed ones, such as ``x * 0.25 * 1.5`` or a lag of 5 plus 10."""
    reasons = []
    for op, kind, value in effective_parameters(graph):
        if kind == "scale":
            ok = value == 0 or _allowed("constant", value) or _allowed("constant", 1 / value)
        else:
            ok = _allowed(kind, value) or (kind == "constant" and math.isnan(value))
        if not ok:
            reasons.append(
                f"the constants combine to {op} {kind} {value:g}, which is not one of the allowed values; "
                "use one allowed value instead of chaining several"
            )
    return list(dict.fromkeys(reasons))


def _allowed(kind: str, value) -> bool:
    if kind == "constant" and value == "nan":
        return True
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    allowed = PARAMETER_VALUES[kind]
    magnitude = abs(value) if kind == "constant" else value
    return any(math.isclose(magnitude, v, rel_tol=1e-9, abs_tol=0.0) or magnitude == v for v in allowed)


def _candidate_part(graph: Graph) -> tuple[list[str], int, int, int]:
    """Return the input fields the candidate reads and its node, operator and hand-set constant counts.

    A derived input counts as one node; the bar mask counts as none.
    """
    hashes = canonical_hashes(graph)
    derived, mask = reference_hashes()
    columns = {column: name for name, column in INPUT_COLUMNS.items()}
    fields, seen = set(), set()
    nodes = operators = constants = 0
    stack = [len(graph.nodes) - 1]
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        node = graph.nodes[i]
        if hashes[i] == mask:
            continue
        nodes += 1
        if hashes[i] in derived:
            fields.add(derived[hashes[i]])
            continue
        if node.op == "Input":
            column = dict(node.attrs)["name"]
            fields.add(columns.get(column, column))
            continue
        if node.op != "ConstantOp":
            operators += 1
        params = OPERATORS[node.op].params
        constants += sum(
            1 for key, value in node.attrs
            if params.get(key) == "constant" and value != "nan" and abs(float(value)) not in _STRUCTURAL
        )
        stack += list(node.inputs) + ([] if node.parent is None else [node.parent])
    return sorted(fields), nodes, operators, constants


@cache
def reference_hashes() -> tuple[dict[str, str], str]:
    """The canonical hashes of ``vwap`` and ``amount``, and of the bar mask of the NaN-preserving operators."""
    fields = inputs()
    derived = {canonical_hashes(Graph.record(fields[name]))[-1]: name for name in ("vwap", "amount")}
    with kunquant_nan.missing_bars_only(bar_mask_inputs(fields)):
        probe = kunquant_nan.SetInfOrNanToValue(fields["close"])
    mask = canonical_hashes(Graph.record(probe.inputs[1]))[-1]
    return derived, mask
