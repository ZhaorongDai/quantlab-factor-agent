"""Checks on a built graph: size, parameter values, dimensions and complexity.

Complexity is measured on the candidate's own nodes: a derived field (such
as ``vwap``) counts as one node and one field, and the bar mask the
NaN-preserving operators add is not the candidate's. Parameter values and
dimensions are the vocabulary's checks.
"""

from typing import TYPE_CHECKING

from factor_agent.gate._canonical import canonical_hashes
from factor_agent.gate.graph import Graph

if TYPE_CHECKING:
    from factor_agent.config import GateConfig
    from factor_agent.gate.vocabulary import Vocabulary

#: Magnitudes of constants that carry no tuning: zero and one (NaN is not counted either).
_STRUCTURAL = (0.0, 1.0)


def check(graph: Graph, config: "GateConfig", vocabulary: "Vocabulary") -> tuple[list[str], dict]:
    """Return the reasons ``graph`` is refused (empty if none) and its statistics under ``config``'s limits."""
    reasons: list[str] = []
    fields, nodes, operators, constants = _candidate_part(graph, vocabulary)
    if nodes > config.max_nodes:
        reasons.append(f"the graph has {nodes} nodes, more than the limit of {config.max_nodes}")
    vocabulary_reasons, dimension = vocabulary.check(graph)
    reasons += vocabulary_reasons
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
        "dimension": dimension,
    }
    return reasons, stats


def _candidate_part(graph: Graph, vocabulary: "Vocabulary") -> tuple[list[str], int, int, int]:
    """Return the input fields the candidate reads and its node, operator and hand-set constant counts.

    A derived input counts as one node; the bar mask counts as none.
    """
    hashes = canonical_hashes(graph)
    derived, mask = vocabulary.reference_hashes
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
            fields.add(vocabulary.field_name(column))
            continue
        if node.op != "ConstantOp":
            operators += 1
        params = vocabulary.spec(node.op).params
        constants += sum(
            1 for key, value in node.attrs
            if params.get(key) == "constant" and value != "nan" and abs(float(value)) not in _STRUCTURAL
        )
        stack += list(node.inputs) + ([] if node.parent is None else [node.parent])
    return sorted(fields), nodes, operators, constants
