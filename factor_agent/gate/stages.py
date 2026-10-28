"""The stage split: a candidate's recorded graph as the graphs it runs as, and the stage rules it breaks.

A candidate of an intraday family is one graph in two stages. Every session
aggregation (``factor_agent.aggregations``) and what it depends on is its
intraday stage, computed on each session's intraday bars; everything above
the aggregations is its daily stage, computed on one value per stock and
day. ``split`` gives the daily graph, each aggregation subtree replaced by
an ``Input`` of its session column, named after the subtree's canonical hash
(``session_column``), and the intraday graphs by those names, each ending in
its aggregation: two aggregations that compute the same, in one graph or in
two, are one column. The daily family's bar mask, which quantlab's
NaN-preserving stand-ins add to what they compute, is left out of the
intraday stage: a bar there is an intraday bar. A graph without an
aggregation or an intraday field is its own daily graph.

The rules ``split`` checks, each broken one a reason:

- an intraday field reaches the daily stage only through an aggregation;
- the intraday stage reads no daily field;
- each operator is used only in the stages its operator-file entry names
  (``OpSpec.stages``).

The Gate, the Evaluator and the export all split a graph here; nothing else
knows how its stages are found.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from KunQuant.ops import Add

from factor_agent.aggregations import SessionAggregation
from factor_agent.gate._canonical import canonical_hashes
from factor_agent.gate.graph import Graph, Node

if TYPE_CHECKING:
    from factor_agent.gate.vocabulary import Vocabulary

#: The stage of the nodes above every aggregation, and of those below one.
DAILY, INTRADAY = "daily", "intraday"


@dataclass(frozen=True)
class Staged:
    """A graph split into its stages; see the module docstring.

    ``daily`` reads each session column as an ``Input`` of its name;
    ``intraday`` maps each name to its intraday graph; ``reasons`` are the
    stage rules the graph breaks, empty when it breaks none (and the graphs
    are then the ones it runs as).
    """

    daily: Graph
    intraday: Mapping[str, Graph] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()


def session_column(subtree_hash: str) -> str:
    """The name of the session column of an aggregation subtree, from its canonical hash."""
    return f"session_{subtree_hash}"


def is_aggregation(graph: Graph, node: Node) -> bool:
    """Whether ``node`` of ``graph`` is a session aggregation."""
    cls = (graph.classes or {}).get(node.op)
    return isinstance(cls, type) and issubclass(cls, SessionAggregation)


def split(graph: Graph, vocabulary: "Vocabulary") -> Staged:
    """Return ``graph``'s daily and intraday graphs and the stage rules it breaks; see the module docstring."""
    aggregations = [is_aggregation(graph, node) for node in graph.nodes]
    if not any(aggregations) and not any(_column(node) in vocabulary.intraday_columns for node in graph.nodes):
        return Staged(graph)
    masked = _masked(graph, vocabulary)
    stages = _stages(graph, aggregations, masked)
    reasons = _reasons(graph, vocabulary, stages, aggregations)
    if reasons:
        return Staged(graph, {}, reasons)
    intraday = {
        i: _rebuilt(graph, i, lambda j: masked.get(j))
        for i, aggregation in enumerate(aggregations) if aggregation and DAILY in stages[i]
    }
    names = {i: session_column(subgraph.fingerprint()) for i, subgraph in intraday.items()}
    daily = _rebuilt(graph, len(graph.nodes) - 1, lambda j: Node("Input", (("name", names[j]),), (), None) if j in names else None)
    return Staged(daily, {names[i]: subgraph for i, subgraph in intraday.items()})


def _column(node: Node) -> str | None:
    return dict(node.attrs)["name"] if node.op == "Input" else None


def _masked(graph: Graph, vocabulary: "Vocabulary") -> dict[int, int]:
    """The nodes that add the daily bar mask to a value, each with the value's node.

    Inside ``missing_bars_only`` the NaN-preserving stand-ins add the daily
    family's bar mask (``Vocabulary.reference_hashes``) to what they
    compute; in the intraday stage there is no daily bar, so the value is
    taken without it.
    """
    _, mask = vocabulary.reference_hashes
    if not mask:
        return {}
    hashes = canonical_hashes(graph)
    masked = {}
    for i, node in enumerate(graph.nodes):
        cls = (graph.classes or {}).get(node.op)
        if not (isinstance(cls, type) and issubclass(cls, Add)):
            continue
        values = [j for j in node.inputs if hashes[j] not in mask]
        if len(values) == 1 and len(node.inputs) == 2:
            masked[i] = values[0]
    return masked


def _stages(graph: Graph, aggregations: list[bool], masked: dict[int, int]) -> list[dict[str, str | None]]:
    """The stages of each node, each with the aggregation it is under (None in the daily stage).

    The root is in the daily stage; an aggregation's inputs, and everything
    they depend on but the daily bar mask (``_masked``), are in the
    intraday stage under it.
    """
    stages: list[dict[str, str | None]] = [{} for _ in graph.nodes]
    stack = [(len(graph.nodes) - 1, DAILY, None)]
    while stack:
        i, stage, under = stack.pop()
        if stage == INTRADAY and i in masked:
            stack.append((masked[i], stage, under))
            continue
        if stage in stages[i]:
            continue
        stages[i][stage] = under
        node = graph.nodes[i]
        below, by = (INTRADAY, node.op) if aggregations[i] else (stage, under)
        stack += [(j, below, by) for j in node.inputs]
        if node.parent is not None:
            stack.append((node.parent, stage, under))
    return stages


def _reasons(graph: Graph, vocabulary: "Vocabulary", stages: list[dict], aggregations: list[bool]) -> tuple[str, ...]:
    """The stage rules the nodes break, in node order, each once."""
    intraday_columns = vocabulary.intraday_columns
    reasons = []
    for i, node in enumerate(graph.nodes):
        column = _column(node)
        if column is not None:
            name = vocabulary.written(column)
            if DAILY in stages[i] and column in intraday_columns:
                reasons.append(f"the intraday field {name} reaches the daily stage without a session aggregation")
            if INTRADAY in stages[i] and column not in intraday_columns:
                reasons.append(f"the daily field {name} is read inside the intraday stage, under {stages[i][INTRADAY]}")
            continue
        if aggregations[i]:
            continue
        allowed = vocabulary.spec(node.op).stages
        for stage in stages[i]:
            if stage not in allowed:
                reasons.append(
                    f"{node.op} is used in the {stage} stage, but may be used in the {' and '.join(allowed)} stage only"
                )
    return tuple(dict.fromkeys(reasons))


def _rebuilt(graph: Graph, root: int, substitute: Callable[[int], int | Node | None]) -> Graph:
    """The graph of what ``root`` depends on, in node order, with ``substitute``'s replacements.

    ``substitute(i)`` gives the node to take in node ``i``'s place, a leaf
    node to put there (once, however often it is put) or None to keep it.
    """
    index: dict[int, int] = {}
    leaves: dict[Node, int] = {}
    nodes: list[Node] = []

    def visit(i: int) -> int:
        if i in index:
            return index[i]
        node, replacement = graph.nodes[i], substitute(i)
        if isinstance(replacement, int):
            index[i] = visit(replacement)
        elif isinstance(replacement, Node):
            if replacement not in leaves:
                nodes.append(replacement)
                leaves[replacement] = len(nodes) - 1
            index[i] = leaves[replacement]
        else:
            parent = visit(node.parent) if node.parent is not None else None
            inputs = tuple(visit(j) for j in node.inputs)
            nodes.append(Node(node.op, node.attrs, inputs, parent))
            index[i] = len(nodes) - 1
        return index[i]

    visit(root)
    return Graph(tuple(nodes), graph.classes)
