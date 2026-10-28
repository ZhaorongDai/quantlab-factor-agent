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

Every window of an intraday graph stays within the session: each
time-series operator outside a loop body, and each reduction of a loop, is
wrapped in ``WithinSession`` by the bars it reaches back, its inputs'
windows included, so it is missing on a session's first bars until it
fits and never reads the session before. A session column is named after
the intraday graph so rewritten, what runs. Cross-sectional operators rank
across all the symbols of each bar.

The rules ``split`` checks, each broken one a reason with a hint, how to
write it instead, for the proposer:

- an intraday field reaches the daily stage only through an aggregation;
- the intraday stage reads no daily field;
- an aggregation's inputs are intraday: no aggregation of a daily value
  (inputs reading no intraday field), none nested in another;
- each operator is used only in the stages its operator-file entry names
  (``OpSpec.stages``);
- a window or a lag (``window_kind``) is one of its stage's set: in the
  intraday stage the intraday family's windows and lags
  (``Vocabulary.intraday_windows``, ``intraday_lags``), lags chained
  included; in the daily stage the operator file's set of the attribute.

The Gate, the Evaluator and the export all split a graph here; nothing else
knows how its stages are found.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING

from KunQuant.Op import ForeachBackWindow, WindowedTrait
from KunQuant.ops import Add

from factor_agent.aggregations import SESSION_POSITION, SessionAggregation, WithinSession
from factor_agent.gate._canonical import canonical_hashes, effective_parameters
from factor_agent.gate.graph import Graph, Node

#: The parameter set of an attribute that is bookkeeping (``vocabulary.FLAG``), never checked.
FLAG = "flag"

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
    are then the ones it runs as); ``hints`` say how to write the
    candidate instead, for the proposer.
    """

    daily: Graph
    intraday: Mapping[str, Graph] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    hints: tuple[str, ...] = ()


def session_column(subtree_hash: str) -> str:
    """The name of the session column of an aggregation subtree, from its canonical hash."""
    return f"session_{subtree_hash}"


def is_aggregation(graph: Graph, node: Node) -> bool:
    """Whether ``node`` of ``graph`` is a session aggregation."""
    cls = (graph.classes or {}).get(node.op)
    return isinstance(cls, type) and issubclass(cls, SessionAggregation)


def is_staged(graph: Graph, vocabulary: "Vocabulary") -> bool:
    """Whether ``graph`` runs in two stages: it holds a session aggregation or reads an intraday field."""
    intraday_columns = vocabulary.intraday_columns
    return any(is_aggregation(graph, node) or _column(node) in intraday_columns for node in graph.nodes)


def window_kind(vocabulary: "Vocabulary", op: str, attribute: str) -> str | None:
    """``"lag"`` for an attribute of ``op`` in the lag set, ``"window"`` for another named ``window``, else None.

    A staged graph's windows and lags are checked by the stage rules,
    against the set of the stage they are used in.
    """
    kind = vocabulary.spec(op).params.get(attribute)
    if kind is None or kind == FLAG:
        return None
    if kind == vocabulary.lag_set:
        return "lag"
    return "window" if attribute == "window" else None


def split(graph: Graph, vocabulary: "Vocabulary") -> Staged:
    """Return ``graph``'s daily and intraday graphs and the stage rules it breaks; see the module docstring."""
    if not is_staged(graph, vocabulary):
        return Staged(graph)
    aggregations = [is_aggregation(graph, node) for node in graph.nodes]
    masked = _masked(graph, vocabulary)
    stages = _stages(graph, aggregations, masked)
    reasons, hints = _reasons(graph, vocabulary, stages, aggregations, masked)
    if reasons:
        return Staged(graph, {}, reasons, hints)
    intraday = {
        i: _within_session(_rebuilt(graph, i, lambda j: masked.get(j)))
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


def _stages(graph: Graph, aggregations: list[bool], masked: dict[int, int]) -> list[dict[str, int | None]]:
    """The stages of each node, each with the aggregation node it is under (None in the daily stage).

    The root is in the daily stage; an aggregation's inputs, and everything
    they depend on but the daily bar mask (``_masked``), are in the
    intraday stage under it.
    """
    stages: list[dict[str, int | None]] = [{} for _ in graph.nodes]
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
        below, by = (INTRADAY, i) if aggregations[i] else (stage, under)
        stack += [(j, below, by) for j in node.inputs]
        if node.parent is not None:
            stack.append((node.parent, stage, under))
    return stages


def _reasons(
    graph: Graph, vocabulary: "Vocabulary", stages: list[dict], aggregations: list[bool], masked: dict[int, int],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The stage rules the nodes break, in node order, each once, and how to write each instead."""
    intraday_columns = vocabulary.intraday_columns
    found: dict[str, str] = {}
    daily_values = {
        i for i, aggregation in enumerate(aggregations)
        if aggregation and not _reads_intraday(graph, i, aggregations, masked, intraday_columns)
    }
    for i, node in enumerate(graph.nodes):
        column = _column(node)
        if column is not None:
            name = vocabulary.written(column)
            if DAILY in stages[i] and column in intraday_columns:
                found[f"the intraday field {name} reaches the daily stage without a session aggregation"] = (
                    f"reduce each session's {name} to one value per day with a session aggregation first, "
                    f"such as SessionMean({name}) or SessionLast({name}), and use that value in the daily stage"
                )
            under = stages[i].get(INTRADAY)
            if INTRADAY in stages[i] and column not in intraday_columns and under not in daily_values:
                found[f"the daily field {name} is read inside the intraday stage, under {graph.nodes[under].op}"] = (
                    f"combine the daily field {name} with a session's value outside its aggregation, in the daily "
                    f"stage: SessionMean(x) / {name}, not SessionMean(x / {name})"
                )
            continue
        if aggregations[i]:
            if i in daily_values:
                found[f"{node.op} aggregates a daily value: an aggregation reduces the intraday bars of each session"] = (
                    f"aggregate a value of the intraday fields, such as {node.op}(bar_close); a daily field or "
                    "a daily value is used as it is in the daily stage"
                )
            if INTRADAY in stages[i]:
                outer = graph.nodes[stages[i][INTRADAY]].op
                found[
                    f"{node.op} is nested inside {outer}: an aggregation's input is a value of the intraday bars"
                ] = (
                    f"aggregate each session once; to combine the values of several sessions, apply a daily "
                    f"window to the aggregation, such as WindowedAvg({node.op}(x), 20)"
                )
            continue
        allowed = vocabulary.spec(node.op).stages
        for stage in stages[i]:
            if stage not in allowed:
                found[
                    f"{node.op} is used in the {stage} stage, but may be used in the {' and '.join(allowed)} stage only"
                ] = _OUTSIDE_ITS_STAGES[stage].format(op=node.op)
        found.update(dict.fromkeys(_window_reasons(vocabulary, node, stages[i]), _window_hint(vocabulary)))
    for i, aggregation in enumerate(aggregations):
        if aggregation:
            found.update(dict.fromkeys(_chained_lags(vocabulary, _rebuilt(graph, i, masked.get)), _window_hint(vocabulary)))
    return tuple(found), tuple(dict.fromkeys(found.values()))


def _values(values) -> str:
    return ", ".join(f"{v:g}" for v in sorted(set(values)))


def _window_reasons(vocabulary: "Vocabulary", node: Node, stages: dict) -> list[str]:
    """Why each window and lag of ``node`` is outside the set of a stage it is used in."""
    reasons = []
    for key, value in node.attrs:
        kind = window_kind(vocabulary, node.op, key)
        if kind is None:
            continue
        if INTRADAY in stages:
            allowed = vocabulary.intraday_lags if kind == "lag" else vocabulary.intraday_windows
            if not any(value == v for v in allowed):
                reasons.append(f"{node.op}'s {kind} {value:g} is not one of the intraday {kind}s ({_values(allowed)} bars)")
        if DAILY in stages:
            daily = vocabulary.spec(node.op).params[key]
            if not vocabulary.allowed(daily, value):
                reasons.append(
                    f"{node.op}'s {kind} {value:g} is not one of the daily {kind}s ({_values(vocabulary.parameters[daily])})"
                )
    return reasons


def _chained_lags(vocabulary: "Vocabulary", subgraph: Graph) -> list[str]:
    """Why lags chained in an aggregation's intraday stage add up to a lag outside the intraday set."""
    allowed = vocabulary.intraday_lags
    # A lag of one BackRef is its own window, refused by _window_reasons when it is outside the set.
    written = {value for node in subgraph.nodes for key, value in node.attrs if window_kind(vocabulary, node.op, key) == "lag"}
    return [
        f"the lags chain to {op} lag {value:g} in the intraday stage, which is not one of the intraday lags "
        f"({_values(allowed)} bars)"
        for op, kind, value in effective_parameters(subgraph)
        if kind == "lag" and not any(value == v for v in allowed) and value not in written
    ]


def _window_hint(vocabulary: "Vocabulary") -> str:
    lags = vocabulary.intraday_lags
    return (
        f"in the intraday stage a window is one of {_values(vocabulary.intraday_windows)} bars and a lag one of "
        f"{_values(lags)} bars; in the daily stage a window or a lag is one of the values of its parameter set, "
        "in days; use one value of the stage's set rather than chaining several"
    )


#: How to write an operator used outside its stages instead, by the stage it was used in.
_OUTSIDE_ITS_STAGES = {
    INTRADAY: "use {op} on a session's aggregated value in the daily stage, such as {op}(SessionMean(x), ...), "
              "or within the session an operator the intraday stage allows",
    DAILY: "use {op} on the intraday bars, inside a session aggregation, such as SessionMean({op}(x, ...))",
}


def _reads_intraday(
    graph: Graph, aggregation: int, aggregations: list[bool], masked: dict[int, int], intraday_columns: frozenset,
) -> bool:
    """Whether the inputs of ``aggregation`` read an intraday field or hold another aggregation."""
    seen, stack = set(), list(graph.nodes[aggregation].inputs)
    while stack:
        i = stack.pop()
        i = masked.get(i, i)
        if i in seen:
            continue
        seen.add(i)
        node = graph.nodes[i]
        if aggregations[i] or _column(node) in intraday_columns:
            return True
        stack += list(node.inputs) + ([] if node.parent is None else [node.parent])
    return False


def _within_session(graph: Graph) -> Graph:
    """``graph``, an intraday graph, with each window wrapped in ``WithinSession`` by the bars it reaches back.

    A node reaches back as far as its own window (``BackRef``'s lag, ``w -
    1`` for a window of ``w`` bars, as KunQuant's ``required_input_window``
    says) and the farthest of its inputs and its loop. A window outside a
    loop body (a time-series operator, or a reduction of a loop's items) is
    wrapped, so it and every node above it are missing until all it reads
    lies within the session; a graph without a window is returned as it is.
    """
    classes = graph.classes or {}
    reach: list[int] = []
    for node in graph.nodes:
        cls = classes.get(node.op)
        own = 0
        if isinstance(cls, type) and issubclass(cls, WindowedTrait):
            own = cls.required_input_window(SimpleNamespace(attrs=dict(node.attrs))) - 1
        under = [reach[j] for j in node.inputs] + ([] if node.parent is None else [reach[node.parent]])
        reach.append(own + max(under, default=0))

    def windowed(i: int) -> bool:
        node, cls = graph.nodes[i], classes.get(graph.nodes[i].op)
        if node.parent is not None or reach[i] == 0 or (isinstance(cls, type) and issubclass(cls, ForeachBackWindow)):
            return False
        own = reach[i] > max([reach[j] for j in node.inputs], default=0)
        return own or any(graph.nodes[j].parent is not None for j in node.inputs)

    wrapped = {i for i in range(len(graph.nodes)) if windowed(i)}
    if not wrapped:
        return graph
    # The position's Input first; a graph that reads it too holds it twice, which builds one Input of it.
    out: list[Node] = [Node("Input", (("name", SESSION_POSITION),), (), None)]
    at: dict[int, int] = {}
    for i, node in enumerate(graph.nodes):
        parent = None if node.parent is None else at[node.parent]
        out.append(Node(node.op, node.attrs, tuple(at[j] for j in node.inputs), parent))
        at[i] = len(out) - 1
        if i in wrapped:
            out.append(Node("WithinSession", (("bars", reach[i]),), (at[i], 0), None))
            at[i] = len(out) - 1
    return Graph(tuple(out), {**classes, "WithinSession": WithinSession})


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
