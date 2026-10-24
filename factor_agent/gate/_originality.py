"""Originality against the zoo: rewrites, parameter-only refinements and large common subtrees.

All three compare canonical graphs (``_canonical``):

- a rewrite is a candidate whose canonical graph, without rank-keeping or rank-reversing
  root operators, is a zoo factor's;
- a parameter-only refinement has a zoo factor's shape: the same graph up
  to its windows, lags and constants;
- a common subtree is a node of the candidate whose canonical hash is a
  node of a zoo factor; its size counts the distinct nodes under it as the
  graph-size limit does (a derived input is one node, the bar mask none),
  and the largest one may not exceed the limit (AlphaAgent's largest common
  subtree, MIT; the limit of 5 nodes is QuantaAlpha's).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from factor_agent.gate._canonical import canonical_hashes, canonical_terms, ranked_root, shape_key, term_key
from factor_agent.gate._constraints import reference_hashes
from factor_agent.gate.graph import Graph
from factor_agent.gate.vocabulary import INPUT_COLUMNS

if TYPE_CHECKING:
    from factor_agent.zoo import Zoo, ZooEntry

#: Characters of a common subtree quoted in a reason.
_RENDER_LIMIT = 400

#: Factors named in one reason before the rest are counted.
_NAMED = 3


@dataclass(frozen=True)
class _Index:
    """What a graph is compared by: its node hashes, its ranked root and that root's shape."""

    hashes: frozenset[str]
    root: str
    shape: str


#: Indexes of zoo graphs by ``id``, with the graph kept so the id stays its own.
_INDEXES: dict[int, tuple[Graph, _Index]] = {}


def _index(graph: Graph) -> _Index:
    """Return the graph's index, cached for zoo graphs (a graph's attributes may be unhashable lists)."""
    cached = _INDEXES.get(id(graph))
    if cached is None or cached[0] is not graph:
        root = ranked_root(canonical_terms(graph)[-1])
        cached = _INDEXES[id(graph)] = (graph, _Index(frozenset(canonical_hashes(graph)), term_key(root), shape_key(root)))
    return cached[1]


def reasons(graph: Graph, zoo: "Zoo", max_common_nodes: int) -> list[str]:
    """Return why ``graph`` is not original against ``zoo`` (empty if it is)."""
    root = ranked_root(canonical_terms(graph)[-1])
    mine = _Index(frozenset(), term_key(root), shape_key(root))
    entries = [(entry, _index(entry.graph)) for entry in zoo.entries]
    rewrites = [entry for entry, index in entries if index.root == mine.root]
    if rewrites:
        terms = canonical_terms(graph)
        root = max((i for i, term in enumerate(terms) if term_key(term) == mine.root), default=len(terms) - 1)
        return [
            f"the factor rewrites {_names(rewrites)}: {_render(graph, root)} is that factor's whole graph, up to "
            "a transform that keeps or reverses its ranking; propose a different mechanism"
        ]
    found = []
    tuned = [entry for entry, index in entries if index.shape == mine.shape]
    if tuned:
        found.append(
            f"the factor is {_names(tuned)} with other windows or parameter values only; a refinement must "
            "change the expression of the mechanism, not its parameters"
        )
    size, node, sharing = _largest_common_subtree(graph, entries)
    if size > max_common_nodes:
        found.append(
            f"the factor shares a {size}-node subtree with {_names(sharing)}, more than the limit of "
            f"{max_common_nodes}: {_render(graph, node)}; express the mechanism without reusing that expression"
        )
    return found


def _largest_common_subtree(graph: Graph, entries: list) -> tuple[int, int | None, list["ZooEntry"]]:
    """Return the size and node of the candidate's largest subtree found in a zoo factor, and those factors."""
    hashes = canonical_hashes(graph)
    sizes = _subtree_sizes(graph, hashes)
    best, node, sharing = 0, None, []
    for i in sorted(range(len(hashes)), key=lambda i: -sizes[i]):
        if sizes[i] <= best and node is not None:
            break
        holders = [entry for entry, index in entries if hashes[i] in index.hashes]
        if holders:
            best, node, sharing = sizes[i], i, holders
    return best, node, sharing


def _subtree_sizes(graph: Graph, hashes: list[str]) -> list[int]:
    """Return the number of distinct nodes under each node, itself included."""
    derived, mask = reference_hashes()
    reach: list[frozenset[str]] = []
    for i, node in enumerate(graph.nodes):
        if hashes[i] == mask:
            reach.append(frozenset())
        elif hashes[i] in derived:
            reach.append(frozenset({hashes[i]}))
        else:
            under = [reach[j] for j in node.inputs] + ([] if node.parent is None else [reach[node.parent]])
            reach.append(frozenset({hashes[i]}).union(*under))
    return [len(r) for r in reach]


def _names(entries: list["ZooEntry"]) -> str:
    named = ", ".join(entry.label for entry in entries[:_NAMED])
    rest = len(entries) - _NAMED
    return named + (f" and {rest} more" if rest > 0 else "")


def _render(graph: Graph, node: int) -> str:
    """Return the expression under ``node`` as candidate code would write it."""
    hashes = canonical_hashes(graph)
    derived, mask = reference_hashes()
    columns = {column: name for name, column in INPUT_COLUMNS.items()}

    def render(i: int) -> str:
        n = graph.nodes[i]
        if hashes[i] in derived:
            return derived[hashes[i]]
        attrs = dict(n.attrs)
        if n.op == "Input":
            return columns.get(attrs["name"], attrs["name"])
        if n.op == "ConstantOp":
            return _display(attrs["value"])
        inputs = [j for j in n.inputs if hashes[j] != mask]
        if n.op == "Add" and len(inputs) == 1:  # a NaN-preserving stand-in: the operator plus the bar mask
            return render(inputs[0])
        values = [_display(v) for k, v in n.attrs if not (k == "swap" and not v)]
        return f"{n.op}({', '.join([render(j) for j in inputs] + values)})"

    text = render(node)
    return text if len(text) <= _RENDER_LIMIT else text[:_RENDER_LIMIT] + "..."


def _display(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value:g}" if isinstance(value, float) else str(value)
