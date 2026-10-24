"""A candidate's KunQuant graph as plain data: what the Gate records and the Evaluator compiles.

The sandbox subprocess builds the graph and sends it back as a node list, so
the parent process never runs candidate code or unpickles anything; it
rebuilds the operators from registered classes only. A node records its
operator under the vocabulary's name for it; a graph keeps the classes of
the vocabulary it was read or recorded with (``classes``, not part of its
identity), which ``build`` makes the operators from.
"""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from KunQuant.Op import Input, OpBase

from factor_agent.gate._canonical import canonical_hashes

if TYPE_CHECKING:
    from factor_agent.gate.vocabulary import Vocabulary

#: Instance attributes an operator may carry; anything else cannot be recorded.
_OP_STATE = frozenset({"inputs", "attrs", "_parent_loop"})


@dataclass(frozen=True)
class Node:
    """One operator: its class name, attributes, input nodes and enclosing loop node."""

    op: str
    attrs: tuple[tuple[str, object], ...]
    inputs: tuple[int, ...]
    parent: int | None


@dataclass(frozen=True)
class Graph:
    """A factor graph in topological order; the last node is the factor."""

    nodes: tuple[Node, ...]
    #: The class of every operator name, from the vocabulary; None for a graph that is never built.
    classes: Mapping[str, type] | None = field(default=None, compare=False, repr=False)

    @classmethod
    def record(cls, output: OpBase, vocabulary: "Vocabulary", strict: bool = True) -> "Graph":
        """Record every operator ``output`` depends on, its inputs and loops first, under ``vocabulary``'s names.

        With ``strict=False`` (a reference factor's graph, compared but never
        rebuilt) an operator outside the vocabulary is recorded under its class
        name, other state is ignored and an attribute that is not a number,
        string or bool is recorded as its ``repr``.

        Raises
        ------
        ValueError
            If ``strict`` and an operator is not registered or carries
            state other than its inputs, attributes and loop.
        """
        index: dict[int, int] = {}
        nodes: list[Node] = []

        def visit(op: OpBase) -> int:
            if id(op) in index:
                return index[id(op)]
            name = vocabulary.recorded_name(type(op))
            attrs = tuple(op.attrs.items())
            if not strict:
                name = name or type(op).__name__
                attrs = tuple((k, v if isinstance(v, (bool, int, float, str)) else repr(v)) for k, v in attrs)
            if name is None:
                raise ValueError(f"operator {type(op).__name__} is not an operator of the vocabulary")
            extra = set(vars(op)) - _OP_STATE
            if extra and strict:
                raise ValueError(f"operator {name} carries state the gate cannot record: {sorted(extra)}")
            loop = op.get_parent()
            parent = visit(loop) if loop is not None else None
            inputs = tuple(visit(inp) for inp in op.inputs)
            index[id(op)] = len(nodes)
            nodes.append(Node(name, attrs, inputs, parent))
            return index[id(op)]

        visit(output)
        return cls(tuple(nodes), vocabulary.classes)

    @property
    def input_names(self) -> tuple[str, ...]:
        """The dataset columns the graph reads, sorted."""
        return tuple(sorted(dict(node.attrs)["name"] for node in self.nodes if node.op == "Input"))

    def build(self, inputs: dict[str, OpBase]) -> tuple[OpBase, list[OpBase]]:
        """Rebuild the operators and return ``(factor, every new operator)``.

        ``inputs`` maps a column name to its ``Input`` operator, so several
        graphs rebuilt into one function share their inputs; a missing name
        is added to it.

        Raises
        ------
        ValueError
            If the graph has no classes to build from.
        """
        if self.classes is None:
            raise ValueError("Graph.build: the graph was made without the classes of a vocabulary")
        made: list[OpBase] = []
        new: list[OpBase] = []
        for node in self.nodes:
            if node.op == "Input":
                name = dict(node.attrs)["name"]
                if name not in inputs:
                    inputs[name] = Input(name)
                    new.append(inputs[name])
                made.append(inputs[name])
                continue
            cls = self.classes[node.op]
            # quantlab's quantile-bound operators pick their per-bounds subclass in __new__.
            op = cls.__new__(cls) if cls.__new__ is object.__new__ else cls.__new__(cls, None, **dict(node.attrs))
            op.inputs = [made[i] for i in node.inputs]
            op.attrs = OrderedDict(node.attrs)
            op._parent_loop = made[node.parent] if node.parent is not None else None
            made.append(op)
            new.append(op)
        return made[-1], new

    def to_json(self) -> list:
        """Return the nodes as JSON-ready lists."""
        return [[n.op, [list(kv) for kv in n.attrs], list(n.inputs), n.parent] for n in self.nodes]

    @classmethod
    def from_json(cls, data: list, classes: Mapping[str, type]) -> "Graph":
        """Rebuild a graph from ``to_json``; every node must name one of ``classes`` (``Vocabulary.classes``).

        Raises
        ------
        ValueError
            If a node names a class that is not registered or refers to a
            node that does not precede it.
        """
        nodes = []
        for position, (op, attrs, inputs, parent) in enumerate(data):
            if op not in classes:
                raise ValueError(f"graph node {position} names {op!r}, which is not an operator of the vocabulary")
            refs = list(inputs) + ([] if parent is None else [parent])
            if any(not isinstance(r, int) or not 0 <= r < position for r in refs):
                raise ValueError(f"graph node {position} refers to a node that does not precede it")
            nodes.append(Node(op, tuple((str(k), v) for k, v in attrs), tuple(inputs), parent))
        return cls(tuple(nodes), dict(classes))

    def fingerprint(self) -> str:
        """Return the SHA-256 of the canonical graph.

        Two graphs that differ only by the operand order of a commutative
        operator, by constant subexpressions that fold to the same value, or
        by a repeated subexpression get the same fingerprint.
        """
        return canonical_hashes(self)[-1]
