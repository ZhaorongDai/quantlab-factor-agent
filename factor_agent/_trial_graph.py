"""The Trial graph: the evaluated trials of the rounds before one, over which that round's Picked parents are chosen.

Each evaluated trial, accepted or not, is a node with three own utilities:

- quality, its residual (partial IC) t in its direction;
- fertility, its Lineage's (``_lineage``), 0 when nothing built on it;
- novelty, half the mean structure distance plus half the mean correlation
  distance to the ``nearest_accepted`` accepted factors nearest to it by
  their sum (all of them when fewer, itself left out; 0 when there is
  none). The structure distance is 1 minus the Jaccard similarity of the
  two trials' sets of canonical subtree hashes (``canonical_hashes``), the
  correlation distance 1 minus the absolute correlation of the tried
  trials' matrix (``Evaluator.tried``), 1 where it is NaN or a trial is not
  in the matrix.

Three relations join the nodes: generation (weight 1 between a trial and
the trial its direction names as parent, both ways), structure (the Jaccard
similarity, where it is at least ``min_jaccard``) and correlation (the
absolute correlation, where it is at least ``min_correlation``; NaN makes
no edge); the last two keep each node's ``structure_neighbours`` and
``correlation_neighbours`` strongest edges and are made symmetric by union.
Each relation is normalised by row on its own, a node without an edge
keeping itself, and weighted by ``weights`` into P. The utilities are
propagated over it, F = (1 - gamma) R + gamma P F from F = R, until no
utility moves more than ``tolerance`` or after ``iterations``.

A node is ineligible when ``max_expansions`` trials have built on it (its
Lineage's attempts), or the planner's cap on children (``round.max_children``)
as many. The pick order sorts the nodes the expansion cap leaves into Pareto
fronts on the three utilities (all to be maximised), each front by crowding
distance (the largest first), ties by trial id (round, then slot), and then
passes over the nodes the planner's cap leaves out, so that cap changes no
other node's front. The names in code quotes are the settings
(``TrialGraphConfig``, the config's ``trial_graph``), by default
CosmosAlpha's (arXiv:2611.04555, Appendix A and C.3).
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from factor_agent._admission import directional_t
from factor_agent._feedback import trial_order
from factor_agent._lineage import Lineage
from factor_agent.config import RELATIONS, TrialGraphConfig
from factor_agent.gate._canonical import canonical_hashes
from factor_agent.gate.graph import Graph
from factor_agent.ledger import TrialRecord

#: Why a node is ineligible: the expansion cap, or the planner's cap on children.
EXPANSION_CAP, MAX_CHILDREN = "expansion_cap", "max_children"


class Utilities(NamedTuple):
    """A node's three utilities, all to be maximised."""

    quality: float
    fertility: float
    novelty: float


@dataclass(frozen=True)
class Node:
    """A trial in the graph: its own and propagated utilities, and whether it may be picked.

    ``neighbourhood`` is the weighted average of the utilities the last step
    of propagation took from its neighbours (within ``tolerance`` of their
    propagated utilities once propagation has converged), so that
    ``propagated`` is ``(1 - gamma) own + gamma neighbourhood``. ``reason``
    is ``EXPANSION_CAP`` or ``MAX_CHILDREN`` for an ineligible node, None
    for an eligible one.
    """

    trial: str
    own: Utilities
    propagated: Utilities
    neighbourhood: Utilities
    reason: str | None = None

    @property
    def eligible(self) -> bool:
        """Whether the node may be picked."""
        return self.reason is None


@dataclass(frozen=True)
class Edge:
    """A weight of P: ``relation``'s row-normalised weight from ``source`` to ``target``, times its ``weights`` share.

    A node without an edge of a relation has one to itself.
    """

    relation: str
    source: str
    target: str
    weight: float


@dataclass(frozen=True)
class Pick:
    """A node in pick order, with the utilities it was ranked on and its Pareto front (1 is the non-dominated)."""

    trial: str
    quality: float
    fertility: float
    novelty: float
    front: int


@dataclass(frozen=True)
class TrialGraph:
    """A round's Trial graph: ``gamma``, the nodes in the run's trial order and P's edges by relation."""

    gamma: float
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]

    def node(self, trial: str) -> Node:
        """The node of ``trial``."""
        return next(n for n in self.nodes if n.trial == trial)

    def pick_order(self, propagated: bool = True) -> list[Pick]:
        """The eligible nodes in pick order on their propagated utilities, or on their own; see the module docstring."""
        utilities = np.array([n.propagated if propagated else n.own for n in self.nodes]).reshape(-1, 3)
        ranked = [i for i, n in enumerate(self.nodes) if n.reason != EXPANSION_CAP]
        order = []
        for number, front in enumerate(_fronts(utilities, ranked), 1):
            crowding = _crowding(utilities, front)
            for i in sorted(front, key=lambda i: (-crowding[i], trial_order(self.nodes[i].trial))):
                if self.nodes[i].eligible:
                    order.append(Pick(self.nodes[i].trial, *(float(u) for u in utilities[i]), front=number))
        return order

    def neighbourhood(self, trial: str) -> Utilities:
        """The weighted average of the utilities ``trial`` took from its neighbours; see ``Node``."""
        return self.node(trial).neighbourhood

    def heaviest(self, trial: str, n: int) -> list[Edge]:
        """``trial``'s ``n`` heaviest edges to other trials, the heaviest first, ties in trial and relation order."""
        edges = [e for e in self.edges if e.source == trial and e.target != trial]
        return sorted(edges, key=lambda e: (-e.weight, trial_order(e.target), RELATIONS.index(e.relation)))[:n]

    def to_dict(self) -> dict:
        """The graph as plain JSON values; ``from_dict`` reads it back."""
        return {
            "gamma": self.gamma,
            "nodes": [
                {"trial": n.trial, "own": n.own._asdict(), "propagated": n.propagated._asdict(),
                 "neighbourhood": n.neighbourhood._asdict(), "eligible": n.eligible, "reason": n.reason}
                for n in self.nodes
            ],
            "edges": {
                relation: [[e.source, e.target, e.weight] for e in self.edges if e.relation == relation]
                for relation in RELATIONS
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrialGraph":
        """The graph ``to_dict`` wrote."""
        return cls(
            data["gamma"],
            tuple(
                Node(n["trial"], Utilities(**n["own"]), Utilities(**n["propagated"]),
                     Utilities(**n["neighbourhood"]), n["reason"])
                for n in data["nodes"]
            ),
            tuple(Edge(relation, *edge) for relation, edges in data["edges"].items() for edge in edges),
        )


def build(
    trials: list[tuple[str, TrialRecord]], names: list[str], matrix, lineage: Mapping[str, Lineage],
    classes: Mapping[str, type], settings: TrialGraphConfig, max_children: int | None = None,
) -> TrialGraph:
    """The Trial graph of the evaluated trials of ``trials``; see the module docstring.

    ``trials`` are the run's trials (``_feedback.named``), all of earlier
    rounds; ``names`` and ``matrix`` the tried trials and their
    correlations (``Evaluator.tried``); ``lineage`` every trial's Lineage by
    id (``_lineage.lineages``); ``classes`` the vocabulary's
    (``Vocabulary.classes``), which the recorded graphs are read with;
    ``settings`` the config's ``trial_graph``; ``max_children`` the
    planner's cap on children, None for none.
    """
    nodes = [(trial, r) for trial, r in trials if r.evaluated]
    ids = [trial for trial, _ in nodes]
    hashes = [frozenset(canonical_hashes(Graph.from_json(r.graph, classes))) if r.graph else frozenset() for _, r in nodes]
    jaccard = np.array([[_jaccard(a, b) for b in hashes] for a in hashes]).reshape(len(ids), len(ids))
    correlation = _correlations(ids, names, matrix)
    own = np.array([
        [_quality(r), lineage[trial].fertility if trial in lineage else 0.0,
         _novelty(i, [j for j, (_, e) in enumerate(nodes) if e.status == "accepted" and j != i], jaccard, correlation,
                  settings.nearest_accepted)]
        for i, (trial, r) in enumerate(nodes)
    ]).reshape(-1, 3)
    relations = _relations(nodes, jaccard, correlation, settings)
    propagated, neighbourhood = _propagate(own, sum(relations.values()), settings)
    return TrialGraph(
        settings.gamma,
        tuple(
            Node(trial, Utilities(*map(float, own[i])), Utilities(*map(float, propagated[i])),
                 Utilities(*map(float, neighbourhood[i])),
                 _reason(lineage.get(trial), settings.max_expansions, max_children))
            for i, trial in enumerate(ids)
        ),
        tuple(
            Edge(relation, ids[i], ids[j], float(weights[i, j]))
            for relation, weights in relations.items() for i, j in zip(*np.nonzero(weights))
        ),
    )


def _reason(lineage: Lineage | None, max_expansions: int, max_children: int | None) -> str | None:
    """Why a node with ``lineage`` may not be picked, None when it may."""
    attempts = lineage.attempts if lineage is not None else 0
    if attempts >= max_expansions:
        return EXPANSION_CAP
    if max_children is not None and attempts >= max_children:
        return MAX_CHILDREN
    return None


def _quality(r: TrialRecord) -> float:
    """The residual t in the trial's direction, 0 when it is not a number."""
    t = directional_t(r.metrics)
    return t if math.isfinite(t) else 0.0


def _jaccard(a: frozenset, b: frozenset) -> float:
    """The Jaccard similarity of two sets of subtree hashes, 0 when both are empty."""
    return len(a & b) / len(a | b) if a | b else 0.0


def _correlations(ids: list[str], names: list[str], matrix) -> np.ndarray:
    """The absolute correlation of every pair of ``ids`` from the tried matrix, NaN where a trial is not in it."""
    position = {name: i for i, name in enumerate(names)}
    at = [position.get(trial) for trial in ids]
    values = np.asarray(matrix, dtype=float)
    return np.array([
        [abs(values[a, b]) if a is not None and b is not None else math.nan for b in at] for a in at
    ]).reshape(len(at), len(at))


def _novelty(i: int, accepted: list[int], jaccard: np.ndarray, correlation: np.ndarray, nearest: int) -> float:
    """Half the mean structure and half the mean correlation distance to the ``nearest`` accepted factors nearest."""
    if not accepted:
        return 0.0
    distances = sorted(
        ((1 - jaccard[i, j]) / 2 + (1 - (correlation[i, j] if math.isfinite(correlation[i, j]) else 0.0)) / 2, j)
        for j in accepted
    )
    return float(np.mean([d for d, _ in distances[:nearest]]))


def _relations(
    nodes: list[tuple[str, TrialRecord]], jaccard: np.ndarray, correlation: np.ndarray, settings: TrialGraphConfig,
) -> dict[str, np.ndarray]:
    """P's parts: the generation, structure and correlation relations, each normalised by row and weighted."""
    n = len(nodes)
    index = {trial: i for i, (trial, _) in enumerate(nodes)}
    generation = np.zeros((n, n))
    for i, (_, r) in enumerate(nodes):
        parent = index.get((r.direction or {}).get("parent") or "")
        if parent is not None and parent != i:
            generation[i, parent] = generation[parent, i] = 1.0
    structure = _strongest(np.where(jaccard >= settings.min_jaccard, jaccard, 0.0), settings.structure_neighbours)
    known = np.nan_to_num(correlation, nan=0.0)  # NaN makes no edge
    correlated = _strongest(np.where(known >= settings.min_correlation, known, 0.0), settings.correlation_neighbours)
    parts = {"generation": generation, "structure": structure, "correlation": correlated}
    return {relation: settings.weights[relation] * _by_row(parts[relation]) for relation in RELATIONS}


def _strongest(weights: np.ndarray, strongest: int) -> np.ndarray:
    """Each node's ``strongest`` strongest edges (no edge to itself), made symmetric by union."""
    np.fill_diagonal(weights, 0.0)
    kept = np.zeros_like(weights)
    for i, row in enumerate(weights):
        for j in sorted(np.flatnonzero(row), key=lambda j: (-row[j], j))[:strongest]:
            kept[i, j] = kept[j, i] = row[j]
    return kept


def _by_row(weights: np.ndarray) -> np.ndarray:
    """``weights`` normalised by row; a row without an edge keeps its node."""
    normalised = weights.copy()
    for i, total in enumerate(weights.sum(axis=1)):
        if total > 0:
            normalised[i] /= total
        else:
            normalised[i, i] = 1.0
    return normalised


def _propagate(own: np.ndarray, relations: np.ndarray, settings: TrialGraphConfig) -> tuple[np.ndarray, np.ndarray]:
    """F = (1 - gamma) R + gamma P F from F = R, until it moves at most ``tolerance`` or after ``iterations``.

    Returns F and the P F of its last step.
    """
    propagated, neighbourhood = own, own
    for _ in range(settings.iterations):
        neighbourhood = relations @ propagated
        moved = (1 - settings.gamma) * own + settings.gamma * neighbourhood
        done = not moved.size or np.max(np.abs(moved - propagated)) <= settings.tolerance
        propagated = moved
        if done:
            break
    return propagated, neighbourhood


def _fronts(utilities: np.ndarray, nodes: list[int]) -> list[list[int]]:
    """``nodes`` sorted into Pareto fronts on ``utilities``, the non-dominated first."""
    def dominates(a: int, b: int) -> bool:
        """Whether node ``a`` is at least as good as ``b`` on every utility and better on one."""
        return bool(np.all(utilities[a] >= utilities[b]) and np.any(utilities[a] > utilities[b]))

    fronts, left = [], list(nodes)
    while left:
        front = [i for i in left if not any(dominates(j, i) for j in left)]
        fronts.append(front)
        left = [i for i in left if i not in front]
    return fronts


def _crowding(utilities: np.ndarray, front: list[int]) -> dict[int, float]:
    """Each node's crowding distance in ``front``: the normalised gaps around it, infinite at the ends of a utility that varies."""
    distance = dict.fromkeys(front, 0.0)
    for column in range(utilities.shape[1]):
        ordered = sorted(front, key=lambda i: utilities[i, column])
        span = utilities[ordered[-1], column] - utilities[ordered[0], column]
        if span <= 0:  # a utility all of the front shares tells none of it apart
            continue
        distance[ordered[0]] = distance[ordered[-1]] = math.inf
        for before, i, after in zip(ordered, ordered[1:], ordered[2:]):
            distance[i] += (utilities[after, column] - utilities[before, column]) / span
    return distance
