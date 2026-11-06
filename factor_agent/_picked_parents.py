"""Picked parents: the evaluated trials the code chooses for a round's slots to build on.

Each evaluated trial, accepted or not, is a node with three raw utilities:

- quality, its residual (partial IC) t in its direction;
- fertility, its Lineage's (``_lineage``), 0 when nothing built on it;
- novelty, half the mean structure distance plus half the mean correlation
  distance to the ``NEAREST_ACCEPTED`` accepted factors nearest to it by
  their sum (all of them when fewer, itself left out; 0 when there is
  none). The structure distance is 1 minus the Jaccard similarity of the
  two trials' sets of canonical subtree hashes (``canonical_hashes``), the
  correlation distance 1 minus the absolute correlation of the tried
  trials' matrix (``Evaluator.tried``), 1 where it is NaN or a trial is not
  in the matrix.

Three relations join the nodes: lineage (weight 1 between a trial and the
trial its direction names as parent, both ways), structure (the Jaccard
similarity, where it is at least ``MIN_JACCARD``) and correlation (the
absolute correlation, where it is at least ``MIN_CORRELATION``; NaN makes
no edge); the last two keep each node's ``STRONGEST`` strongest edges and
are made symmetric by union. Each relation is normalised by row on its own,
a node without an edge keeping itself, and the three are averaged into P.
The utilities flow over it, F = (1 - GAMMA) R + GAMMA P F from F = R, until
no utility moves more than ``TOLERANCE`` or after ``ITERATIONS``.

The trials with fewer than ``MAX_ATTEMPTS`` attempts are then sorted into
Pareto fronts on the three flowed utilities (all to be maximised), each
front by crowding distance (the largest first), ties by trial id (round,
then slot). The constants are CosmosAlpha's (arXiv:2611.04555, Appendix A
and C.3), fixed here.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from factor_agent._admission import directional_t
from factor_agent._feedback import trial_order
from factor_agent._lineage import Lineage
from factor_agent.gate._canonical import canonical_hashes
from factor_agent.gate.graph import Graph
from factor_agent.ledger import TrialRecord

#: The accepted factors a trial's novelty is measured against.
NEAREST_ACCEPTED = 10
#: The smallest Jaccard similarity that makes a structure edge.
MIN_JACCARD = 0.20
#: The smallest absolute correlation that makes a correlation edge.
MIN_CORRELATION = 0.10
#: The structure and correlation edges each node keeps, the strongest.
STRONGEST = 30
#: The share of a utility that flows in from the related trials.
GAMMA = 0.75
TOLERANCE = 1e-8
ITERATIONS = 100
#: A trial built on this many times is not picked again.
MAX_ATTEMPTS = 5


@dataclass(frozen=True)
class PickedParent:
    """A picked parent with its utilities after the flow and its Pareto front (1 is the non-dominated)."""

    trial: str
    quality: float
    fertility: float
    novelty: float
    front: int


def picked_parents(
    trials: list[tuple[str, TrialRecord]], names: list[str], matrix, lineage: Mapping[str, Lineage],
    classes: Mapping[str, type], k: int, propagate: bool = True,
) -> list[PickedParent]:
    """At most ``k`` parents in pick order; see the module docstring.

    ``trials`` are the run's trials (``_feedback.named``), all of earlier
    rounds; ``names`` and ``matrix`` the tried trials and their
    correlations (``Evaluator.tried``); ``lineage`` every trial's Lineage by
    id (``_lineage.lineages``); ``classes`` the vocabulary's
    (``Vocabulary.classes``), which the recorded graphs are read with.
    Without ``propagate`` the raw utilities are ranked.
    """
    nodes = [(trial, r) for trial, r in trials if r.evaluated]
    if not nodes or k <= 0:
        return []
    ids = [trial for trial, _ in nodes]
    hashes = [frozenset(canonical_hashes(Graph.from_json(r.graph, classes))) if r.graph else frozenset() for _, r in nodes]
    jaccard = np.array([[_jaccard(a, b) for b in hashes] for a in hashes])
    correlation = _correlations(ids, names, matrix)
    raw = np.array([
        [_quality(r), lineage[trial].fertility if trial in lineage else 0.0,
         _novelty(i, [j for j, (_, e) in enumerate(nodes) if e.status == "accepted" and j != i], jaccard, correlation)]
        for i, (trial, r) in enumerate(nodes)
    ])
    flowed = _flow(raw, _relations(nodes, jaccard, correlation)) if propagate else raw
    eligible = [i for i, trial in enumerate(ids) if (lineage[trial].attempts if trial in lineage else 0) < MAX_ATTEMPTS]
    picked = []
    for number, front in enumerate(_fronts(flowed, eligible), 1):
        crowding = _crowding(flowed, front)
        for i in sorted(front, key=lambda i: (-crowding[i], trial_order(ids[i]))):
            picked.append(PickedParent(ids[i], *(float(u) for u in flowed[i]), front=number))
    return picked[:k]


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
    ])


def _novelty(i: int, accepted: list[int], jaccard: np.ndarray, correlation: np.ndarray) -> float:
    """Half the mean structure and half the mean correlation distance to the nearest accepted factors."""
    if not accepted:
        return 0.0
    distances = sorted(
        ((1 - jaccard[i, j]) / 2 + (1 - (correlation[i, j] if math.isfinite(correlation[i, j]) else 0.0)) / 2, j)
        for j in accepted
    )
    return float(np.mean([d for d, _ in distances[:NEAREST_ACCEPTED]]))


def _relations(nodes: list[tuple[str, TrialRecord]], jaccard: np.ndarray, correlation: np.ndarray) -> np.ndarray:
    """P: the mean of the lineage, structure and correlation relations, each normalised by row."""
    n = len(nodes)
    index = {trial: i for i, (trial, _) in enumerate(nodes)}
    lineage = np.zeros((n, n))
    for i, (_, r) in enumerate(nodes):
        parent = index.get((r.direction or {}).get("parent") or "")
        if parent is not None and parent != i:
            lineage[i, parent] = lineage[parent, i] = 1.0
    structure = _strongest(np.where(jaccard >= MIN_JACCARD, jaccard, 0.0))
    known = np.nan_to_num(correlation, nan=0.0)  # NaN makes no edge
    correlated = _strongest(np.where(known >= MIN_CORRELATION, known, 0.0))
    return sum(_by_row(w) for w in (lineage, structure, correlated)) / 3


def _strongest(weights: np.ndarray) -> np.ndarray:
    """Each node's ``STRONGEST`` strongest edges (no edge to itself), made symmetric by union."""
    np.fill_diagonal(weights, 0.0)
    kept = np.zeros_like(weights)
    for i, row in enumerate(weights):
        for j in sorted(np.flatnonzero(row), key=lambda j: (-row[j], j))[:STRONGEST]:
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


def _flow(raw: np.ndarray, relations: np.ndarray) -> np.ndarray:
    """F = (1 - GAMMA) R + GAMMA P F from F = R, until it moves at most ``TOLERANCE`` or after ``ITERATIONS``."""
    flowed = raw
    for _ in range(ITERATIONS):
        moved = (1 - GAMMA) * raw + GAMMA * relations @ flowed
        done = np.max(np.abs(moved - flowed)) <= TOLERANCE
        flowed = moved
        if done:
            break
    return flowed


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
