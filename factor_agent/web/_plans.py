"""The rounds' plans and the run's Lineage: each trial's Lineage and Picked parents, and one round's plan read in full.

A round's plan is its latest Plan event: a round a resume dropped is planned
again under the same number after the resume, and until it is, the dropped
plan is not the round's. A run without events has no plans; a written
round then reads its slots from the ledger's directions, with no quota,
issues or Picked parents. A slot was given a Picked parent when the plan
picked the trial its direction builds on for a slot of its mechanism type:
the planner puts a type's picked slots first among its directions, so each
Picked parent fills the first slot of its type naming it.

Each trial's Lineage is ``factor_agent._lineage``'s, computed from the
ledger with the admission settings the run was last opened with (the
defaults for a run without events), its NaN read as ``None``.

A round that picked parents kept its Trial graph in the run directory
(``factor_agent._trial_graph``); the round's plan then carries it with each
node's rank before and after propagation, computed again by the graph's
pick order, so that the panel's ranks are the campaign's.
"""

import math
from dataclasses import dataclass
from pathlib import Path

from factor_agent import _trial_graph, events
from factor_agent._feedback import named
from factor_agent._lineage import lineages
from factor_agent.config import AdmissionConfig
from factor_agent.ledger import TrialRecord
from factor_agent.web._files import Replayed, read_json
from factor_agent.web._ledger import Trial, read_trials


@dataclass(frozen=True)
class TrialLineage:
    """A trial's Lineage (``factor_agent._lineage.Lineage``): the trials built on it, and how they did.

    ``accepted`` and ``best_child`` name trials by their ids (``T<round>-<slot
    from 1, two digits>``), as a direction's ``parent`` does; ``best_child``
    is ``None`` when no child was evaluated. ``best_child_t`` and ``own_t``
    are residual t in each trial's direction, ``None`` when not evaluated.
    """

    attempts: int
    refinements: int
    mutations: int
    unevaluated: int
    accepted: tuple[str, ...]
    best_child: str | None
    best_child_t: float | None
    own_t: float | None
    fertility: float


@dataclass(frozen=True)
class PlanSlot:
    """One slot of a round's plan.

    ``mechanism_type``, ``action``, ``parent`` (a trial id, ``""`` for a new
    direction) and ``direction`` are its research direction's. ``picked`` is
    true when the code picked its parent (a Picked parent), false when the
    planner chose it or it has none. ``outcome`` is its trial's status once
    the round is written, ``None`` while it is pending.
    """

    slot: int
    mechanism_type: str
    action: str
    parent: str
    direction: str
    picked: bool
    outcome: str | None


@dataclass(frozen=True)
class Neighbour:
    """One of a node's heaviest edges in the Trial graph: the trial it joins, its relation and its weight in P."""

    trial: str
    relation: str
    weight: float


@dataclass(frozen=True)
class GraphNode:
    """A trial in a round's Trial graph (``factor_agent._trial_graph.Node``), as the propagation view reads it.

    ``own``, ``propagated`` and ``neighbourhood`` hold its quality, fertility
    and novelty; ``propagated`` is ``(1 - gamma) own + gamma neighbourhood``.
    ``rank_own`` and ``rank_propagated`` are its places in the pick order on
    its own and on its propagated utilities, from 1, ``None`` for a node that
    may not be picked, ``reason`` saying why (``expansion_cap`` or
    ``max_children``). ``heaviest`` are its five heaviest edges to other
    trials, the heaviest first.
    """

    trial: str
    own: dict[str, float]
    propagated: dict[str, float]
    neighbourhood: dict[str, float]
    rank_own: int | None
    rank_propagated: int | None
    eligible: bool
    reason: str | None
    heaviest: tuple[Neighbour, ...]


@dataclass(frozen=True)
class RoundGraph:
    """A round's Trial graph: its ``gamma``, the round's Picked parents in pick order and its nodes in trial order."""

    gamma: float
    picked: tuple[str, ...]
    nodes: tuple[GraphNode, ...]


#: The heaviest edges each node of a round's graph lists.
HEAVIEST = 5


def _graph(run_dir: Path, number: int, plan: events.Plan) -> RoundGraph | None:
    """Round ``number``'s Trial graph from its file, ``None`` when the run directory holds none."""
    graph = _trial_graph.read(run_dir, number)
    if graph is None:
        return None
    ranks = {propagated: {p.trial: rank for rank, p in enumerate(graph.pick_order(propagated), 1)}
             for propagated in (False, True)}
    return RoundGraph(gamma=graph.gamma, picked=tuple(p["trial"] for p in plan.picked), nodes=tuple(
        GraphNode(trial=n.trial, own=n.own._asdict(), propagated=n.propagated._asdict(),
                  neighbourhood=n.neighbourhood._asdict(), rank_own=ranks[False].get(n.trial),
                  rank_propagated=ranks[True].get(n.trial), eligible=n.eligible, reason=n.reason,
                  heaviest=tuple(Neighbour(e.target, e.relation, e.weight) for e in graph.heaviest(n.trial, HEAVIEST)))
        for n in graph.nodes))


@dataclass(frozen=True)
class RoundPlan:
    """One round's plan: its slots in slot order, its quota by mechanism type and the plan's issues.

    ``quota`` is empty and ``issues`` too for a round whose plan the events
    file does not hold. ``graph`` is the round's Trial graph, ``None`` when
    the run directory holds none for it.
    """

    round: int
    slots: tuple[PlanSlot, ...]
    quota: dict[str, int]
    issues: tuple[str, ...]
    graph: RoundGraph | None = None


def plans(replayed: Replayed) -> dict[int, events.Plan]:
    """Each round's plan, by round: its latest Plan event, unless a resume dropped the round since."""
    found: dict[int, events.Plan] = {}
    for event in replayed.all:
        if isinstance(event, events.Resume) and event.dropped is not None:
            found = {number: plan for number, plan in found.items() if number < event.dropped}
        elif isinstance(event, events.Plan):
            found[event.round] = event
    return found


def picked_slots(plan: events.Plan) -> set[int]:
    """The slots of ``plan`` given a Picked parent: for each, the first slot of its type naming it."""
    left = [(p["trial"], p["mechanism_type"]) for p in plan.picked]
    slots = set()
    for slot, direction in enumerate(plan.directions):
        named_parent = (direction.get("parent"), direction.get("mechanism_type"))
        if named_parent in left:
            left.remove(named_parent)
            slots.add(slot)
    return slots


def trial_id(round: int, slot: int) -> str:
    """The id of the trial of ``round`` in ``slot`` (from 0), as ``_feedback.trial_id``."""
    return f"T{round}-{slot + 1:02d}"


def _nan(value):
    return float("nan") if value is None else value


def _lineages(trials: tuple[Trial, ...], admission: AdmissionConfig) -> dict[str, TrialLineage]:
    """Each trial's Lineage by its id; the ledger's NaN, read as ``None``, is NaN again while it is computed."""
    records = [TrialRecord(
        round=t.round, candidate=t.candidate, status=t.status, stage=t.stage, reasons=list(t.reasons),
        metrics={key: _nan(value) for key, value in t.metrics.items()} if t.metrics is not None else None,
        direction=t.direction,
    ) for t in trials]
    finite = lambda v: v if math.isfinite(v) else None  # noqa: E731
    return {trial: TrialLineage(
        attempts=found.attempts, refinements=found.refinements, mutations=found.mutations,
        unevaluated=found.unevaluated, accepted=found.accepted, best_child=found.best_child or None,
        best_child_t=finite(found.best_child_t), own_t=finite(found.own_t), fertility=found.fertility,
    ) for trial, found in lineages(named(records), admission).items()}


def admission(replayed: Replayed) -> AdmissionConfig:
    """The admission settings the run was last opened with; the defaults for a run without events."""
    opened = replayed.opened
    if opened is None:
        return AdmissionConfig()
    return AdmissionConfig(
        t_threshold=opened.t_threshold, max_correlation=opened.max_correlation,
        min_monotonicity=opened.min_monotonicity, min_coverage=opened.min_coverage,
        trial_exponent=opened.trial_exponent,
    )


@dataclass(frozen=True)
class Lineages:
    """What the run's Lineage gives each trial: its Lineage, the rounds it was picked in, whether its parent was."""

    lineage: dict[str, TrialLineage]
    picked_in: dict[str, tuple[int, ...]]
    parent_picked: set[str]


def lineage_of(trials: tuple[Trial, ...], replayed: Replayed, last: int) -> Lineages:
    """The Lineage of ``trials`` and their picks in the plans of rounds up to ``last``: the written ones, and the
    round in progress when there is one."""
    picked_in: dict[str, list[int]] = {}
    parent_picked = set()
    for number, plan in sorted(plans(replayed).items()):
        if number > last:
            continue
        for p in plan.picked:
            picked_in.setdefault(p["trial"], []).append(number)
        parent_picked |= {trial_id(number, slot) for slot in picked_slots(plan)}
    return Lineages(lineage=_lineages(trials, admission(replayed)),
                    picked_in={trial: tuple(rounds) for trial, rounds in picked_in.items()},
                    parent_picked=parent_picked)


def read_round(run_dir: str | Path, number: int) -> RoundPlan | None:
    """Round ``number`` of the run in ``run_dir`` as planned, ``None`` when it was neither written nor planned.

    A round beyond the written ones is read from its plan, its slots pending.
    """
    run_dir = Path(run_dir)
    state = read_json(run_dir / "state.json") or {}
    written = [t for t in read_trials(run_dir / "ledger.jsonl", state.get("rounds", 0)) if t.round == number]
    outcomes = {t.slot: t.status for t in written}
    plan = plans(Replayed.read(run_dir)).get(number)
    if plan is None:
        if not written:
            return None
        return RoundPlan(round=number, slots=tuple(
            PlanSlot(slot=t.slot, mechanism_type=(t.direction or t.candidate).get("mechanism_type") or "",
                     action=(t.direction or {}).get("action") or "new", parent=(t.direction or {}).get("parent") or "",
                     direction=(t.direction or {}).get("direction") or "", picked=False, outcome=t.status)
            for t in written), quota={}, issues=())
    picked = picked_slots(plan)
    return RoundPlan(round=number, slots=tuple(
        PlanSlot(slot=slot, mechanism_type=d.get("mechanism_type") or "", action=d.get("action") or "new",
                 parent=d.get("parent") or "", direction=d.get("direction") or "", picked=slot in picked,
                 outcome=outcomes.get(slot))
        for slot, d in enumerate(plan.directions)), quota=dict(plan.quota), issues=tuple(plan.issues),
        graph=_graph(run_dir, number, plan))
