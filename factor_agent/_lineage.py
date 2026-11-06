"""Each trial's Lineage: the trials that built on it, and how they did.

A trial's children are the trials whose direction names it as ``parent``
(direct children only). Its Lineage counts them by action (refinement or
mutation), those that stopped before evaluation (refused by the parser,
the Gate or the reviewer, or not compiled) and those accepted, and keeps
the best evaluated child with its residual t in its direction, beside the
trial's own. The prompts carry it as one line (``Lineage.text``); the
fertility is the same counts as one number, for choosing parents.

A child's outcome is 0 when it stopped before evaluation and otherwise the
amount its residual t in its direction exceeds the admission threshold it
was judged against (``_admission.t_threshold`` at the evaluated-trial
count after its round: admission judges every candidate of a round at that
count, whatever its place in the round), floored at 0. The fertility is the mean of the
three best outcomes, padded with zeros, so a parent is rewarded for
children that pass, not for many attempts.
"""

import math
from dataclasses import dataclass

from factor_agent._admission import direction, t_threshold
from factor_agent.config import AdmissionConfig
from factor_agent.ledger import TrialRecord

#: The best child outcomes the fertility is the mean of.
FERTILE_CHILDREN = 3


@dataclass(frozen=True)
class Lineage:
    """The trials built on one trial; see the module docstring."""

    attempts: int = 0
    refinements: int = 0
    mutations: int = 0
    #: Children that stopped before evaluation.
    unevaluated: int = 0
    #: The accepted children's ids, in the order they were tried.
    accepted: tuple[str, ...] = ()
    #: The evaluated child with the largest residual t in its direction, "" when none was evaluated.
    best_child: str = ""
    best_child_t: float = float("nan")
    #: The trial's own residual t in its direction, NaN when it was not evaluated.
    own_t: float = float("nan")
    fertility: float = 0.0

    def text(self) -> str:
        """The line the planner's accepted factors and the proposer's parent block carry."""
        if not self.attempts:
            return "lineage: no trial has built on it yet"
        line = (
            f"lineage: {_count(self.attempts, 'trial')} built on it ({_count(self.refinements, 'refinement')}, "
            f"{_count(self.mutations, 'mutation')}), {self.unevaluated} stopped before evaluation, "
            + (f"{len(self.accepted)} accepted ({', '.join(self.accepted)})" if self.accepted else "none accepted")
        )
        if not self.best_child:
            return line + "; no child was evaluated"
        return (
            f"{line}; best evaluated child {self.best_child}, residual t in its direction {_num(self.best_child_t)} "
            f"against this trial's {_num(self.own_t)}"
        )


def lineages(trials: list[tuple[str, TrialRecord]], admission: AdmissionConfig) -> dict[str, Lineage]:
    """The Lineage of every trial of ``trials`` (``_feedback.named``, in the ledger's order), by id."""
    evaluated_through: dict[int, int] = {}
    for _, r in trials:
        evaluated_through[r.round] = evaluated_through.get(r.round, 0) + r.evaluated
    count = 0
    for number in sorted(evaluated_through):
        count += evaluated_through[number]
        evaluated_through[number] = count
    children: dict[str, list[tuple[str, TrialRecord]]] = {}
    for trial, r in trials:
        parent = (r.direction or {}).get("parent")
        if parent:
            children.setdefault(parent, []).append((trial, r))
    result = {}
    for trial, r in trials:
        built = children.get(trial, [])
        actions = [(c.direction or {}).get("action") for _, c in built]
        scored = [(child, _directional_t(c), c) for child, c in built if c.evaluated]
        best = max(((t, child) for child, t, _ in scored if math.isfinite(t)), default=(float("nan"), ""))
        outcomes = sorted(
            (max(t - t_threshold(admission, evaluated_through[c.round]), 0.0) if math.isfinite(t) else 0.0
             for _, t, c in scored),
            reverse=True,
        )
        result[trial] = Lineage(
            attempts=len(built), refinements=actions.count("refinement"), mutations=actions.count("mutation"),
            unevaluated=len(built) - len(scored), accepted=tuple(child for child, c in built if c.status == "accepted"),
            best_child=best[1], best_child_t=best[0], own_t=_directional_t(r) if r.evaluated else float("nan"),
            fertility=sum(outcomes[:FERTILE_CHILDREN]) / FERTILE_CHILDREN,
        )
    return result


def _directional_t(r: TrialRecord) -> float:
    """An evaluated trial's residual (partial IC) t in its direction."""
    return direction(r.metrics["ic_mean"]) * r.metrics["partial_ic_t"]


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" + ("" if n == 1 else "s")


def _num(value: float) -> str:
    return f"{value:+.2f}" if math.isfinite(value) else "n/a"
