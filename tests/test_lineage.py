"""A trial's Lineage: how the trials that built on it did, as the prompts read it and as its fertility.

The trials are written by hand; the thresholds are the admission policy's
at the default anchor of 3.0, worked out from its formula: 3.8172 at 10
evaluated trials, 3.8406 at 11 and 3.3996 at 2.
"""

import math

from factor_agent._feedback import named
from factor_agent._lineage import lineages
from factor_agent.config import AdmissionConfig
from factor_agent.ledger import TrialRecord


def _trial(number, status="rejected", stage="admission", action="new", parent="", ic=None, t=None):
    """A trial of round ``number``; evaluated when ``ic`` and ``t`` (the residual t, not in its direction) are given."""
    metrics = None if ic is None else {"ic_mean": ic, "partial_ic_t": t}
    return TrialRecord(
        round=number, candidate={"description": f"trial of round {number}"}, status=status, stage=stage, reasons=[],
        metrics=metrics, direction={"mechanism_type": "momentum", "action": action, "parent": parent, "direction": ""},
    )


def _run() -> list[TrialRecord]:
    """One evaluated trial in round 1, nine in round 2 (ten in all), one more in round 3 (eleven)."""
    return [
        _trial(1, status="accepted", ic=0.02, t=3.5),                                        # T1-01
        _trial(2, status="failed", stage="gate", action="refinement", parent="T1-01"),       # T2-01
        _trial(2, action="refinement", parent="T1-01", ic=-0.03, t=-4.5),                    # T2-02
        _trial(2, status="accepted", action="mutation", parent="T1-01", ic=0.04, t=5.0),     # T2-03
        _trial(2, action="refinement", parent="T1-01", ic=0.01, t=2.0),                      # T2-04, worse than its parent
        _trial(2, stage="review", action="mutation", parent="T1-01"),                        # T2-05
        *[_trial(2, ic=0.01, t=0.5) for _ in range(6)],                                     # T2-06 to T2-11
        _trial(3, action="refinement", parent="T1-01", ic=0.02, t=4.0),                      # T3-01
        _trial(3, status="failed", stage="evaluate", action="refinement", parent="T2-03"),   # T3-02
    ]


def test_a_parent_counts_its_direct_children_by_action_and_outcome():
    lineage = lineages(named(_run()), AdmissionConfig())["T1-01"]
    assert (lineage.attempts, lineage.refinements, lineage.mutations, lineage.unevaluated) == (6, 4, 2, 2)
    assert lineage.accepted == ("T2-03",)
    assert lineage.best_child == "T2-03"
    assert lineage.best_child_t == 5.0 and lineage.own_t == 3.5


def test_the_line_names_the_counts_the_accepted_children_and_the_best_childs_residual_t_against_the_parents():
    assert lineages(named(_run()), AdmissionConfig())["T1-01"].text() == (
        "lineage: 6 trials built on it (4 refinements, 2 mutations), 2 stopped before evaluation, "
        "1 accepted (T2-03); best evaluated child T2-03, residual t in its direction +5.00 against this trial's +3.50"
    )


def test_fertility_is_the_mean_of_the_three_best_outcomes_above_each_childs_own_threshold():
    # T2-03: 5.0 - 3.8172, T2-02 (in its direction): 4.5 - 3.8172, T3-01: 4.0 - 3.8406; T2-04 and the
    # unevaluated children score 0.
    fertility = lineages(named(_run()), AdmissionConfig())["T1-01"].fertility
    assert math.isclose(fertility, (1.1828 + 0.6828 + 0.1594) / 3, abs_tol=1e-4)


def test_fewer_than_three_children_are_padded_with_zeros():
    trials = named([_trial(1, ic=0.02, t=3.5), _trial(2, action="mutation", parent="T1-01", ic=0.02, t=5.0)])
    # The child is the second evaluated trial: its threshold is 3.3996.
    assert math.isclose(lineages(trials, AdmissionConfig())["T1-01"].fertility, (5.0 - 3.3996) / 3, abs_tol=1e-4)


def test_a_child_only_of_a_grandparent_is_not_counted_and_no_evaluated_child_says_so():
    lineage = lineages(named(_run()), AdmissionConfig())["T2-03"]
    assert (lineage.attempts, lineage.unevaluated, lineage.fertility) == (1, 1, 0.0)
    assert lineage.text() == (
        "lineage: 1 trial built on it (1 refinement, 0 mutations), 1 stopped before evaluation, none accepted; "
        "no child was evaluated"
    )


def test_a_trial_nothing_built_on_has_no_fertility():
    lineage = lineages(named(_run()), AdmissionConfig())["T2-06"]
    assert (lineage.attempts, lineage.fertility) == (0, 0.0)
    assert lineage.text() == "lineage: no trial has built on it yet"
