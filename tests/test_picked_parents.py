"""Picked parents: the evaluated trials ranked by quality, fertility and novelty shared over a graph of the trials.

The trials, their Lineage and the tried trials' correlation matrix are
written by hand; the graphs are recorded from code on the price-volume
vocabulary, so two trials share structure only where their code does.
"""

import math

import numpy as np

from factor_agent._feedback import named
from factor_agent._lineage import Lineage
from factor_agent._picked_parents import picked_parents
from factor_agent.gate.graph import Graph
from factor_agent.ledger import TrialRecord
from tests.vocabulary_files import price_volume

VOCABULARY = price_volume()

#: Codes with nothing in common but their inputs, so no structure edge joins them.
APART = [
    "alpha = WindowedAvg(close, 5)",
    "alpha = volume / WindowedStddev(volume, 30)",
    "alpha = Log(high / low)",
    "alpha = Sqrt(Abs(open))",
]


def _graph(code: str) -> list:
    return Graph.record(VOCABULARY.execute(code)["alpha"], VOCABULARY).to_json()


def _trial(number, t, code, status="rejected", parent=""):
    """An evaluated trial of round ``number`` with residual t ``t`` in its (positive) direction."""
    return TrialRecord(
        round=number, candidate={"description": code, "mechanism_type": "momentum"}, status=status, stage="admission",
        reasons=[], metrics={"ic_mean": 0.02, "partial_ic_t": t}, graph=_graph(code),
        direction={"mechanism_type": "momentum", "action": "refinement" if parent else "new", "parent": parent,
                   "direction": ""},
    )


def _rank(records, lineage=None, matrix=None, k=10, **options):
    trials = named(records)
    names = [trial for trial, _ in trials]
    if matrix is None:
        matrix = np.eye(len(names))
    return picked_parents(trials, names, np.asarray(matrix, dtype=float), lineage or {}, VOCABULARY.classes, k, **options)


def _order(picked):
    return [p.trial for p in picked]


def test_higher_quality_is_picked_first():
    records = [_trial(1, 1.0, APART[0]), _trial(1, 3.0, APART[1]), _trial(1, 2.0, APART[2])]
    assert _order(_rank(records)) == ["T1-02", "T1-03", "T1-01"]


def test_of_equal_quality_the_fertile_trial_is_picked_first():
    records = [_trial(1, 2.0, APART[0]), _trial(1, 2.0, APART[1])]
    lineage = {"T1-02": Lineage(attempts=2, refinements=2, fertility=0.8)}
    picked = _rank(records, lineage)
    assert _order(picked) == ["T1-02", "T1-01"]
    assert [p.front for p in picked] == [1, 2]


def test_of_equal_quality_the_trial_farther_from_the_accepted_factors_is_picked_first():
    accepted = _trial(1, 2.0, "alpha = WindowedAvg(close, 5) / WindowedAvg(volume, 5)", status="accepted")
    near = _trial(2, 2.0, "alpha = WindowedAvg(close, 5) / WindowedAvg(volume, 10)")   # T2-01: shares 3 of 7 subtrees
    far = _trial(2, 2.0, APART[2])                                                       # T2-02: shares none
    matrix = [[1.0, 0.8, 0.0], [0.8, 1.0, 0.0], [0.0, 0.0, 1.0]]
    picked = {p.trial: p for p in _rank([accepted, near, far], matrix=matrix, propagate=False)}
    # Structure distance 1 - 3/7, correlation distance 1 - 0.8; the far trial is at 1 on both.
    assert math.isclose(picked["T2-01"].novelty, (1 - 3 / 7) / 2 + 0.2 / 2)
    assert picked["T2-02"].novelty == 1.0
    order = _order(_rank([accepted, near, far], matrix=matrix))
    assert order.index("T2-02") < order.index("T2-01")


def test_a_trial_linked_to_a_strong_trial_is_picked_above_its_unlinked_twin_and_not_without_the_flow():
    strong = _trial(1, 6.0, APART[1])                     # T1-01
    twin = _trial(2, 1.0, APART[0])                       # T2-01
    linked = _trial(2, 1.0, APART[2], parent="T1-01")     # T2-02, a refinement of the strong trial
    assert _order(_rank([strong, twin, linked])) == ["T1-01", "T2-02", "T2-01"]
    assert _order(_rank([strong, twin, linked], propagate=False)) == ["T1-01", "T2-01", "T2-02"]


def test_a_trial_built_on_five_times_is_never_picked():
    records = [_trial(1, 9.0, APART[0]), _trial(1, 1.0, APART[1]), _trial(1, 0.5, APART[2])]
    lineage = {"T1-01": Lineage(attempts=5, refinements=5, fertility=2.0)}
    assert _order(_rank(records, lineage)) == ["T1-02", "T1-03"]


def test_the_same_inputs_give_the_same_order():
    records = [_trial(1, t, code) for t, code in zip([2.0, 2.0, 3.0, 1.0], APART)]
    lineage = {"T1-01": Lineage(attempts=1, fertility=0.5), "T1-04": Lineage(attempts=1, fertility=0.9)}
    matrix = [[1.0, 0.3, 0.2, 0.0], [0.3, 1.0, 0.0, 0.4], [0.2, 0.0, 1.0, 0.1], [0.0, 0.4, 0.1, 1.0]]
    first = _rank(records, lineage, matrix, k=3)
    assert len(first) == 3
    assert all(_rank(records, lineage, matrix, k=3) == first for _ in range(3))


def test_without_an_accepted_factor_the_novelty_is_zero_and_quality_and_fertility_rank():
    records = [_trial(1, 3.0, APART[0]), _trial(1, 1.0, APART[1]), _trial(1, 1.0, APART[2])]
    lineage = {"T1-02": Lineage(attempts=1, fertility=1.0)}
    picked = _rank(records, lineage)
    assert [p.novelty for p in picked] == [0.0, 0.0, 0.0]
    # Neither of the first two dominates the other; the third is dominated by both.
    assert [(p.trial, p.front) for p in picked] == [("T1-01", 1), ("T1-02", 1), ("T1-03", 2)]


def test_a_nan_correlation_makes_no_edge_and_no_error():
    strong, twin, other = _trial(1, 6.0, APART[1]), _trial(2, 1.0, APART[0]), _trial(2, 1.0, APART[2])
    matrix = [[1.0, math.nan, 0.0], [math.nan, 1.0, 0.0], [0.0, 0.0, 1.0]]
    picked = {p.trial: p for p in _rank([strong, twin, other], matrix=matrix)}
    assert picked["T2-01"].quality == picked["T2-02"].quality
    linked = {p.trial: p for p in _rank([strong, twin, other], matrix=[[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])}
    assert linked["T2-01"].quality > linked["T2-02"].quality
