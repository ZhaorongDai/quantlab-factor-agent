"""The Trial graph: the evaluated trials, their quality, fertility and novelty propagated over their relations, and the pick order.

The trials, their Lineage and the tried trials' correlation matrix are
written by hand; the graphs are recorded from code on the price-volume
vocabulary, so two trials share structure only where their code does.
"""

import json
import math

import numpy as np

from factor_agent._feedback import named
from factor_agent._lineage import Lineage
from factor_agent._trial_graph import TrialGraph, build, read, write
from factor_agent.config import TrialGraphConfig
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


def _graph_of(records, lineage=None, matrix=None, max_children=None, settings=TrialGraphConfig()):
    trials = named(records)
    names = [trial for trial, _ in trials]
    if matrix is None:
        matrix = np.eye(len(names))
    return build(trials, names, np.asarray(matrix, dtype=float), lineage or {}, VOCABULARY.classes, settings, max_children)


def _rank(records, lineage=None, matrix=None, k=10, propagated=True, settings=TrialGraphConfig()):
    return _graph_of(records, lineage, matrix, settings=settings).pick_order(propagated)[:k]


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
    picked = {p.trial: p for p in _rank([accepted, near, far], matrix=matrix, propagated=False)}
    # Structure distance 1 - 3/7, correlation distance 1 - 0.8; the far trial is at 1 on both.
    assert math.isclose(picked["T2-01"].utilities.novelty, (1 - 3 / 7) / 2 + 0.2 / 2)
    assert picked["T2-02"].utilities.novelty == 1.0
    order = _order(_rank([accepted, near, far], matrix=matrix))
    assert order.index("T2-02") < order.index("T2-01")


def test_a_trial_linked_to_a_strong_trial_is_picked_above_its_unlinked_twin_and_not_without_the_flow():
    strong = _trial(1, 6.0, APART[1])                     # T1-01
    twin = _trial(2, 1.0, APART[0])                       # T2-01
    linked = _trial(2, 1.0, APART[2], parent="T1-01")     # T2-02, a refinement of the strong trial
    assert _order(_rank([strong, twin, linked])) == ["T1-01", "T2-02", "T2-01"]
    assert _order(_rank([strong, twin, linked], propagated=False)) == ["T1-01", "T2-01", "T2-02"]


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
    assert [p.utilities.novelty for p in picked] == [0.0, 0.0, 0.0]
    # Neither of the first two dominates the other; the third is dominated by both.
    assert [(p.trial, p.front) for p in picked] == [("T1-01", 1), ("T1-02", 1), ("T1-03", 2)]


def test_a_nan_correlation_makes_no_edge_and_no_error():
    strong, twin, other = _trial(1, 6.0, APART[1]), _trial(2, 1.0, APART[0]), _trial(2, 1.0, APART[2])
    matrix = [[1.0, math.nan, 0.0], [math.nan, 1.0, 0.0], [0.0, 0.0, 1.0]]
    picked = {p.trial: p for p in _rank([strong, twin, other], matrix=matrix)}
    assert picked["T2-01"].utilities.quality == picked["T2-02"].utilities.quality
    linked = {p.trial: p for p in _rank([strong, twin, other], matrix=[[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])}
    assert linked["T2-01"].utilities.quality > linked["T2-02"].utilities.quality


def test_an_ineligible_node_says_why_and_is_never_picked_and_the_planners_cap_changes_no_other_front():
    records = [_trial(1, 3.0, APART[0]), _trial(1, 2.0, APART[1]), _trial(1, 1.0, APART[2])]
    graph = _graph_of(records, {"T1-01": Lineage(attempts=2)}, max_children=2)
    assert [(n.trial, n.eligible, n.reason) for n in graph.nodes] == [
        ("T1-01", False, "max_children"), ("T1-02", True, None), ("T1-03", True, None),
    ]
    # The capped trial still holds the first front: the others keep theirs.
    assert [(p.trial, p.front) for p in graph.pick_order()] == [("T1-02", 2), ("T1-03", 3)]
    expanded = _graph_of(records, {"T1-01": Lineage(attempts=5)}, max_children=6)
    assert expanded.node("T1-01").reason == "expansion_cap"
    assert [(p.trial, p.front) for p in expanded.pick_order()] == [("T1-02", 1), ("T1-03", 2)]


def test_a_propagated_utility_lies_gamma_of_the_way_from_its_own_to_its_neighbourhood():
    strong, twin, linked = _trial(1, 6.0, APART[1]), _trial(2, 1.0, APART[0]), _trial(2, 1.0, APART[2], parent="T1-01")
    matrix = [[1.0, 0.0, 0.2], [0.0, 1.0, 0.5], [0.2, 0.5, 1.0]]
    graph = _graph_of([strong, twin, linked], {"T1-01": Lineage(attempts=1, fertility=0.4)}, matrix)
    assert graph.gamma == 0.75
    for node in graph.nodes:
        assert node.propagated == tuple((1 - 0.75) * o + 0.75 * h for o, h in zip(node.own, graph.neighbourhood(node.trial)))
        # The neighbourhood is the edges' weighted average of the neighbours' propagated utilities.
        spread = [sum(e.weight * graph.node(e.target).propagated[u] for e in graph.edges if e.source == node.trial)
                  for u in range(3)]
        assert np.allclose(graph.neighbourhood(node.trial), spread, atol=1e-7)
    # The strong trial's child is lifted above its own quality of 1; its unlinked twin only by correlation.
    assert graph.node("T2-02").propagated.quality > graph.node("T2-01").propagated.quality > 1.0


def test_the_heaviest_neighbours_carry_their_total_weight_and_relations_and_each_relation_keeps_a_lone_node_to_itself():
    strong, twin, linked = _trial(1, 6.0, APART[1]), _trial(2, 1.0, APART[0]), _trial(2, 1.0, APART[2], parent="T1-01")
    matrix = [[1.0, 0.0, 0.2], [0.0, 1.0, 0.5], [0.2, 0.5, 1.0]]
    graph = _graph_of([strong, twin, linked], matrix=matrix)
    # Each relation is a third of P; T2-02's correlations 0.5 and 0.2 normalise to 5/7 and 2/7 of it, and its parent
    # T1-01 is joined by generation (1/3) and correlation (2/21), one neighbour of 9/21.
    assert [(h.trial, h.relations, round(h.weight, 12)) for h in graph.heaviest("T2-02", 5)] == [
        ("T1-01", ("generation", "correlation"), round(9 / 21, 12)), ("T2-01", ("correlation",), round(5 / 21, 12)),
    ]
    assert [h.trial for h in graph.heaviest("T2-02", 1)] == ["T1-01"]
    # No code is shared: the structure relation keeps every node to itself, and no node is its own neighbour.
    assert [(e.source, e.target) for e in graph.edges if e.relation == "structure"] == [
        ("T1-01", "T1-01"), ("T2-01", "T2-01"), ("T2-02", "T2-02"),
    ]
    assert all(h.trial != "T2-02" for h in graph.heaviest("T2-02", 5))


def test_each_node_ranks_in_the_pick_order_on_its_own_and_its_propagated_utilities_and_an_ineligible_one_not_at_all():
    records = [_trial(1, 3.0, APART[0]), _trial(1, 2.0, APART[1]), _trial(1, 1.0, APART[2], parent="T1-01")]
    graph = _graph_of(records, {"T1-01": Lineage(attempts=2), "T1-02": Lineage(fertility=0.5)}, max_children=2)
    ranks = graph.ranks()
    assert ranks["T1-01"] == (None, None)
    for propagated in (False, True):
        assert [trial for trial, _ in sorted(((t, r[propagated]) for t, r in ranks.items() if r[propagated] is not None),
                                             key=lambda p: p[1])] == ["T1-02", "T1-03"]
    assert ranks["T1-02"].own == 1 and ranks["T1-03"].propagated == 2


def test_the_graph_reads_back_from_its_file_dictionary_with_the_same_pick_order():
    records = [_trial(1, t, code, parent="T1-01" if i == 3 else "") for i, (t, code) in enumerate(zip([2.0, 2.0, 3.0, 1.0], APART))]
    lineage = {"T1-01": Lineage(attempts=1, fertility=0.5), "T1-03": Lineage(attempts=2)}
    matrix = [[1.0, 0.3, 0.2, 0.0], [0.3, 1.0, 0.0, 0.4], [0.2, 0.0, 1.0, 0.1], [0.0, 0.4, 0.1, 1.0]]
    graph = _graph_of(records, lineage, matrix, max_children=2)
    read = TrialGraph.from_dict(json.loads(json.dumps(graph.to_dict())))
    assert read == graph
    assert read.pick_order() == graph.pick_order() and read.pick_order(False) == graph.pick_order(False)
    assert set(graph.to_dict()["edges"]) == {"generation", "structure", "correlation"}


def test_a_rounds_graph_file_reads_back_as_the_graph_and_is_written_anew(tmp_path):
    records = [_trial(1, t, code) for t, code in zip([2.0, 1.0, 3.0], APART)]
    first = _graph_of(records[:2])
    write(first, tmp_path, 3)
    assert (tmp_path / "trial_graph" / "round_03.json").exists()
    assert read(tmp_path, 3) == first and read(tmp_path, 4) is None
    second = _graph_of(records, matrix=[[1.0, 0.5, 0.2], [0.5, 1.0, 0.0], [0.2, 0.0, 1.0]])
    write(second, tmp_path, 3)
    assert read(tmp_path, 3) == second and read(tmp_path, 3).gamma == 0.75


def test_a_round_without_an_evaluated_trial_has_an_empty_graph():
    unevaluated = TrialRecord(round=1, candidate={"description": "x", "mechanism_type": "momentum"}, status="rejected",
                              stage="gate", reasons=[], metrics=None)
    graph = build(named([unevaluated]), [], np.zeros((0, 0)), {}, VOCABULARY.classes, TrialGraphConfig())
    assert graph.nodes == () and graph.edges == () and graph.pick_order() == []


def test_without_propagation_the_linked_trial_ranks_on_its_own_utilities():
    strong = _trial(1, 6.0, APART[1])                     # T1-01
    twin = _trial(2, 1.0, APART[0])                       # T2-01
    linked = _trial(2, 1.0, APART[2], parent="T1-01")     # T2-02, a refinement of the strong trial
    graph = _graph_of([strong, twin, linked], settings=TrialGraphConfig(gamma=0.0))
    assert graph.gamma == 0.0
    assert all(n.propagated == n.own for n in graph.nodes)
    assert _order(graph.pick_order()) == ["T1-01", "T2-01", "T2-02"]


def test_the_relation_weights_decide_how_much_a_relation_lifts():
    strong, twin, linked = _trial(1, 6.0, APART[1]), _trial(2, 1.0, APART[0]), _trial(2, 1.0, APART[2], parent="T1-01")
    matrix = [[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]]   # the twin correlates with the strong trial
    # Generation alone lifts the child; correlation alone the twin.
    by_generation = TrialGraphConfig(weights={"generation": 1.0, "structure": 0.0, "correlation": 0.0})
    by_correlation = TrialGraphConfig(weights={"generation": 0.0, "structure": 0.0, "correlation": 1.0})
    assert _order(_rank([strong, twin, linked], matrix=matrix, settings=by_generation)) == ["T1-01", "T2-02", "T2-01"]
    assert _order(_rank([strong, twin, linked], matrix=matrix, settings=by_correlation)) == ["T1-01", "T2-01", "T2-02"]


def test_a_correlation_below_the_threshold_makes_no_edge():
    strong, twin, other = _trial(1, 6.0, APART[1]), _trial(2, 1.0, APART[0]), _trial(2, 1.0, APART[2])
    matrix = [[1.0, 0.3, 0.0], [0.3, 1.0, 0.0], [0.0, 0.0, 1.0]]
    lifted = {p.trial: p for p in _rank([strong, twin, other], matrix=matrix)}
    assert lifted["T2-01"].utilities.quality > lifted["T2-02"].utilities.quality
    apart = {p.trial: p for p in _rank([strong, twin, other], matrix=matrix, settings=TrialGraphConfig(min_correlation=0.5))}
    assert apart["T2-01"].utilities.quality == apart["T2-02"].utilities.quality


def test_the_expansion_cap_is_the_settings():
    records = [_trial(1, 9.0, APART[0]), _trial(1, 1.0, APART[1])]
    lineage = {"T1-01": Lineage(attempts=2, refinements=2)}
    assert _order(_rank(records, lineage)) == ["T1-01", "T1-02"]
    graph = _graph_of(records, lineage, settings=TrialGraphConfig(max_expansions=2))
    assert graph.node("T1-01").reason == "expansion_cap" and _order(graph.pick_order()) == ["T1-02"]


def test_novelty_is_measured_against_as_many_accepted_factors_as_the_settings_say():
    near = _trial(1, 2.0, APART[0], status="accepted")
    far = _trial(1, 2.0, APART[1], status="accepted")
    trial = _trial(2, 2.0, APART[2])
    matrix = [[1.0, 0.0, 0.6], [0.0, 1.0, 0.0], [0.6, 0.0, 1.0]]
    # The correlation distance is 0.4 to the near factor and 1 to the far one; no structure is shared.
    assert math.isclose(_graph_of([near, far, trial], matrix=matrix).node("T2-01").own.novelty, (0.7 + 1.0) / 2)
    nearest = _graph_of([near, far, trial], matrix=matrix, settings=TrialGraphConfig(nearest_accepted=1))
    assert math.isclose(nearest.node("T2-01").own.novelty, 0.7)
