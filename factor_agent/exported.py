"""Accepted factors as quantlab factors: a ``FactorKunQuant`` whose outputs are recorded graphs.

Each output is a candidate's graph ranked across the whole market (KunQuant
``Rank``), the values the Evaluator neutralises and scores, built by the
same ``ranked_function``. ``Acceptance.export`` writes one module per
factor, a subclass of ``GraphFactor`` that fixes ``GRAPHS``;
``Acceptance.validate`` uses ``GraphFactor`` itself, with the graphs in
``config.kwargs["graphs"]``.
"""

from collections.abc import Mapping

from KunQuant.Op import OpBase, Output, Rank
from KunQuant.ops import MulConst
from KunQuant.Stage import Function
from quantlab.factor.kunquant import FactorKunQuant

from factor_agent.gate.graph import Graph


def ranked_function(graphs: Mapping[str, Graph], presence: Mapping[str, str] | None = None) -> Function:
    """One KunQuant function with an output per graph, named after it, ranked across the market.

    KunQuant's ``Rank`` ranks a missing value too, so the ranked output has a
    value wherever the symbol has a bar. ``presence`` maps a graph's name to
    the name of a second output, 0 where the graph itself has a value and NaN
    where it has none.
    """
    inputs: dict[str, OpBase] = {}
    ops: list[OpBase] = []
    for name, graph in graphs.items():
        output, new = graph.build(inputs)
        ranked = Rank(output)
        ops += [*new, ranked, Output(ranked, name)]
        if presence and name in presence:
            zero = MulConst(output, 0.0)
            ops += [zero, Output(zero, presence[name])]
    return Function(ops)


class GraphFactor(FactorKunQuant):
    """Recorded graphs, each ranked across the market, as one factor.

    The graphs, name to ``Graph.to_json()``, are the class's ``GRAPHS`` (an
    exported factor) or, when it has none, ``config.kwargs["graphs"]``.
    ``config.data_columns`` must hold every column the graphs read
    (``data_columns``).
    """

    #: Output name -> ``Graph.to_json()``; set by an exported subclass.
    GRAPHS: dict = {}

    def _recorded(self) -> dict:
        return type(self).GRAPHS or dict((self.config.kwargs or {})["graphs"])

    def _get_factor_names(self) -> tuple[str, ...]:
        return tuple(self._recorded())

    def _get_factor_func(self) -> Function:
        return ranked_function({name: Graph.from_json(graph) for name, graph in self._recorded().items()})

    @staticmethod
    def data_columns(graphs: Mapping[str, list]) -> tuple[str, ...]:
        """The dataset columns the recorded graphs read, sorted."""
        return tuple(sorted({c for graph in graphs.values() for c in Graph.from_json(graph).input_names}))
