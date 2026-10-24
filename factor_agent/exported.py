"""Accepted factors as quantlab factors: a ``FactorKunQuant`` whose outputs are recorded graphs.

Each output is a candidate's graph ranked across the whole market (KunQuant
``Rank``), the values the Evaluator scores (``factor_agent.scoring``), built by the
same ``ranked_function``. ``Acceptance.export`` writes one module per
factor, a subclass of ``GraphFactor`` that fixes ``GRAPHS`` and the
``CLASSES`` the campaign's vocabulary built them from;
``Acceptance.validate`` uses ``GraphFactor`` itself, with both in
``config.kwargs``.
"""

from collections.abc import Mapping

from KunQuant.Op import OpBase, Output, Rank
from KunQuant.ops import MulConst
from KunQuant.Stage import Function
from factor_agent import period_lag
from factor_agent.gate.graph import Graph
from factor_agent.gate.vocabulary import resolve_classes


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


class GraphFactor(period_lag.PeriodLagFactor):
    """Recorded graphs, each ranked across the market, as one factor.

    The graphs, name to ``Graph.to_json()``, are the class's ``GRAPHS`` (an
    exported factor) or, when it has none, ``config.kwargs["graphs"]``; the
    class path of each operator they record, ``CLASSES`` or
    ``config.kwargs["classes"]`` (``Vocabulary.class_paths``); how each lag
    column they read is computed, ``PERIOD_LAGS`` or
    ``config.kwargs["period_lags"]`` (``period_lags``), none when neither is
    set. ``config.data_columns`` must hold every dataset column the graphs
    and their lags read (``data_columns``).
    """

    #: Output name -> ``Graph.to_json()``; set by an exported subclass.
    GRAPHS: dict = {}
    #: Operator name -> class path, of every operator ``GRAPHS`` records; set by an exported subclass.
    CLASSES: dict = {}
    #: Lag column -> how it is computed, of every lag column ``GRAPHS`` reads; set by an exported subclass.
    PERIOD_LAGS: dict = {}

    def _recorded(self) -> dict:
        return type(self).GRAPHS or dict((self.config.kwargs or {})["graphs"])

    def _classes(self) -> dict:
        return resolve_classes(type(self).CLASSES or dict((self.config.kwargs or {})["classes"]))

    def _period_lags(self) -> dict:
        return type(self).PERIOD_LAGS or dict((self.config.kwargs or {}).get("period_lags") or {})

    def _get_factor_names(self) -> tuple[str, ...]:
        return tuple(self._recorded())

    def _get_factor_func(self) -> Function:
        classes = self._classes()
        return ranked_function({name: Graph.from_json(graph, classes) for name, graph in self._recorded().items()})

    @staticmethod
    def data_columns(graphs: Mapping[str, list], period_lags: Mapping[str, Mapping] | None = None) -> tuple[str, ...]:
        """The dataset columns the recorded graphs read, with the columns and keys of their ``period_lags``, sorted."""
        return period_lag.data_columns(_inputs(graphs), period_lags or {})

    @staticmethod
    def period_lags(graphs: Mapping[str, list], lag_columns: Mapping[str, Mapping]) -> dict:
        """The entries of ``lag_columns`` (``Vocabulary.lag_columns``) the recorded graphs read."""
        return period_lag.read(_inputs(graphs), lag_columns)


def _inputs(graphs: Mapping[str, list]) -> set[str]:
    """The names of the ``Input`` nodes of recorded graphs."""
    return {dict(node[1])["name"] for graph in graphs.values() for node in graph if node[0] == "Input"}
