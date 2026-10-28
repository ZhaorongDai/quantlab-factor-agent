"""Accepted factors as quantlab factors: a ``FactorKunQuant`` whose outputs are recorded graphs.

Each output is a candidate's graph ranked across the whole market (KunQuant
``Rank``), the values the Evaluator scores (``factor_agent.scoring``), built by the
same ``ranked_function``. ``Acceptance.export`` writes one module per
factor, a subclass of ``GraphFactor`` that fixes ``GRAPHS`` and the
``CLASSES`` the campaign's vocabulary built them from;
``Acceptance.validate`` uses ``GraphFactor`` itself, with both in
``config.kwargs``.

A factor of an intraday family is held in its stages
(``factor_agent.gate.stages``, ``staged``): its graph is the daily stage,
reading each session column as an input, and ``SESSIONS`` maps each session
column to its intraday graph, computed from the intraday dataset of
``INTRADAY`` by ``factor_agent.session.SessionColumns``, as the Evaluator
computes it.
"""

from collections.abc import Mapping

from KunQuant.Op import OpBase, Output, Rank
from KunQuant.ops import AddConst, MulConst
from KunQuant.Stage import Function
from quantlab.core.component import rebuild

from factor_agent import derived, period_lag
from factor_agent.gate import stages
from factor_agent.gate.graph import Graph
from factor_agent.gate.vocabulary import Vocabulary, resolve_classes
from factor_agent.session import SessionColumns


def ranked_function(
    graphs: Mapping[str, Graph], presence: Mapping[str, str] | None = None, directions: Mapping[str, int] | None = None,
) -> Function:
    """One KunQuant function with an output per graph, named after it, ranked across the market.

    A graph whose ``directions`` entry is -1 (a higher value went with a lower
    return) is output as one minus its rank, so every output reads the same
    way: a higher value, a higher return.

    KunQuant's ``Rank`` leaves a missing value missing and ranks the finite
    values among themselves. ``presence`` maps a graph's name to the name of
    a second output, 0 where the graph itself has a value and NaN where it
    has none, read raw (never neutralised, so a symbol without exposures
    still counts) and before any fill: it is computed from
    ``Graph.without_fills``, so a missing value a ``SetInfOrNanToValue``
    fills stays missing in it.
    """
    inputs: dict[str, OpBase] = {}
    ops: list[OpBase] = []
    for name, graph in graphs.items():
        output, new = graph.build(inputs)
        ranked = Rank(output)
        ops += [*new, ranked]
        if directions and directions.get(name) == -1:
            flipped = MulConst(ranked, -1.0)
            ranked = AddConst(flipped, 1.0)
            ops += [flipped, ranked]
        ops.append(Output(ranked, name))
        if presence and name in presence:
            unfilled = graph.without_fills()
            if unfilled is not graph:
                output, new = unfilled.build(inputs)
                ops += new
            zero = MulConst(output, 0.0)
            ops += [zero, Output(zero, presence[name])]
    return Function(ops)


class GraphFactor(derived.DerivedColumnsFactor):
    """Recorded graphs, each ranked across the market, as one factor.

    The graphs, name to ``Graph.to_json()``, are the class's ``GRAPHS`` (an
    exported factor) or, when it has none, ``config.kwargs["graphs"]``; the
    class path of each operator they record, ``CLASSES`` or
    ``config.kwargs["classes"]`` (``Vocabulary.class_paths``); how each lag
    column they read is computed, ``PERIOD_LAGS`` or
    ``config.kwargs["period_lags"]`` (``period_lags``), none when neither is
    set; each output's direction, ``DIRECTIONS`` or
    ``config.kwargs["directions"]`` (+1 when unset), turns it so a higher
    value means a higher return. A graph of an intraday family is its daily
    stage (``staged``): the intraday graph of each session column it reads
    is in ``SESSIONS`` or ``config.kwargs["sessions"]``, and the intraday
    dataset they run on, as a ``get_config()`` dict, in ``INTRADAY`` or
    ``config.kwargs["intraday"]``. ``config.data_columns`` must hold every
    dataset column the graphs and their lags read (``data_columns``).
    """

    #: Output name -> ``Graph.to_json()``; set by an exported subclass.
    GRAPHS: dict = {}
    #: Operator name -> class path, of every operator ``GRAPHS`` records; set by an exported subclass.
    CLASSES: dict = {}
    #: Lag column -> how it is computed, of every lag column ``GRAPHS`` reads; set by an exported subclass.
    PERIOD_LAGS: dict = {}
    #: Output name -> its direction, +1 or -1; set by an exported subclass.
    DIRECTIONS: dict = {}
    #: Session column -> its intraday graph, of every session column ``GRAPHS`` reads; set by an exported subclass.
    SESSIONS: dict = {}
    #: The intraday dataset ``SESSIONS`` runs on, as a ``get_config()`` dict; set by an exported subclass.
    INTRADAY: dict = {}

    def _recorded(self) -> dict:
        return type(self).GRAPHS or dict((self.config.kwargs or {})["graphs"])

    def _class_paths(self) -> dict:
        return type(self).CLASSES or dict((self.config.kwargs or {})["classes"])

    def _classes(self) -> dict:
        return resolve_classes(self._class_paths())

    def _period_lags(self) -> dict:
        return type(self).PERIOD_LAGS or dict((self.config.kwargs or {}).get("period_lags") or {})

    def _sessions(self) -> dict:
        return type(self).SESSIONS or dict((self.config.kwargs or {}).get("sessions") or {})

    def _providers(self) -> list[derived.Provider]:
        providers = period_lag.providers(self._period_lags())
        sessions = self._sessions()
        if sessions:
            intraday = type(self).INTRADAY or (self.config.kwargs or {})["intraday"]
            providers.append(SessionColumns(rebuild(intraday), sessions, self._class_paths(), self.config.njobs))
        return providers

    def _directions(self) -> dict:
        return type(self).DIRECTIONS or dict((self.config.kwargs or {}).get("directions") or {})

    def _get_factor_names(self) -> tuple[str, ...]:
        return tuple(self._recorded())

    def _get_factor_func(self) -> Function:
        classes = self._classes()
        graphs = {name: Graph.from_json(graph, classes) for name, graph in self._recorded().items()}
        return ranked_function(graphs, directions=self._directions())

    @staticmethod
    def data_columns(
        graphs: Mapping[str, list], period_lags: Mapping[str, Mapping] | None = None, sessions: Mapping[str, list] | None = None,
    ) -> tuple[str, ...]:
        """The daily dataset columns the recorded graphs read, with the columns and keys of their ``period_lags``, sorted.

        The session columns of ``sessions`` are filled from the intraday dataset, not read.
        """
        return tuple(sorted(set(derived.data_columns(_inputs(graphs), period_lag.providers(period_lags or {})))
                            - set(sessions or {})))

    @staticmethod
    def staged(graphs: Mapping[str, list], vocabulary: Vocabulary) -> tuple[dict, dict]:
        """Recorded graphs in their stages: each one's daily graph, by name, and every session column's intraday graph.

        A graph of no intraday family is its own daily graph and reads no
        session column. Both are ``Graph.to_json()``.

        Raises
        ------
        ValueError
            If a graph breaks a stage rule (``stages.split``).
        """
        daily, sessions = {}, {}
        for name, recorded in graphs.items():
            split = stages.split(Graph.from_json(recorded, vocabulary.classes), vocabulary)
            if split.reasons:
                raise ValueError(f"{name}: the graph breaks the stage rules: {'; '.join(split.reasons)}")
            daily[name] = split.daily.to_json()
            sessions |= {column: graph.to_json() for column, graph in split.intraday.items()}
        return daily, sessions

    @staticmethod
    def period_lags(graphs: Mapping[str, list], lag_columns: Mapping[str, Mapping]) -> dict:
        """The entries of ``lag_columns`` (``Vocabulary.lag_columns``) the recorded graphs read."""
        return period_lag.read(_inputs(graphs), lag_columns)


def _inputs(graphs: Mapping[str, list]) -> set[str]:
    """The names of the ``Input`` nodes of recorded graphs."""
    return {dict(node[1])["name"] for graph in graphs.values() for node in graph if node[0] == "Input"}
