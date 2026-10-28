"""Session aggregations: the operators that reduce each session's intraday bars to one value per stock and day.

Each is a marker: a KunQuant operator class that computes nothing itself.
Candidate code calls it like any operator and the Gate records it as one
node of the graph, so a graph's identity does not depend on how the
aggregation is computed; ``factor_agent.session.SessionFactor`` expands it
into KunQuant accumulators when it builds its function. An operator file
registers an aggregation like any operator, by its class here.
"""

from KunQuant.Op import OpBase


class SessionAggregation(OpBase):
    """The base of the session aggregations: a reduction of its inputs over each session's bars."""

    def __init__(self, *inputs: OpBase) -> None:
        super().__init__(list(inputs), None)


class SessionMean(SessionAggregation):
    """The mean of ``x`` over the session's bars where it has a value; NaN when it has none."""

    def __init__(self, x: OpBase) -> None:
        super().__init__(x)
