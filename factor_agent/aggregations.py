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


class SessionSum(SessionAggregation):
    """The sum of ``x`` over the session's bars where it has a value; 0 when it has none."""

    def __init__(self, x: OpBase) -> None:
        super().__init__(x)


class SessionCount(SessionAggregation):
    """The number of the session's bars where ``x`` has a value; 0 when it has none."""

    def __init__(self, x: OpBase) -> None:
        super().__init__(x)


class SessionStd(SessionAggregation):
    """The unbiased standard deviation of ``x`` over the session's bars where it has a value; NaN below two."""

    def __init__(self, x: OpBase) -> None:
        super().__init__(x)


class SessionSkew(SessionAggregation):
    """The unbiased skewness of ``x`` over the session's bars where it has a value; NaN below three or when constant."""

    def __init__(self, x: OpBase) -> None:
        super().__init__(x)


class SessionMin(SessionAggregation):
    """The least value of ``x`` in the session; NaN when it has none."""

    def __init__(self, x: OpBase) -> None:
        super().__init__(x)


class SessionMax(SessionAggregation):
    """The greatest value of ``x`` in the session; NaN when it has none."""

    def __init__(self, x: OpBase) -> None:
        super().__init__(x)


class SessionFirst(SessionAggregation):
    """The first value of ``x`` present in the session; NaN when it has none."""

    def __init__(self, x: OpBase) -> None:
        super().__init__(x)


class SessionLast(SessionAggregation):
    """The last value of ``x`` present in the session; NaN when it has none."""

    def __init__(self, x: OpBase) -> None:
        super().__init__(x)


class SessionWeightedMean(SessionAggregation):
    """The mean of ``x`` weighted by ``w`` over the bars where both have a value; NaN when the weights sum to 0."""

    def __init__(self, x: OpBase, w: OpBase) -> None:
        super().__init__(x, w)


class SessionCorr(SessionAggregation):
    """The correlation of ``x`` and ``y`` over the bars where both have a value; NaN when one is constant there."""

    def __init__(self, x: OpBase, y: OpBase) -> None:
        super().__init__(x, y)
