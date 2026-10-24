"""The Zoo: the factors a candidate must not rewrite, as graphs the Gate compares it with.

It holds one graph per output of every KunQuant factor of the reference set
and every factor the campaign has accepted so far. A reference factor
without a KunQuant graph (a Polars factor, a view over a store) cannot be
compared by graph; its names are kept in ``not_covered`` so the gap is
visible, and only the correlation check of admission covers them. A view
or wrapper that keeps its factor's output names (``NeutralizedFactor``,
``StartClippedFactor``, ...) is compared by the graph of the factor it
wraps (``owner``).
"""

from collections.abc import Iterable
from dataclasses import dataclass

from KunQuant.Op import Output
from quantlab.factor.kunquant import FactorKunQuant

from factor_agent.gate.graph import Graph


def owner(factor):
    """The factor whose outputs ``factor`` shows: through every wrapper (``config.factor``) that keeps the names.

    The walk stops at a ``FactorKunQuant``, at a factor that wraps none, or
    before a wrapped factor lacking one of the outer names (a composite
    that derives new outputs); that factor is returned.
    """
    names = set(factor.get_factor_names())
    while not isinstance(factor, FactorKunQuant):
        inner = getattr(factor.config, "factor", None)
        if inner is None or not names <= set(inner.get_factor_names()):
            break
        factor = inner
    return factor


@dataclass(frozen=True)
class ZooEntry:
    """One factor: its name, where it comes from, and its graph."""

    name: str
    source: str
    graph: Graph

    @property
    def label(self) -> str:
        """``name (source)``, as rejection reasons name the factor."""
        return f"{self.name} ({self.source})"


class Zoo:
    """The factors the Gate checks originality against.

    Parameters
    ----------
    entries : sequence of ZooEntry
        The factors with a graph.
    not_covered : sequence of str
        Labels of reference factors without a KunQuant graph.
    """

    def __init__(self, entries: Iterable[ZooEntry] = (), not_covered: Iterable[str] = ()):
        self._entries = list(entries)
        self._not_covered = tuple(not_covered)

    @classmethod
    def from_reference(cls, factors) -> "Zoo":
        """Return the zoo of ``factors``, quantlab factor objects.

        Each output of a ``FactorKunQuant`` among ``config.factor_names`` is
        an entry named after it, with the factor's class as its source; the
        names of every other factor go to ``not_covered``.
        """
        entries, not_covered = [], []
        for factor in factors:
            names = factor.get_factor_names()
            factor = owner(factor)
            source = type(factor).__name__
            if not isinstance(factor, FactorKunQuant):
                not_covered += [f"{name} ({source})" for name in names]
                continue
            outputs = {op.attrs["name"]: op for op in factor.graph().ops if isinstance(op, Output)}
            for name in names:
                entries.append(ZooEntry(name, source, Graph.record(outputs[name].inputs[0], strict=False)))
        return cls(entries, not_covered)

    def add(self, name: str, source: str, graph: Graph) -> ZooEntry:
        """Add a factor, such as one the campaign has just accepted, and return its entry."""
        entry = ZooEntry(name, source, graph)
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> tuple[ZooEntry, ...]:
        """Every factor with a graph, reference factors first."""
        return tuple(self._entries)

    @property
    def covered(self) -> tuple[str, ...]:
        """The labels of the factors compared by graph."""
        return tuple(entry.label for entry in self._entries)

    @property
    def not_covered(self) -> tuple[str, ...]:
        """The labels of the reference factors the graph comparison cannot cover."""
        return self._not_covered
