"""The Zoo: the factors a candidate must not rewrite, as graphs the Gate compares it with.

It holds one graph per output of every KunQuant factor of the reference set,
of the known factors and of every factor the campaign has accepted so far,
each entry tagged with where it comes from (``kind``: ``"reference"``,
``"known"`` or ``"accepted"``); a known factor's label says it is one, so a
refusal names it as known. A reference or known factor without a KunQuant
graph (a Polars factor, a view over a store) cannot be compared by graph;
its names are kept in ``not_covered`` so the gap is visible, and only the
correlation check of admission covers them. A view
or wrapper that keeps its factor's output names (``NeutralizedFactor``,
``StartClippedFactor``, ...) is compared by the graph of the factor it
wraps (``owner``).
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from KunQuant.Op import Output
from quantlab.factor.kunquant import FactorKunQuant

from factor_agent.gate.graph import Graph

if TYPE_CHECKING:
    from factor_agent.gate.vocabulary import Vocabulary


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


#: What a known factor's label adds after its source.
KNOWN = "a known factor"


@dataclass(frozen=True)
class ZooEntry:
    """One factor: its name, where it comes from (``source``, and ``kind``: reference, known or accepted), and its graph."""

    name: str
    source: str
    graph: Graph
    kind: str = "reference"

    @property
    def label(self) -> str:
        """``name (source)``, or ``name (source, a known factor)`` for a known one, as rejection reasons name the factor."""
        return _label(self.name, self.source, self.kind)


def _label(name: str, source: str, kind: str) -> str:
    return f"{name} ({source}, {KNOWN})" if kind == "known" else f"{name} ({source})"


class Zoo:
    """The factors the Gate checks originality against.

    Parameters
    ----------
    entries : sequence of ZooEntry
        The factors with a graph.
    not_covered : sequence of str
        Labels of reference and known factors without a KunQuant graph.
    """

    def __init__(self, entries: Iterable[ZooEntry] = (), not_covered: Iterable[str] = ()):
        self._entries = list(entries)
        self._not_covered = tuple(not_covered)

    @classmethod
    def from_reference(cls, factors, vocabulary: "Vocabulary", known=()) -> "Zoo":
        """Return the zoo of ``factors``, the reference set, and ``known``, the known factors: quantlab factor objects.

        Each output of a ``FactorKunQuant`` among ``config.factor_names`` is
        an entry named after it, with the factor's class as its source, its
        graph recorded under ``vocabulary``'s operator names; the names of
        every other factor go to ``not_covered``. The reference factors come
        first; ``known`` (``CampaignConfig.load_factors`` has dropped the
        outputs already in the reference set) are tagged as known.
        """
        entries, not_covered = [], []
        for kind, group in (("reference", factors), ("known", known)):
            for factor in group:
                names = factor.get_factor_names()
                factor = owner(factor)
                source = type(factor).__name__
                if not isinstance(factor, FactorKunQuant):
                    not_covered += [_label(name, source, kind) for name in names]
                    continue
                outputs = {op.attrs["name"]: op for op in factor.graph().ops if isinstance(op, Output)}
                for name in names:
                    graph = Graph.record(outputs[name].inputs[0], vocabulary, strict=False)
                    entries.append(ZooEntry(name, source, graph, kind))
        return cls(entries, not_covered)

    def add(self, name: str, source: str, graph: Graph) -> ZooEntry:
        """Add a factor the campaign has just accepted, and return its entry."""
        entry = ZooEntry(name, source, graph, "accepted")
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> tuple[ZooEntry, ...]:
        """Every factor with a graph: reference factors, then known factors, then accepted ones."""
        return tuple(self._entries)

    @property
    def covered(self) -> tuple[str, ...]:
        """The labels of the factors compared by graph."""
        return tuple(entry.label for entry in self._entries)

    @property
    def not_covered(self) -> tuple[str, ...]:
        """The labels of the reference and known factors the graph comparison cannot cover."""
        return self._not_covered
