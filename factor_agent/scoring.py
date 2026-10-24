"""Scoring: how a factor's raw values become the values that are judged.

One setting, shared by the Evaluator and Acceptance, so what is admitted is
what is validated and exported. A factor is

1. ranked across the whole market each day: the graph factors rank their own
   outputs (``exported.ranked_function``), so the rank is in the compiled
   graph and an exported factor computes it;
2. neutralised, when the config asks for it, against the industry code and
   the logarithm of the market cap of the configured exposures and z-scored
   (quantlab's ``NeutralizedFactor``), or left as ranked;
3. kept on the members of each day t (``IndexConstituentDataset.mask``), or
   on every symbol when there is no membership.

Steps 1 and 2 give the whole-market values a store holds (``apply``);
step 3 is how they, and any stored reference or baseline factor, are read
(``on_members``, ``read``, ``mask``). A stored factor is read from its
store's first bar on: it has no value, so no score, before.
"""

from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import xarray as xr
from quantlab.factor.config import (
    MembershipMaskedFactorConfig, NeutralizedConfig, RosterConfig, StartClippedConfig,
)
from quantlab.factor.predefined.membership_mask import MembershipMaskedFactor
from quantlab.factor.predefined.neutralized import NeutralizedFactor
from quantlab.factor.predefined.roster import RosterFactor
from quantlab.factor.predefined.start_clipped import StartClippedFactor

from factor_agent.config import DataConfig, ScoringConfig, load


@dataclass(frozen=True)
class Neutralization:
    """What a ranked factor is neutralised against.

    ``exposures`` is the dataset, or list of datasets, holding
    ``size_column`` (the market cap, whose logarithm is regressed on) and
    ``industry_column`` (the integer industry code): ``NeutralizedConfig``'s
    ``dataset``, ``size_column`` and ``industry_column``.
    """

    exposures: Any
    size_column: str
    industry_column: str


class Scoring:
    """Score factors one way; see the module docstring.

    Parameters
    ----------
    membership : quantlab IndexConstituentDataset or None
        The members of each day; None scores every symbol.
    neutralization : Neutralization or None
        What ranked factors are neutralised against; None leaves them ranked.
    njobs : int
        KunQuant executor threads of the neutralisation.
    """

    def __init__(self, membership=None, neutralization: Neutralization | None = None, njobs: int = 1):
        self.membership = membership
        self.neutralization = neutralization
        self._njobs = njobs

    @classmethod
    def from_config(cls, scoring: ScoringConfig, data: DataConfig, njobs: int) -> "Scoring":
        """The scoring of a campaign config: its ``scoring`` section and ``data.membership``."""
        membership = data.load("membership") if data.membership is not None else None
        neutralization = None
        if scoring.neutralize is not None:
            exposures = scoring.neutralize.exposures
            neutralization = Neutralization(
                [load(item) for item in exposures] if isinstance(exposures, list) else load(exposures),
                scoring.neutralize.size_column, scoring.neutralize.industry_column,
            )
        return cls(membership, neutralization, njobs)

    def apply(self, factor, file_path: str | Path | None = None, names: tuple[str, ...] | None = None):
        """``factor``, whose outputs are ranked across the market, as scored on the whole market.

        The neutralised factor, or ``factor`` itself without neutralisation;
        with ``file_path``, the returned factor's store is there (``build``
        writes it). ``names`` are the outputs scored, all by default; a
        neutralised factor holds only them, an unneutralised one keeps the
        rest unchanged.
        """
        path = None if file_path is None else str(file_path)
        if self.neutralization is None:
            return factor if path is None else type(factor)(replace(factor.config, file_path=path))
        n = self.neutralization
        pinned = {} if names is None else {"factor_names": tuple(names)}
        return NeutralizedFactor(NeutralizedConfig(
            factor=factor, dataset=n.exposures, size_column=n.size_column, industry_column=n.industry_column,
            file_path=path, njobs=self._njobs, **pinned,
        ))

    def on_members(self, factor):
        """``factor`` read from its store's first bar on, on the membership's roster, masked to the members of day t.

        Without membership, ``factor`` read from its store's first bar on, on every symbol.
        """
        clipped = StartClippedFactor(StartClippedConfig(factor=factor))
        if self.membership is None:
            return clipped
        viewed = RosterFactor(RosterConfig(factor=clipped, roster=self.membership))
        return MembershipMaskedFactor(MembershipMaskedFactorConfig(factor=viewed, membership=self.membership))

    def read(self, factor, start, end) -> Iterator[tuple[str, xr.DataArray]]:
        """Yield ``(name, values)`` of every output of ``factor`` from ``start`` to ``end``, masked to day-t members.

        A factor with a store is read from it, from its first bar on; any
        other is computed. The outputs are masked one at a time, so only one
        masked copy is in memory.
        """
        clipped = StartClippedFactor(StartClippedConfig(factor=factor))
        panel = clipped.read(start, end) if factor.store_range() is not None else factor.compute(start, end)
        for name in factor.get_factor_names():
            yield name, self.mask(panel[[name]])[name]

    def mask(self, panel: xr.Dataset) -> xr.Dataset:
        """``panel`` NaN where the symbol is not a member on the bar's date; unchanged without membership."""
        return panel if self.membership is None else self.membership.mask(panel)

    def describe(self) -> str:
        """One sentence for the prompts: what is done to a factor's values before they are judged."""
        if self.neutralization is None:
            neutral = "not neutralised"
        else:
            n = self.neutralization
            neutral = (
                f"neutralised against the industry ({n.industry_column}) and the logarithm of the market cap "
                f"({n.size_column}) and z-scored"
            )
        members = "on the members of the research universe that day" if self.membership is not None else "on every symbol"
        return f"A factor is ranked across the whole market each day, {neutral}, and scored {members}."
