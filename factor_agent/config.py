"""The campaign config: one YAML file holds every setting of a run.

``data`` and ``reference`` hold quantlab components: each is either a
``get_config()`` dict inline or the path of a store folder whose
``component.json`` describes it (``quantlab.core.store_folder.save_component``).
``reference`` lists the reference set: the factors candidates must be
original against (the ``Zoo``) and, in admission, uncorrelated with and
incremental to; a factor's ``factor_names`` selects its columns.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from quantlab.core.component import rebuild
from quantlab.core.store_folder import load_component

from factor_agent.candidate import MECHANISM_TYPES
from factor_agent.segments import Segment


@dataclass(frozen=True)
class DataConfig:
    """The data the Evaluator reads.

    ``prices`` is the whole-market price dataset, ``label`` the forward
    label masked by day-t membership, ``membership`` the members of each
    day (an ``IndexConstituentDataset``) and ``exposures`` the market cap
    and industry dataset, or a list of datasets, that neutralisation
    regresses on (``NeutralizedConfig.dataset``).
    """

    prices: dict | str
    label: dict | str
    membership: dict | str
    exposures: dict | str | list

    def load(self, name: str) -> Any:
        """Rebuild the component ``name``; a list of components is rebuilt item by item."""
        value = getattr(self, name)
        return [load(item) for item in value] if isinstance(value, list) else load(value)


def load(component: dict | str) -> Any:
    """Rebuild a component given as a ``get_config()`` dict or a store folder path."""
    return load_component(component) if isinstance(component, str) else rebuild(component)


@dataclass(frozen=True)
class EvaluatorConfig:
    """How the Evaluator computes the round's graph."""

    #: Bars read before the segment start so every window is filled.
    warmup_bars: int = 260
    #: KunQuant executor threads.
    njobs: int = 51


@dataclass(frozen=True)
class GateConfig:
    """The Gate's sandbox limits and graph constraints."""

    #: Seconds the sandboxed graph build may take.
    timeout_seconds: float = 10.0
    #: Memory, in MB, the sandboxed graph build may use.
    memory_mb: int = 1024
    #: Nodes a candidate's graph may have.
    max_nodes: int = 200
    #: Hand-set constants (other than 0 and +-1) per operator.
    max_free_parameter_ratio: float = 0.5
    #: Distinct input fields (open, high, low, close, volume, vwap, amount) a candidate may read.
    max_base_fields: int = 5
    #: Nodes of the largest subtree a candidate may share with a zoo factor.
    max_common_subtree_nodes: int = 5


@dataclass(frozen=True)
class AdmissionConfig:
    """What an evaluated candidate needs to be accepted."""

    #: Newey-West t of the partial IC the first evaluated trial needs; the
    #: threshold rises with the trial count (Bonferroni, see ``_admission``).
    t_threshold: float = 3.0
    #: Largest absolute cross-sectional correlation with a reference or accepted factor.
    max_correlation: float = 0.7
    #: Smallest Spearman correlation of the decile returns with the decile number.
    min_monotonicity: float = 0.6


@dataclass(frozen=True)
class RoundConfig:
    """The size of a round, the planner's quotas and when the campaign stops."""

    #: Candidates asked for per round.
    candidates: int = 20
    #: Candidates of every mechanism type per round; the rest are allocated by Thompson sampling.
    min_per_type: int = 3
    #: Times a candidate the parser or the Gate refused is sent back to the proposer to fix.
    retries: int = 2
    #: Rounds the campaign runs at most.
    max_rounds: int = 30
    #: Trials (candidates, failed ones included) the campaign runs at most; a round that
    #: would go past it is not started.
    max_trials: int = 600
    #: Consecutive rounds without an acceptance after which the campaign stops.
    patience: int = 5
    #: US dollars the LLM calls may cost; the campaign stops once they have. None: no limit.
    budget_usd: float | None = None
    #: Rounds between two distillations of the ledger into the memory.
    memory_every: int = 5
    #: Seed of the Thompson sampling; round n draws from ``(seed, n)``.
    seed: int = 0

    def __post_init__(self):
        if self.candidates < self.min_per_type * len(MECHANISM_TYPES):
            raise ValueError(
                f"round: {self.candidates} candidates cannot hold {self.min_per_type} of each of "
                f"{len(MECHANISM_TYPES)} mechanism types"
            )


@dataclass(frozen=True)
class RoleConfig:
    """One role's LLM: the backend name, the model and the backend's own settings."""

    backend: str
    model: str
    options: dict

    @classmethod
    def from_dict(cls, data: dict) -> "RoleConfig":
        """Split a role's YAML mapping into the backend, the model and the rest."""
        data = dict(data)
        return cls(backend=data.pop("backend"), model=data.pop("model"), options=data)

    def spec(self) -> dict:
        """The dict ``make_backend`` takes."""
        return {"backend": self.backend, "model": self.model, **self.options}


#: The LLM roles of a round, each configured under ``llm``.
ROLES = ("planner", "proposer", "reviewer", "distiller")


@dataclass(frozen=True)
class CampaignConfig:
    """Every setting of a campaign, read from one YAML file with ``from_file``.

    ``roles`` maps each of ``ROLES`` to its LLM; the roles may use
    different backends, so the reviewer can be another provider's model.
    ``directions`` are the seed research directions the planner is shown.
    """

    run_dir: str
    training: Segment
    data: DataConfig
    roles: dict
    #: The reference factors, as components; see the module docstring.
    reference: tuple = ()
    directions: tuple = ()
    evaluator: EvaluatorConfig = EvaluatorConfig()
    gate: GateConfig = GateConfig()
    admission: AdmissionConfig = AdmissionConfig()
    round: RoundConfig = RoundConfig()

    @classmethod
    def from_file(cls, path: str | Path) -> "CampaignConfig":
        """Read a campaign config from YAML.

        Raises
        ------
        TypeError
            If a section holds a key its config class does not have.
        KeyError
            If a required section or role is missing.
        ValueError
            If a section's values do not fit together.
        """
        raw = yaml.safe_load(Path(path).read_text())
        sections = {
            "evaluator": EvaluatorConfig, "gate": GateConfig,
            "admission": AdmissionConfig, "round": RoundConfig,
        }
        unknown = set(raw) - {"run_dir", "segments", "data", "reference", "directions", "llm", *sections}
        if unknown:
            raise TypeError(f"campaign config: unknown section(s) {sorted(unknown)}")
        unknown_roles = set(raw["llm"]) - set(ROLES)
        if unknown_roles:
            raise TypeError(f"campaign config: unknown LLM role(s) {sorted(unknown_roles)}; known: {list(ROLES)}")
        training = raw["segments"]["training"]
        return cls(
            run_dir=raw["run_dir"],
            training=Segment("training", str(training["start"]), str(training["end"])),
            data=DataConfig(**raw["data"]),
            roles={role: RoleConfig.from_dict(raw["llm"][role]) for role in ROLES},
            reference=tuple(raw.get("reference") or ()),
            directions=tuple(raw.get("directions") or ()),
            **{name: klass(**raw[name]) for name, klass in sections.items() if name in raw},
        )
