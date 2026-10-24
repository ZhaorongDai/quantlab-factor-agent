"""The campaign config: one YAML file holds every setting of a run.

``data`` and ``reference`` hold quantlab components: each is a
``get_config()`` dict inline, the path of a store folder whose
``component.json`` describes it (``quantlab.core.store_folder.save_component``),
or ``{store: <folder>, factor_names: [...]}``, that store's factor pinned to
some of its outputs. ``reference`` lists the reference set: the factors
candidates must be original against (the ``Zoo``) and, in admission,
uncorrelated with and incremental to; a factor's ``factor_names`` selects
its columns.

``fields`` (field files, one per input family) and ``operators`` (the
operator file) make the ``Vocabulary`` candidates write with; the
framework has none of its own. ``operator_modules`` lists the modules,
besides KunQuant and ``quantlab.factor``, an operator may come from.

``market`` describes the data in the prompts, ``mechanisms`` lists the
mechanism types candidates belong to (by default ``DEFAULT_MECHANISMS``),
each with a one-line description, and ``prompts`` maps a template's name
(see ``_prompts.TEMPLATES``) to the file that replaces it.
"""

import dataclasses

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from quantlab.core.component import rebuild
from quantlab.core.store_folder import load_component

from factor_agent._prompts import Templates
from factor_agent.gate.vocabulary import Vocabulary
from factor_agent.segments import PURGE_BARS, Segment, Segments
from factor_agent.snapshot import SearchFiles



@dataclass(frozen=True)
class DataConfig:
    """The data the Evaluator reads.

    ``prices`` is the whole-market price dataset, ``label`` the forward
    label (masked by day-t membership when there is a membership) and
    ``membership`` the members of each day (an ``IndexConstituentDataset``)
    or None, when every symbol is scored.
    """

    prices: dict | str
    label: dict | str
    membership: dict | str | None = None

    def load(self, name: str) -> Any:
        """Rebuild the component ``name``; a list of components is rebuilt item by item."""
        value = getattr(self, name)
        return [load(item) for item in value] if isinstance(value, list) else load(value)


@dataclass(frozen=True)
class NeutralizeConfig:
    """What ranked factors are neutralised against.

    ``exposures`` is the dataset, or list of datasets, as components, that
    holds ``size_column`` (the market cap; its logarithm is regressed on) and
    ``industry_column`` (the integer industry code).
    """

    exposures: dict | str | list
    size_column: str
    industry_column: str


@dataclass(frozen=True)
class ScoringConfig:
    """How a factor's raw values become the values that are judged (``factor_agent.scoring``).

    ``neutralize`` is what ranked factors are neutralised against, or None
    to score them as ranked. It must be given, as null to turn
    neutralisation off, so no campaign is scored one way by default.
    """

    neutralize: NeutralizeConfig | None

    @classmethod
    def from_dict(cls, raw: dict) -> "ScoringConfig":
        """Read the ``scoring`` section.

        Raises
        ------
        TypeError
            If the section holds another key than ``neutralize``, or ``neutralize`` another key than its fields.
        KeyError
            If ``neutralize`` is missing.
        """
        unknown = set(raw) - {"neutralize"}
        if unknown:
            raise TypeError(f"campaign config: unknown scoring key(s) {sorted(unknown)}")
        if "neutralize" not in raw:
            raise KeyError("campaign config: scoring needs neutralize (null to score the ranked values)")
        neutralize = raw["neutralize"]
        return cls(neutralize=None if neutralize is None else NeutralizeConfig(**neutralize))


def load(component: dict | str) -> Any:
    """Rebuild a component given as a ``get_config()`` dict, a store folder path or a store entry.

    Raises
    ------
    ValueError
        If a store entry names an output its store does not have.
    TypeError
        If a store entry holds a key other than ``store`` and ``factor_names``, or names
        outputs of a store whose component has no ``factor_names`` (a dataset).
    """
    if isinstance(component, str):
        return load_component(component)
    if "store" not in component:
        return rebuild(component)
    unknown = set(component) - {"store", "factor_names"}
    if unknown:
        raise TypeError(f"store entry {component['store']}: unknown key(s) {sorted(unknown)}")
    stored = load_component(component["store"])
    names = component.get("factor_names")
    if names is None:
        return stored
    missing = [name for name in names if name not in stored.get_factor_names()]
    if missing:
        raise ValueError(f"store entry {component['store']}: no output(s) {missing}")
    return type(stored)(dataclasses.replace(stored.config, factor_names=tuple(names)))


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
    #: Smallest mean daily share of the members on which the candidate has a value.
    min_coverage: float = 0.5


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


@dataclass(frozen=True)
class AcceptanceConfig:
    """How ``Acceptance`` validates, opens the hold-out and exports.

    ``features`` are the baseline model's features (SEL0's), as components;
    empty, the reference set is used. Every feature, and each accepted
    factor's scored values, is read as the campaign's ``scoring`` reads.
    """

    features: tuple = ()
    #: The xgb hyperparameters (``ModelConfig.hyperparameters``); empty: quantlab's XGBoost defaults.
    hyperparameters: dict = field(default_factory=dict)
    #: The seeds every model is trained with.
    seeds: tuple[int, ...] = (0, 1, 2)
    #: Share of the training window the xgb validates (early-stops) on, at its end.
    val_size: float = 0.2
    #: Members held by the plain TopK book.
    top_k: int = 100
    #: Where exported stores go, ``<factor_root>/<name>/``; an export needs it.
    factor_root: str | None = None
    #: The package the exported ``FactorKunQuant`` subclasses are written into, one module each.
    package: str = "factor_agent.accepted"
    #: First date of an exported store; None: the training segment's start.
    export_start: str | None = None
    #: Last date of an exported store; None: the hold-out's end, else the validation's.
    export_end: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "features", tuple(self.features))
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if not self.seeds:
            raise ValueError("acceptance: at least one seed is needed")


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
class Mechanism:
    """A mechanism type a candidate's hypothesis belongs to, with the line the prompts describe it by."""

    name: str
    description: str


#: The mechanism types of a config that names none.
DEFAULT_MECHANISMS = (
    Mechanism("behavioural_bias", "investors' systematic errors (over- and underreaction, anchoring, limited "
              "attention) that prices correct later"),
    Mechanism("risk_compensation", "a return paid for bearing a risk investors demand to be compensated for"),
    Mechanism("liquidity", "the cost and the pressure of trading: illiquidity premia, liquidity demand and its "
              "reversal"),
    Mechanism("microstructure", "the mechanics of trading within and between bars: order flow, the bid-ask "
              "bounce, the open and the close"),
    Mechanism("information_diffusion", "news that reaches prices gradually, across investors, related firms "
              "or time"),
)


@dataclass(frozen=True)
class CampaignConfig:
    """Every setting of a campaign, read from one YAML file with ``from_file``.

    ``roles`` maps each of ``ROLES`` to its LLM; the roles may use
    different backends, so the reviewer can be another provider's model.
    ``directions`` are the seed research directions the planner is shown.
    ``market`` describes the data in the prompts, ``mechanisms`` are the
    mechanism types in plan order, and ``prompts`` maps a template's name to
    the file that replaces it; the replacing files are read and checked for
    their placeholders when the config is made.

    Raises
    ------
    PromptError
        Naming every problem of the replaced templates.
    ValueError
        If the mechanism types repeat or a round cannot hold its quota of them.
    """

    run_dir: str
    segments: Segments
    data: DataConfig
    scoring: ScoringConfig
    roles: dict
    #: The field files, one per input family candidates may read; see ``Vocabulary``.
    fields: tuple
    #: The operator file: the operators candidates may call and their parameter values.
    operators: str
    #: The data as the prompts describe it, such as "adjusted daily bars of every US common stock".
    market: str
    #: Modules besides KunQuant and ``quantlab.factor`` an operator's class or callable may come from.
    operator_modules: tuple = ()
    #: The reference factors, as components; see the module docstring.
    reference: tuple = ()
    directions: tuple = ()
    evaluator: EvaluatorConfig = EvaluatorConfig()
    gate: GateConfig = GateConfig()
    admission: AdmissionConfig = AdmissionConfig()
    round: RoundConfig = RoundConfig()
    acceptance: AcceptanceConfig = AcceptanceConfig()
    mechanisms: tuple = DEFAULT_MECHANISMS
    prompts: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "mechanisms", tuple(self.mechanisms))
        Templates.load(self.prompts)  # every problem of the replaced templates, before anything runs
        names = self.mechanism_types
        if not names or len(set(names)) != len(names):
            raise ValueError(f"campaign config: the mechanism types {list(names)} must be given and distinct")
        if self.round.candidates < self.round.min_per_type * len(names):
            raise ValueError(
                f"round: {self.round.candidates} candidates cannot hold {self.round.min_per_type} of each of "
                f"{len(names)} mechanism types"
            )

    @property
    def mechanism_types(self) -> tuple[str, ...]:
        """The names of the mechanism types, in plan order."""
        return tuple(m.name for m in self.mechanisms)

    @property
    def training(self) -> Segment:
        """The segment the campaign evaluates."""
        return self.segments.training

    def search_files(self) -> SearchFiles:
        """The files the run searches with: the field files, the operator file and the replaced templates."""
        return SearchFiles(self.fields, self.operators, self.prompts)

    def templates(self, files: SearchFiles | None = None) -> Templates:
        """The prompt templates: the built-in ones, those ``files`` replace read from their files.

        ``files`` are the config's own by default, or a run snapshot's copies.

        Raises
        ------
        PromptError
            With every problem of the replacing files.
        """
        return Templates.load((files or self.search_files()).prompts)

    def vocabulary(self, columns=None, files: SearchFiles | None = None) -> Vocabulary:
        """The Vocabulary of the field and operator files; ``columns``, given, are the price dataset's.

        ``files`` are the config's own by default, or a run snapshot's copies.

        Raises
        ------
        VocabularyError
            With every problem of the files, a field column missing from ``columns`` included.
        """
        files = files or self.search_files()
        return Vocabulary.load(files.fields, files.operators, self.operator_modules, columns)

    @classmethod
    def from_file(cls, path: str | Path) -> "CampaignConfig":
        """Read a campaign config from YAML; the paths of ``fields``, ``operators`` and ``prompts`` are relative to its folder.

        Raises
        ------
        TypeError
            If a section holds a key its config class does not have.
        KeyError
            If a required section or role is missing.
        ValueError
            If a section's values do not fit together.
        PromptError
            Naming every problem of the replaced templates.
        """
        raw = yaml.safe_load(Path(path).read_text())
        folder = Path(path).parent
        sections = {
            "evaluator": EvaluatorConfig, "gate": GateConfig,
            "admission": AdmissionConfig, "round": RoundConfig, "acceptance": AcceptanceConfig,
        }
        unknown = set(raw) - {
            "run_dir", "segments", "data", "scoring", "fields", "operators", "operator_modules", "reference",
            "directions", "llm", "market", "mechanisms", "prompts", *sections,
        }
        if unknown:
            raise TypeError(f"campaign config: unknown section(s) {sorted(unknown)}")
        unknown_roles = set(raw["llm"]) - set(ROLES)
        if unknown_roles:
            raise TypeError(f"campaign config: unknown LLM role(s) {sorted(unknown_roles)}; known: {list(ROLES)}")
        if "market" not in raw:
            raise KeyError("campaign config: market is needed, the data as the prompts describe it")
        optional = {}
        if raw.get("mechanisms") is not None:
            optional["mechanisms"] = tuple(Mechanism(**m) for m in raw["mechanisms"])
        return cls(
            run_dir=raw["run_dir"],
            market=str(raw["market"]),
            prompts={name: str(folder / file) for name, file in (raw.get("prompts") or {}).items()},
            segments=_segments(raw["segments"]),
            data=DataConfig(**raw["data"]),
            scoring=ScoringConfig.from_dict(raw["scoring"]),
            roles={role: RoleConfig.from_dict(raw["llm"][role]) for role in ROLES},
            fields=tuple(str(folder / f) for f in raw["fields"]),
            operators=str(folder / raw["operators"]),
            operator_modules=tuple(raw.get("operator_modules") or ()),
            reference=tuple(raw.get("reference") or ()),
            directions=tuple(raw.get("directions") or ()),
            **{name: klass(**raw[name]) for name, klass in sections.items() if name in raw},
            **optional,
        )


def _segments(raw: dict) -> Segments:
    """The ``segments`` section: ``training`` and, optionally, ``validation``, ``holdout`` and ``purge_bars``.

    Raises
    ------
    TypeError
        If the section holds another key.
    """
    unknown = set(raw) - {"training", "validation", "holdout", "purge_bars"}
    if unknown:
        raise TypeError(f"campaign config: unknown segment key(s) {sorted(unknown)}")
    segments = {
        name: Segment(name, str(raw[name]["start"]), str(raw[name]["end"]))
        for name in ("training", "validation", "holdout") if name in raw
    }
    return Segments(**segments, purge_bars=int(raw.get("purge_bars", PURGE_BARS)))
