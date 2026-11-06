"""The campaign config: one YAML file holds every setting of a run.

``data``, ``reference`` and ``known`` hold quantlab components: each is a
``get_config()`` dict inline, the path of a store folder whose
``component.json`` describes it (``quantlab.core.store_folder.save_component``),
or ``{store: <folder>, factor_names: [...]}``, that store's factor pinned to
some of its outputs. ``reference`` lists the reference set: the factors
candidates must be original against (the ``Zoo``) and, in admission,
uncorrelated with and incremental to; a factor's ``factor_names`` selects
its columns. ``known`` lists the known factors: every other existing factor
a candidate is checked against for novelty (the ``Zoo``), never one it is
measured incremental to; an output also in ``reference`` is taken there
only (``CampaignConfig.load_factors``). A malformed entry of either list is
refused when the config is read; a store or class is only opened when the
factors are loaded.

``fields`` (field files, one per input family) and ``operators`` (the
operator file) make the ``Vocabulary`` candidates write with; the
framework has none of its own. ``operator_modules`` lists the modules,
besides KunQuant and ``quantlab.factor``, an operator may come from.

``market`` describes the data in the prompts, ``mechanisms`` lists the
mechanism types candidates belong to (by default ``DEFAULT_MECHANISMS``),
each with a one-line description, and ``prompts`` maps a template's name
(see ``factor_agent.prompts.TEMPLATES``) to the file that replaces it.
"""

import dataclasses

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from quantlab.core.component import rebuild
from quantlab.core.store_folder import load_component

from factor_agent import _acceptance_model
from factor_agent.gate.vocabulary import Vocabulary, VocabularyError
from factor_agent.llm.base import backend_options, refuse_secrets
from factor_agent.prompts import Templates
from factor_agent.segments import PURGE_BARS, Segment, Segments
from factor_agent.snapshot import SearchFiles
from factor_agent.zoo import owner



@dataclass(frozen=True)
class DataConfig:
    """The data the Evaluator reads.

    ``prices`` is the whole-market price dataset, ``label`` the forward
    label (masked by day-t membership when there is a membership) and
    ``membership`` the members of each day (an ``IndexConstituentDataset``)
    or None, when every symbol is scored. ``intraday`` is the whole-market
    intraday dataset an intraday family's fields read (one-minute trade
    bars, say), or None without an intraday family.
    """

    prices: dict | str
    label: dict | str
    membership: dict | str | None = None
    intraday: dict | str | None = None

    def load(self, name: str) -> Any:
        """Rebuild the component ``name``; a list of components is rebuilt item by item."""
        return load_each(getattr(self, name))


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


def load_each(value: dict | str | list) -> Any:
    """Rebuild a component (see ``load``), or each of a list of components."""
    return [load(item) for item in value] if isinstance(value, list) else load(value)


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

    #: Seconds the sandboxed graph build may take, counted from when the sandbox is ready.
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
    #: Power of the trial count the anchor's p-value is divided by: 1 is Bonferroni, a smaller
    #: one lets the threshold rise more slowly (0.5 counts 100 trials as 10).
    trial_exponent: float = 1.0
    #: Newey-West t of the partial IC, in the factor's direction, a candidate needs on the
    #: confirmation segment (``segments.confirmation``), whose statistics no prompt shows; read
    #: only when there is one.
    confirm_t: float = 2.0

    def __post_init__(self):
        if isinstance(self.confirm_t, bool) or not isinstance(self.confirm_t, (int, float)):
            raise TypeError(f"admission: confirm_t is {self.confirm_t!r}; it is a number")
        if not 0 < self.trial_exponent <= 1:
            raise ValueError(f"admission: trial_exponent is {self.trial_exponent}; it lies in (0, 1]")


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
    #: Rounds between two distillations of the ledger into the memory.
    memory_every: int = 5
    #: Seed of the Thompson sampling; round n draws from ``(seed, n)``.
    seed: int = 0
    #: Proposal groups a round's directions are split into, each written, fixed and reviewed by
    #: calls of its own that run side by side; a resume may change it.
    groups: int = 1
    #: Trials built on one trial (its Lineage's attempts) at which it is taken off the planner's
    #: parents, so a direction naming it is planned as a new one; None (the default) caps none.
    #: It needs ``context.feedback``, and a resume may not change it.
    max_children: int | None = None
    #: Slots each round whose parent the code chooses (Picked parents, ``_picked_parents``), each a
    #: slot of its parent's mechanism type that the planner writes as a refinement or a mutation of
    #: it; 0 (the default) picks none. It needs ``context.theme_correlation``, and a resume may not
    #: change it.
    picked_parents: int = 0

    def __post_init__(self):
        if self.groups < 1:
            raise ValueError(f"round: groups is {self.groups}; a round needs at least one proposal group")
        if self.max_children is not None and (
            isinstance(self.max_children, bool) or not isinstance(self.max_children, int) or self.max_children < 1
        ):
            raise ValueError(f"round: max_children is {self.max_children!r}; it is a count of at least 1")
        if isinstance(self.picked_parents, bool) or not isinstance(self.picked_parents, int) or self.picked_parents < 0:
            raise ValueError(f"round: picked_parents is {self.picked_parents!r}; it is a count of at least 0")


@dataclass(frozen=True)
class ContextConfig:
    """What the prompt context holds; the run snapshot records it at the first start, and a resume may not change it."""

    #: Rounds whose trials the planner reads one line each (the recent history); older trials are
    #: counted by mechanism type and outcome (the earlier summary).
    recent_rounds: int = 5
    #: Whether the planner and the proposer read the earlier trials (the memory, the history, the
    #: last round and the accepted factors) and a memory is distilled. Off, each round is planned
    #: and written from the data and the seed directions alone, every direction new: the control
    #: of a test of whether the search learns from its trials.
    feedback: bool = True
    #: The correlation, in absolute value, at which an evaluated trial is grouped with an earlier one
    #: (``_feedback.theme_groups``), the groups the planner reads; None (the default) groups none.
    theme_correlation: float | None = None
    #: Whether a new direction's candidate as correlated with an earlier trial is sent back to be
    #: rewritten once before it is evaluated; it needs ``theme_correlation``.
    send_back_repeats: bool = False
    #: Whether the planner's accepted factors and the proposer's parent block carry each trial's
    #: Lineage (``_lineage``): how the trials built on it did; it needs ``feedback``.
    lineage: bool = False

    def __post_init__(self):
        if self.recent_rounds < 1:
            raise ValueError(f"context: recent_rounds is {self.recent_rounds}; at least the last round is listed")
        if not isinstance(self.feedback, bool):
            raise TypeError(f"context: feedback is {self.feedback!r}; it is true or false")
        if self.theme_correlation is not None:
            if isinstance(self.theme_correlation, bool) or not isinstance(self.theme_correlation, (int, float)):
                raise TypeError(f"context: theme_correlation is {self.theme_correlation!r}; it is a number or null")
            if not 0 < self.theme_correlation < 1:
                raise ValueError(f"context: theme_correlation is {self.theme_correlation}; it lies in (0, 1)")
            if not self.feedback:
                raise ValueError("context: theme_correlation needs feedback, since the groups are made of earlier trials")
        if not isinstance(self.send_back_repeats, bool):
            raise TypeError(f"context: send_back_repeats is {self.send_back_repeats!r}; it is true or false")
        if self.send_back_repeats and self.theme_correlation is None:
            raise ValueError("context: send_back_repeats needs theme_correlation, the correlation at which a trial repeats one")
        if not isinstance(self.lineage, bool):
            raise TypeError(f"context: lineage is {self.lineage!r}; it is true or false")
        if self.lineage and not self.feedback:
            raise ValueError("context: lineage needs feedback, since a Lineage is made of earlier trials")


#: The values of ``acceptance.contribution``.
CONTRIBUTIONS = ("none", "drop_one", "add_one", "both")


@dataclass(frozen=True)
class AcceptanceConfig:
    """How ``Acceptance`` validates, opens the hold-out and exports.

    ``model`` is the acceptance model, a template (``factor_agent._acceptance_model``):
    the component config of a quantlab ``BaseModel`` or ``ModelEnsemble``,
    whose own factors are the baseline (empty: the reference set). Every
    baseline feature, and each accepted factor's scored values, is read as
    the campaign's ``scoring`` reads.

    Raises
    ------
    ValueError
        If there is no seed, ``model`` is a ``SeedEnsemble``, or ``contribution`` is not one of ``CONTRIBUTIONS``.
    TypeError
        If ``model`` is not a model's component config.
    """

    #: The acceptance model's template; required, so no campaign is accepted against a default model.
    model: dict
    #: The seeds every model is trained with; each replaces every member's ``random_seed``.
    seeds: tuple[int, ...] = (0, 1, 2)
    #: Share of the training window each model validates (early-stops) on, at its end.
    val_size: float = 0.2
    #: Members held by the plain TopK book.
    top_k: int = 100
    #: Each accepted factor's contribution in validation: none, drop_one, add_one or both; never on the hold-out.
    contribution: str = "none"
    #: Where exported stores go, ``<factor_root>/<name>/``; an export needs it.
    factor_root: str | None = None
    #: The package the exported ``FactorKunQuant`` subclasses are written into, one module each.
    package: str = "factor_agent.accepted"
    #: First date of an exported store; None: the training segment's start.
    export_start: str | None = None
    #: Last date of an exported store; None: the hold-out's end, else the validation's.
    export_end: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if not self.seeds:
            raise ValueError("acceptance: at least one seed is needed")
        if self.contribution not in CONTRIBUTIONS:
            raise ValueError(f"acceptance.contribution {self.contribution!r} is not one of {list(CONTRIBUTIONS)}")
        _acceptance_model.check(self.model)

    @classmethod
    def from_dict(cls, raw: dict | None) -> "AcceptanceConfig":
        """Read the ``acceptance`` section.

        Raises
        ------
        KeyError
            If the section, or its ``model``, is missing.
        TypeError
            If it holds ``hyperparameters`` or ``features``, which the model template now holds.
        """
        raw = dict(raw or {})
        for removed, now in (("hyperparameters", "hyperparameters"), ("features", "factors")):
            if removed in raw:
                raise TypeError(f"campaign config: acceptance.{removed} is gone; give them as acceptance.model's {now}")
        if "model" not in raw:
            raise KeyError(
                "campaign config: acceptance.model is needed, the component config of the quantlab model "
                "(or ModelEnsemble) the accepted factors are tested in"
            )
        return cls(**raw)


@dataclass(frozen=True)
class RoleConfig:
    """One role's LLM: the backend name, the model and the backend's own settings."""

    backend: str
    model: str
    options: dict

    @classmethod
    def from_dict(cls, data: dict, role: str = "role") -> "RoleConfig":
        """Split ``role``'s YAML mapping into the backend, the model and the rest.

        Raises
        ------
        ValueError
            If the backend name is unknown, or the rest holds an option that looks like a key (a key is read
            only from the environment variable ``api_key_env`` names).
        TypeError
            If the rest holds an option the backend does not take, listing those it does.
        """
        data = dict(data)
        backend, model = data.pop("backend"), data.pop("model")
        try:
            options = backend_options(backend)
        except ValueError as error:
            raise ValueError(f"campaign config: llm.{role}: {error}") from None
        try:
            refuse_secrets(backend, data)
        except ValueError as error:
            raise ValueError(f"campaign config: llm.{role}: {error}") from None
        unknown = sorted(set(data) - set(options))
        if unknown:
            raise TypeError(
                f"campaign config: llm.{role}: the {backend} backend does not take {unknown}; "
                f"it takes {', '.join(sorted(options))}"
            )
        return cls(backend=backend, model=model, options=data)

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
    acceptance: AcceptanceConfig
    #: The data as the prompts describe it, such as "adjusted daily bars of every US common stock".
    market: str
    #: Modules besides KunQuant and ``quantlab.factor`` an operator's class or callable may come from.
    operator_modules: tuple = ()
    #: The reference factors, as components; see the module docstring.
    reference: tuple = ()
    #: The known factors besides the reference set, as components; see the module docstring.
    known: tuple = ()
    directions: tuple = ()
    evaluator: EvaluatorConfig = EvaluatorConfig()
    gate: GateConfig = GateConfig()
    admission: AdmissionConfig = AdmissionConfig()
    round: RoundConfig = RoundConfig()
    context: ContextConfig = ContextConfig()
    mechanisms: tuple = DEFAULT_MECHANISMS
    prompts: dict = field(default_factory=dict)
    #: The YAML text ``from_file`` read, which a run keeps for each opening; None for a config made in code.
    source: str | None = field(default=None, compare=False, repr=False)

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
        if self.round.max_children is not None and not self.context.feedback:
            raise ValueError("round: max_children needs feedback, since without it every direction is new")
        if self.round.picked_parents and self.context.theme_correlation is None:
            raise ValueError(
                "round: picked_parents needs context.theme_correlation, since the Picked parents are ranked "
                "on the evaluated trials' correlations"
            )

    def load_factors(self) -> tuple[list, list]:
        """Rebuild the reference factors and the known factors, ``(reference, known)``, each in config order.

        An output of a known factor is dropped when a reference factor shows
        the same output of the same class (through ``zoo.owner``, so a
        neutralised store counts as the factor it neutralises); the factor
        keeps its other outputs, and one left with none is dropped.

        Raises
        ------
        ValueError, TypeError
            As ``load`` does, for an entry its store or class does not fit.
        """
        reference = [load(component) for component in self.reference]
        taken = {key for factor in reference for key in _output_keys(factor)}
        known = []
        for factor in (load(component) for component in self.known):
            names = tuple(name for name, key in zip(factor.get_factor_names(), _output_keys(factor)) if key not in taken)
            if names == tuple(factor.get_factor_names()):
                known.append(factor)
            elif names:
                known.append(type(factor)(dataclasses.replace(factor.config, factor_names=names)))
        return reference, known

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

    def templates(self, files: SearchFiles | None = None, picked_parents: int | None = None) -> Templates:
        """The prompt templates: the built-in ones, those ``files`` replace read from their files.

        ``files`` are the config's own by default, or a run snapshot's copies.
        With picked parents (``round.picked_parents`` by default) a replaced
        plan template must list the picked slots (``{picked}``).

        Raises
        ------
        PromptError
            With every problem of the replacing files.
        """
        picked = self.round.picked_parents if picked_parents is None else picked_parents
        return Templates.load((files or self.search_files()).prompts, picked=picked > 0)

    def vocabulary(self, columns=None, files: SearchFiles | None = None, intraday_columns=None) -> Vocabulary:
        """The Vocabulary of the field and operator files; ``columns``, given, are the price dataset's.

        ``files`` are the config's own by default, or a run snapshot's copies;
        ``intraday_columns``, given, are the intraday dataset's, the session
        inputs included.

        Raises
        ------
        VocabularyError
            With every problem of the files, a field column missing from ``columns`` or
            ``intraday_columns`` included, and an intraday family without ``data.intraday``.
        """
        files = files or self.search_files()
        vocabulary = Vocabulary.load(files.fields, files.operators, self.operator_modules, columns, intraday_columns)
        if vocabulary.intraday_family is not None and self.data.intraday is None:
            raise VocabularyError([
                f"{vocabulary.intraday_family} is an intraday family, but the config names no data.intraday, "
                f"the dataset of its bars"
            ])
        return vocabulary

    @classmethod
    def from_file(cls, path: str | Path) -> "CampaignConfig":
        """Read a campaign config from YAML; the paths of ``fields``, ``operators`` and ``prompts`` are relative to its folder.

        Raises
        ------
        TypeError
            If a section holds a key its config class does not have, a segment's entry another key than
            ``start`` and ``end``, or a role an option its backend does not take (the message lists those
            it does), or if the config names a budget or per-token prices.
        KeyError
            If a required section or role is missing.
        ValueError
            If a section's values do not fit together, or a role names an unknown backend.
        PromptError
            Naming every problem of the replaced templates.
        """
        source = Path(path).read_text()
        raw = yaml.safe_load(source)
        folder = Path(path).parent
        sections = {
            "evaluator": EvaluatorConfig, "gate": GateConfig,
            "admission": AdmissionConfig, "round": RoundConfig, "context": ContextConfig,
        }
        unknown = set(raw) - {
            "run_dir", "segments", "data", "scoring", "fields", "operators", "operator_modules", "reference", "known",
            "directions", "llm", "market", "mechanisms", "prompts", "acceptance", *sections,
        }
        if unknown:
            raise TypeError(f"campaign config: unknown section(s) {sorted(unknown)}")
        unknown_roles = set(raw["llm"]) - set(ROLES)
        if unknown_roles:
            raise TypeError(f"campaign config: unknown LLM role(s) {sorted(unknown_roles)}; known: {list(ROLES)}")
        _refuse_money(raw)
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
            roles={role: RoleConfig.from_dict(raw["llm"][role], role) for role in ROLES},
            fields=tuple(str(folder / f) for f in raw["fields"]),
            operators=str(folder / raw["operators"]),
            operator_modules=tuple(raw.get("operator_modules") or ()),
            reference=_entries("reference", raw.get("reference")),
            known=_entries("known", raw.get("known")),
            directions=tuple(raw.get("directions") or ()),
            acceptance=AcceptanceConfig.from_dict(raw.get("acceptance")),
            source=source,
            **{name: klass(**raw[name]) for name, klass in sections.items() if name in raw},
            **optional,
        )


def _refuse_money(raw: dict) -> None:
    """Refuse a budget or per-token prices: spending limits belong to the provider.

    Raises
    ------
    TypeError
        Naming the first such setting.
    """
    if "budget_usd" in (raw.get("round") or {}):
        raise TypeError("campaign config: round.budget_usd is gone; the provider's balance or account limit caps the spending")
    for role, spec in raw["llm"].items():
        for price in ("input_usd_per_mtok", "output_usd_per_mtok"):
            if price in (spec or {}):
                raise TypeError(f"campaign config: llm.{role}.{price} is gone; the library no longer counts the cost")


def _output_keys(factor) -> list[tuple[str, str]]:
    """``(class of the owner, output name)`` of each output: what makes two outputs the same factor."""
    kind = type(owner(factor)).__name__
    return [(kind, name) for name in factor.get_factor_names()]


def _entries(section: str, raw: list | None) -> tuple:
    """The component entries of ``section`` (``reference`` or ``known``), each checked for its form only.

    Raises
    ------
    TypeError
        Naming every entry that is not a store folder path, a store entry
        (``store`` and optionally a list of ``factor_names``) or a
        ``get_config()`` dict (one with a ``name``).
    """
    entries = tuple(raw or ())
    problems = []
    for i, entry in enumerate(entries):
        if isinstance(entry, str):
            problem = None if entry else "an empty store folder path"
        elif not isinstance(entry, dict):
            problem = f"{entry!r} is neither a store folder, a store entry nor a component's config"
        elif "store" in entry:
            names = entry.get("factor_names", [])
            if set(entry) - {"store", "factor_names"}:
                problem = f"a store entry with the unknown key(s) {sorted(set(entry) - {'store', 'factor_names'})}"
            elif not isinstance(entry["store"], str) or not entry["store"]:
                problem = f"a store entry whose store {entry['store']!r} is not a folder path"
            elif not isinstance(names, list) or not all(isinstance(name, str) for name in names):
                problem = f"a store entry whose factor_names {names!r} is not a list of output names"
            else:
                problem = None
        else:
            problem = None if isinstance(entry.get("name"), str) else "a component's config without the name of its class"
        if problem:
            problems.append(f"{section}[{i}]: {problem}")
    if problems:
        raise TypeError("campaign config: " + "; ".join(problems))
    return entries


def _segments(raw: dict) -> Segments:
    """The ``segments`` section: ``training`` and, optionally, ``confirmation``, ``validation``, ``holdout`` and ``purge_bars``.

    Raises
    ------
    TypeError
        If the section holds another key, or a segment's entry another key than ``start`` and ``end``.
    """
    unknown = set(raw) - {"training", "confirmation", "validation", "holdout", "purge_bars"}
    if unknown:
        raise TypeError(f"campaign config: unknown segment key(s) {sorted(unknown)}")
    for name in ("training", "confirmation", "validation", "holdout"):
        unknown = set(raw.get(name) or {}) - {"start", "end"}
        if unknown:
            raise TypeError(f"campaign config: segments.{name}: unknown key(s) {sorted(unknown)}; it takes start and end")
    segments = {
        name: Segment(name, str(raw[name]["start"]), str(raw[name]["end"]))
        for name in ("training", "confirmation", "validation", "holdout") if name in raw
    }
    return Segments(**segments, purge_bars=int(raw.get("purge_bars", PURGE_BARS)))
