"""The Campaign: the research loop, from one config to accepted and rejected candidates.

One round:

1. the quota: every mechanism type gets ``min_per_type`` slots and the rest
   go by Thompson sampling on each type's admission rate (``_scheduler``);
2. the planner plans one direction per slot: a new mechanism, an
   orthogonal mutation of an accepted factor, or a refinement of one;
3. the proposer writes one candidate per direction; the Gate checks each
   against the reference set, the known factors and the accepted factors
   (the ``Zoo``) and the refused ones go back to the proposer to be fixed;
4. the reviewer, in its own call, judges each gated candidate's hypothesis,
   description and code for consistency, with the data section's fields;
   the inconsistent ones go back to the proposer once, all in one fix
   request with the reviewer's reasons; each rewrite passes the Gate once
   (a refusal fails it) and the reviewer judges the rewrites in one more
   call: one still inconsistent is rejected without being evaluated; with
   ``round.groups`` above 1, steps 3 and 4 make these calls per proposal
   group, the groups' calls of each stage side by side (``Researcher``);
5. the Evaluator scores the rest on the training segment in one compile,
   admission decides (``_admission``) in proposal order, and an accepted
   candidate joins the zoo and the reference pool before the next is admitted;
   a refusal for coverage says at which step of the graph the values are lost
   (``Evaluator.trace_coverage``);
6. the round's records go to the ledger and the state to ``state.json``.

Before anything else ``run`` checks the replaced prompt templates for
their placeholders and builds the ``Vocabulary``, checking every field's
column against the price dataset; every problem is reported at once,
before any LLM call and before anything is written, so a run refused for
its files starts cleanly once they are fixed. It then opens the run's
``RunSnapshot``: a first start copies the config's field, operator and
template files into the run directory, a resume fails if they changed
since and warns if the commits did; every commit is listed in
``state.json``. The run searches with the snapshot's copies. ``check``
makes these checks and the others a start makes before its first LLM call
(the data and the stores it reads, the roles' backends, the reference and
known factors, the snapshot's ``differences``) part by part, writing
nothing, and reports every problem at once in a ``preflight.CheckReport``. A replay (roles on the ``replay`` backend) searches with the
snapshot of the run whose LLM log it replays, whatever the config's files
say now.

The prompts (``factor_agent.prompts.Prompts``) describe the config's market, the
label's horizon (its ``span_bars()``), the scoring, the vocabulary and the
config's mechanism types, through the snapshot's templates.

The planner and the proposer read the context of ``_feedback``, a pure
function of the ledger and the config: the seed directions, the reference
factor families and the known ones (marked as known), the memory, the last
round's feedback and the current best mechanisms; the planner also reads the recent history (the trials of the
last ``context.recent_rounds`` rounds, one line each) and the earlier
summary (the older trials counted by mechanism type and outcome). The
snapshot records ``recent_rounds`` at the first start and a resume may not
change it. Every ``memory_every`` rounds the distiller folds the trials
since the last memory into it; an empty or unparsable reply keeps the
previous memory. ``memory.md`` holds the current memory, which every later
prompt carries, and ``memory/round_NN.md`` every version, by the round it
was written before. Every trial is named by its id (``_feedback.trial_id``);
a direction may build on any evaluated trial by its id, and the proposer
is given that trial (``_feedback.parent``). With ``context.feedback`` off
the planner and the proposer read only the seed directions and the factor
families, every direction is new and no memory is distilled. With
``context.theme_correlation`` every evaluated trial joins the Evaluator's
tried trials and the planner also reads them grouped by what their factors
measure (``_feedback.themes``); a resume computes the tried trials again
from the ledger. With ``context.lineage`` each round computes every trial's
Lineage (``_lineage``) once, and the planner's accepted factors and the
proposer's parent block carry it; ``round.max_children`` takes every trial
built on that often off the trials a direction may build on. With
``round.picked_parents`` (which needs the tried trials) each round, after
the quota, builds the Trial graph (``_trial_graph``) and gives each trial in
its pick order a slot of its mechanism type while that type has one unclaimed
(``_claim``); the planner writes those slots as refinements or mutations of
their parents, and the plan event records them. With
``context.send_back_repeats`` too, before a round is scored, a new
direction's candidate that correlates that much with an earlier trial is
rewritten once (``_renew``); a slot is rewritten once in
all, so a candidate the reviewer sent back is not checked again.

Before each round the stop conditions are checked: the round limit, the
trial limit and ``patience`` rounds in a row without an acceptance; the
first that holds ends the run and is recorded. The library counts no
money: spending limits belong to the provider. When the provider refuses a
call because the balance or quota is exhausted (``ProviderQuotaExhausted``),
the round in progress is dropped, the stop is recorded as
``provider_quota`` with the provider's message, and the run returns; a
resume, once topped up, runs that round again. Every call goes
to the run's LLM log (``llm.log``), which the ``replay`` backend serves
back to reproduce the run without a provider; on a resume the calls of the
dropped round are moved to ``llm_log/dropped/``, so the log holds exactly
the calls the ledger was written from. ``run`` resumes
a run directory from its state: the rounds written are kept, a round a
crash left half written is dropped and run again, and the zoo, the pool and
the graphs the Gate compares duplicates with are restored from the ledger.
The reference set and the known factors are read from the config on every
start, a resume included; neither is snapshotted.

As it goes, the campaign appends what it does to the run's events file
(``factor_agent.events``): every start and resume, plan, LLM call, Gate
check, verdict, evaluation, admission, round end and stop. It never reads
the file, so a resume, a replay and the snapshot do not depend on it.
"""

import json
import math
import os
import socket
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import NamedTuple

from quantlab.core.store_uses import READS, walk_store_uses

from factor_agent import _admission, _feedback, _lineage, _scheduler, _trial_graph
from factor_agent._lineage import Lineage
from factor_agent.config import ROLES, AdmissionConfig, CampaignConfig, ContextConfig, TrialGraphConfig
from factor_agent.evaluator import CompileFailure, DailyIC, Evaluator, Metrics
from factor_agent.events import (
    FILE as EVENTS_FILE, AdmissionOutcome, EvaluationStart, Events, Plan, Resume, RoundEnd, Start, Stopped,
)
from factor_agent.gate import Gate
from factor_agent.gate._originality import render
from factor_agent.gate.graph import Graph
from factor_agent.gate.vocabulary import Vocabulary
from factor_agent.ledger import Ledger, TrialRecord
from factor_agent.llm.base import AskingAgain, LLMBackend, ProviderQuotaExhausted, make_backend
from factor_agent.llm.log import LLMLog, RecordingBackend
from factor_agent.preflight import Check, CheckReport, ping
from factor_agent.prompts import Prompts, Templates
from factor_agent.researcher import Proposal, Researcher
from factor_agent.scoring import Scoring
from factor_agent.segments import code_version
from factor_agent.session import SESSION_INPUTS
from factor_agent.snapshot import RunSnapshot, SearchFiles, SnapshotMismatch
from factor_agent.zoo import Zoo


@dataclass(frozen=True)
class Stop:
    """Why the campaign stopped: ``by`` is ``"max_rounds"``, ``"max_trials"``, ``"patience"`` or ``"provider_quota"``."""

    by: str
    reason: str


@dataclass(frozen=True)
class CampaignResult:
    """Every trial record of the run, why it stopped, and where the ledger is.

    ``evaluated_trials`` counts the candidates the Evaluator scored.
    ``not_covered`` names the reference and known factors without a KunQuant graph,
    which the Gate's originality check cannot compare a candidate with.
    """

    records: list[TrialRecord]
    ledger_path: Path
    stop: Stop
    rounds: int
    memory: str = ""
    not_covered: tuple[str, ...] = ()

    @property
    def accepted(self) -> list[TrialRecord]:
        return [r for r in self.records if r.status == "accepted"]

    @property
    def rejected(self) -> list[TrialRecord]:
        return [r for r in self.records if r.status == "rejected"]

    @property
    def failed(self) -> list[TrialRecord]:
        return [r for r in self.records if r.status == "failed"]

    @property
    def evaluated_trials(self) -> int:
        """The number of evaluated candidates."""
        return sum(r.evaluated for r in self.records)


@dataclass
class _State:
    """What ``state.json`` holds after every round; the ledger holds the trials."""

    rounds: int = 0
    memory: str = ""
    #: The round after which the distiller was last asked.
    memory_round: int = 0
    #: The last round whose trials the memory holds: the next distillation reads the trials after it.
    memory_through: int = 0
    #: Calls in the LLM log when the state was last written at a round's end or after a distillation.
    calls: int = 0
    log: list[dict] = field(default_factory=list)
    stop: dict | None = None
    #: The code versions of every start and resume, as the run's snapshot records them.
    versions: list[dict] = field(default_factory=list)

    @classmethod
    def read(cls, path: Path) -> "_State":
        return cls(**json.loads(path.read_text())) if path.exists() else cls()

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2) + "\n")
        tmp.replace(path)


class Campaign:
    """Run a mining campaign from its config.

    Parameters
    ----------
    config : CampaignConfig
        Every setting of the run.
    backends : dict, optional
        Backends by role that replace the ones the config names (the model
        still comes from the config), such as a test's fake.
    """

    def __init__(self, config: CampaignConfig, backends: dict[str, LLMBackend] | None = None):
        self.config = config
        self._backends = dict(backends or {})
        self._replaced = set(self._backends)

    def _prompts(self, templates: Templates, vocabulary: Vocabulary, label, scoring: Scoring) -> Prompts:
        """What the prompts say: the config's market and mechanism types, ``label``'s horizon, ``scoring``'s sentence."""
        return Prompts(
            templates=templates, market=self.config.market, horizon=label.span_bars(), scoring=scoring.describe(),
            vocabulary=vocabulary, mechanisms={m.name: m.description for m in self.config.mechanisms},
        )

    def _role_backends(self) -> dict[str, LLMBackend]:
        """Each role's backend: the one passed in, or the one its config names, built once."""
        for role in ROLES:
            if role not in self._backends:
                self._backends[role] = make_backend(self.config.roles[role].spec())
        return self._backends

    def run(self) -> CampaignResult:
        """Run, or resume, the campaign until a stop condition holds; return every candidate's outcome.

        Raises
        ------
        PromptError
            Before anything is written, naming every placeholder a replaced template lacks or may not use.
        SnapshotMismatch
            Before any LLM call, on a resume whose field, operator or template files differ from the run's snapshot.
        VocabularyError
            Before any LLM call, with every problem of the field and operator files.

        Warns
        -----
        CodeVersionChanged
            On a resume on other commits than the run was last opened with.
        """
        config, run_dir = self.config, Path(self.config.run_dir)
        state_path = run_dir / "state.json"
        prices, intraday, columns, intraday_columns, files, settings = self._load(run_dir, state_path)
        snapshot = RunSnapshot.open(run_dir, files, code_version(), settings)
        parent_settings = _ParentSettings.of(snapshot.settings)
        _keep_config(config, run_dir, opening=len(snapshot.versions))
        recent_rounds = snapshot.settings["context"]["recent_rounds"]
        vocabulary = config.vocabulary(columns, snapshot.search_files, intraday_columns)
        label = config.data.load("label")
        scoring = Scoring.from_config(config.scoring, config.data, njobs=config.evaluator.njobs)
        prompts = self._prompts(
            config.templates(snapshot.search_files, parent_settings.picked_parents), vocabulary, label, scoring,
        )
        state = _State.read(state_path)
        state.versions = snapshot.versions
        ledger = Ledger(run_dir / "ledger.jsonl")
        log = LLMLog(run_dir / "llm_log")
        written = _written(ledger, log)
        ledger.keep(state.rounds)
        records = ledger.records()
        reference, known = config.load_factors()
        log.drop_after(state.calls)
        events = Events(run_dir / EVENTS_FILE)
        dropped = _written(ledger, log) != written
        events.write(self._opened(snapshot, state, dropped))
        researcher = Researcher(
            {
                role: (RecordingBackend(AskingAgain(backend), log), config.roles[role].model)
                for role, backend in self._role_backends().items()
            },
            prompts,
            groups=config.round.groups,
            events=events,
        )
        store_key = _store_key(config)
        evaluator = Evaluator.for_segment(
            config.training,
            prices=prices,
            label=label,
            scoring=scoring,
            reference=reference,
            known=known,
            warmup_bars=config.evaluator.warmup_bars,
            njobs=config.evaluator.njobs,
            purge_bars=config.segments.purge_bars,
            vocabulary=vocabulary,
            intraday=intraday,
            store=run_dir / "columns" / config.training.name,
            store_key=store_key,
        )
        confirming = None if config.segments.confirmation is None else Evaluator.for_segment(
            config.segments.confirmation,
            prices=prices,
            label=label,
            scoring=scoring,
            reference=reference,
            warmup_bars=config.evaluator.warmup_bars,
            njobs=config.evaluator.njobs,
            purge_bars=config.segments.purge_bars,
            vocabulary=vocabulary,
            intraday=intraday,
            store=run_dir / "columns" / config.segments.confirmation.name,
            store_key=store_key,
        )
        zoo = Zoo.from_existing(reference, vocabulary, known=known)
        _write_zoo(run_dir / "zoo.json", zoo)
        gate = Gate(config.gate, vocabulary, seen=[Graph.from_json(r.graph, vocabulary.classes) for r in records if r.graph])
        _restore(records, zoo, evaluator, confirming, vocabulary, themes=config.context.theme_correlation is not None)
        while (stop := self._stop(state, records)) is None:
            number = state.rounds + 1
            try:
                if config.context.feedback and _feedback.distill_due(
                    state.rounds, state.memory_round, config.round.memory_every,
                ):
                    since = [(t, r) for t, r in _feedback.named(records) if r.round > state.memory_through]
                    memory = researcher.distill(
                        number, state.memory, _feedback.history(since, statistics=False), _feedback.accepted(records),
                    )
                    state.memory_round = state.rounds
                    if memory is None:
                        print(
                            f"Campaign: the distiller's reply before round {number} is empty or not a memory in "
                            f"its three parts; the previous memory is kept", flush=True,
                        )
                    else:
                        state.memory, state.memory_through = memory, state.rounds
                        (run_dir / "memory").mkdir(parents=True, exist_ok=True)
                        (run_dir / "memory" / f"round_{number:02d}.md").write_text(memory + "\n")
                        (run_dir / "memory.md").write_text(memory + "\n")
                    state.calls = len(log.files())
                    state.write(state_path)
                quota = _scheduler.quota(number, records, config.round, prompts.mechanism_types)
                accepted = [r for r in records if r.status == "accepted"]
                trials = _feedback.named(records)
                shown, cap, k = parent_settings.lineage, parent_settings.max_children, parent_settings.picked_parents
                built = _lineage.lineages(
                    trials, config.admission, parent_settings.fertile_children,
                ) if shown or cap is not None or k else {}
                lineage = built if shown else {}
                context = self._context(
                    records, state, reference, known, evaluator, accepted, recent_rounds, prompts, lineage,
                )
                parents = {
                    trial: _feedback.parent(trial, r, trials, lineage.get(trial)) for trial, r in trials if r.evaluated
                } if config.context.feedback else {}
                eligible = [trial for trial in parents if cap is None or built[trial].attempts < cap]
                picked = _claim(k, quota, dict(trials), _trial_graph.build(
                    trials, *evaluator.tried(), built, vocabulary.classes, parent_settings.trial_graph, cap,
                )) if k else []
                directions, issues = researcher.plan(
                    number, quota, context, eligible if config.context.feedback else None,
                    [(p["mechanism_type"], p["trial"]) for p in picked],
                )
                events.write(Plan(
                    round=number, directions=[d.to_dict() for d in directions], quota=quota, issues=issues, picked=picked,
                ))
                proposals = researcher.propose(number, directions, context, gate, zoo, config.round.retries, parents)
                round_records = self._round(
                    number, proposals, researcher, gate, zoo, evaluator, records, events, confirming, vocabulary,
                )
            except ProviderQuotaExhausted as exc:
                stop = Stop("provider_quota", f"{exc}; round {number} was stopped and dropped")
                break
            ledger.append(round_records)
            records += round_records
            state.rounds, state.calls = number, len(log.files())
            accepted_now = sum(r.status == "accepted" for r in round_records)
            state.log.append({"round": number, "quota": quota, "plan_issues": issues, "accepted": accepted_now})
            state.write(state_path)
            events.write(RoundEnd(round=number, trials=len(round_records), accepted=accepted_now))
        state.stop = asdict(stop)
        state.write(state_path)
        events.write(Stopped(by=stop.by, reason=stop.reason))
        return CampaignResult(
            records=records, ledger_path=ledger.path, stop=stop, rounds=state.rounds,
            memory=state.memory, not_covered=zoo.not_covered,
        )

    def check(self, ping: bool = False) -> CheckReport:
        """What ``run`` would refuse before its first LLM call, found without writing anything or computing a factor.

        Each part is checked on its own and every problem is reported at
        once: each data component (it loads, the stores it reads exist and
        the price dataset's columns can be read), the scoring, each role's
        backend (it can be built: an unset API key variable is a problem;
        a backend passed in is not checked), the reference and known
        factors, the run snapshot (on a resume, files or settings that
        differ from it are a problem, other code versions a warning), the
        vocabulary and the templates. The vocabulary needs the price and
        intraday datasets' columns and the templates need the vocabulary,
        the label and the scoring; each is skipped, with the reason, when
        a part it needs failed. The run directory is only read.

        With ``ping``, each distinct backend (roles with the same config
        share one) is then sent one short request outside the LLM log, and
        its error is a problem; roles on the ``replay`` or ``scripted``
        backend, with a backend passed in or whose backend cannot be built
        are not pinged, and say why.

        Returns
        -------
        CheckReport
            One ``Check`` per part, in that order; the run can start when its ``problems`` is empty.
        """
        config, run_dir = self.config, Path(self.config.run_dir)
        checks: list[Check] = []

        def attempt(name: str, work: Callable[[], object]) -> object:
            """``work()``, or ``_FAILED`` with its problems recorded as ``name``'s."""
            try:
                value = work()
            except Exception as exc:  # every problem is reported, none raised
                checks.append(Check(name, problems=_problems(exc)))
                return _FAILED
            checks.append(Check(name))
            return value

        def data(name: str) -> object:
            """The data component ``name``, checked to read only stores that exist."""
            return _reading(config.data.load(name))

        def scored() -> Scoring:
            """The scoring, its neutralisation's exposures checked to read only stores that exist."""
            scoring = Scoring.from_config(config.scoring, config.data, njobs=config.evaluator.njobs)
            if scoring.neutralization is not None:
                _reading(scoring.neutralization.exposures)
            return scoring

        def with_columns(name: str, columns: Callable) -> tuple:
            """The data component ``name`` (``data``) and the ``columns`` read from it."""
            component = data(name)
            return component, columns(component)

        loaded = attempt("data.prices", lambda: with_columns("prices", _columns))
        columns = _FAILED if loaded is _FAILED else loaded[1]
        label = attempt("data.label", lambda: data("label"))
        if config.data.membership is not None:
            attempt("data.membership", lambda: data("membership"))
        intraday_columns = None
        if config.data.intraday is not None:
            loaded = attempt("data.intraday", lambda: with_columns("intraday", _intraday_columns))
            intraday_columns = _FAILED if loaded is _FAILED else loaded[1]
        scoring = attempt("scoring", scored)
        built = {}
        for role in ROLES:
            if role in self._replaced:
                checks.append(Check(f"llm.{role}"))
            else:
                built[role] = attempt(f"llm.{role}", lambda role=role: make_backend(config.roles[role].spec()))
        attempt("reference and known factors", config.load_factors)
        searched = self._check_snapshot(run_dir, checks)
        files = _FAILED if searched is _FAILED else searched[0]

        needs = {"data.prices": columns, "data.intraday": intraday_columns, "run snapshot": files}
        if lacking := [name for name, value in needs.items() if value is _FAILED]:
            vocabulary = _FAILED
            checks.append(Check("vocabulary", skipped=f"it needs {_parts(lacking)}, which failed"))
        else:
            vocabulary = attempt("vocabulary", lambda: config.vocabulary(columns, files, intraday_columns))
        needs = {"vocabulary": vocabulary, "data.label": label, "scoring": scoring}
        if lacking := [name for name, value in needs.items() if value is _FAILED]:
            checks.append(Check("templates", skipped=f"they need {_parts(lacking)}, which failed or did not run"))
        else:
            picked = _ParentSettings.of(searched[1]).picked_parents
            attempt("templates", lambda: self._prompts(config.templates(files, picked), vocabulary, label, scoring))
        if ping:
            checks += self._ping(built)
        return CheckReport(tuple(checks))

    def _ping(self, built: dict) -> list[Check]:
        """A ``ping`` of each distinct backend among ``built`` (each role's backend, or ``_FAILED``), in role order."""
        shared: dict[str, list[str]] = {}
        for role in ROLES:
            shared.setdefault(json.dumps(self.config.roles[role].spec(), sort_keys=True), []).append(role)
        checks = []
        for roles in shared.values():
            first, name = self.config.roles[roles[0]], "ping " + ", ".join(roles)
            if first.backend in ("replay", "scripted"):
                checks.append(Check(name, skipped=f"the {first.backend} backend is not pinged"))
            elif replaced := [role for role in roles if role in self._replaced]:
                checks.append(Check(name, skipped=f"the backend of {', '.join(replaced)} was passed in"))
            elif built[roles[0]] is _FAILED:
                checks.append(Check(name, skipped=f"it needs llm.{roles[0]}, which failed"))
            else:
                checks.append(ping(name, built[roles[0]], roles[0], first.model))
        return checks

    def _check_snapshot(self, run_dir: Path, checks: list[Check]) -> tuple[SearchFiles, dict] | object:
        """Check the files and settings the run would open its snapshot with; return them (``_search``), or ``_FAILED``.

        A first start has nothing to compare; a resume's differences from the
        snapshot are problems and other code versions a warning. A file, or
        the snapshot itself, that cannot be read is a problem, but the files
        are still returned for the vocabulary to report it in its terms.
        """
        try:
            files, settings = self._search(run_dir, run_dir / "state.json")
        except Exception as exc:  # every problem is reported, none raised
            checks.append(Check("run snapshot", problems=_problems(exc)))
            return _FAILED
        if not RunSnapshot.exists(run_dir):
            checks.append(Check("run snapshot"))
            return files, settings
        try:
            snapshot = RunSnapshot.read(run_dir)
            changed = snapshot.version_change(code_version())
        except Exception as exc:  # a snapshot written in part or damaged
            problem = f"{run_dir / 'snapshot.json'} cannot be read ({type(exc).__name__}: {exc}); the run cannot resume"
            checks.append(Check("run snapshot", problems=(problem,)))
            return files, settings
        try:
            problems = tuple(snapshot.differences(files, settings))
        except Exception as exc:  # a file that cannot be read
            problems = _problems(exc)
        checks.append(Check("run snapshot", problems=problems, warnings=(changed,) if changed else ()))
        return files, settings

    def _load(self, run_dir: Path, state_path: Path) -> "_Loaded":
        """The loading half of a start, which writes nothing.

        The price and intraday datasets, their columns (``_columns``,
        ``_intraday_columns``), and the files and settings the run's
        snapshot is opened with (``_search``). The replaced templates and the
        vocabulary (against the price dataset's columns and the intraday
        dataset's) are checked here, so broken files are never snapshotted
        and a run refused for them starts cleanly once they are fixed.

        Raises
        ------
        PromptError
            If a replaced template lacks a placeholder its role needs or uses one it may not.
        VocabularyError
            With every problem of the field and operator files.
        SnapshotMismatch, ValueError
            As ``_search`` does.
        """
        prices = self.config.data.load("prices")
        intraday = self.config.data.load("intraday") if self.config.data.intraday is not None else None
        columns, intraday_columns = _columns(prices), _intraday_columns(intraday)
        files, settings = self._search(run_dir, state_path)
        self.config.templates(files, _ParentSettings.of(settings).picked_parents)
        self.config.vocabulary(columns, files, intraday_columns)
        return _Loaded(prices, intraday, columns, intraday_columns, files, settings)

    def _search(self, run_dir: Path, state_path: Path) -> tuple[SearchFiles, dict]:
        """The files and settings to open the run's snapshot with: the config's, or, in a replay, the replayed run's.

        The settings recorded are the prompt context, ``round.max_children``
        and ``round.picked_parents`` when set, with Picked parents every
        ``trial_graph`` value and, with an intraday dataset, that dataset,
        which the intraday family's fields mean.

        Raises
        ------
        SnapshotMismatch
            If the run directory holds a state but no snapshot (a run started before snapshots).
        ValueError
            If the roles replay more than one recorded run.
        """
        if state_path.exists() and not RunSnapshot.exists(run_dir):
            raise SnapshotMismatch(
                f"campaign: {run_dir} holds a run but no snapshot of its files, so a resume cannot be checked; "
                f"start a new run directory"
            )
        replayed = {
            Path(self.config.roles[role].options["log"]).parent for role in ROLES
            if role not in self._replaced and self.config.roles[role].backend == "replay"
        }
        replayed = {path for path in replayed if RunSnapshot.exists(path)}
        if len(replayed) > 1:
            raise ValueError(f"campaign: the roles replay several runs {sorted(map(str, replayed))}; replay one")
        if replayed:
            recorded = RunSnapshot.read(replayed.pop())
            return recorded.search_files, recorded.settings
        context = asdict(self.config.context)
        if context["feedback"]:
            del context["feedback"]  # as runs started before the setting recorded their context
        if context["theme_correlation"] is None:
            del context["theme_correlation"]  # likewise
        if not context["send_back_repeats"]:
            del context["send_back_repeats"]  # likewise
        if not context["lineage"]:
            del context["lineage"]  # likewise
        if context["fertile_children"] == ContextConfig.fertile_children:
            del context["fertile_children"]  # likewise
        settings = {"context": context}
        cap, k = self.config.round.max_children, self.config.round.picked_parents
        rounds = ({"max_children": cap} if cap is not None else {}) | ({"picked_parents": k} if k else {})
        if rounds:
            settings["round"] = rounds  # each left out unset, likewise
        if k:
            settings["trial_graph"] = asdict(self.config.trial_graph)  # every value, so no change of a default moves it
        if self.config.data.intraday is not None:
            settings["intraday"] = {"dataset": json.loads(json.dumps(self.config.data.intraday))}
        return self.config.search_files(), settings

    def _opened(self, snapshot: RunSnapshot, state: _State, dropped: bool) -> Start:
        """The events file's record of this opening: a start, or a resume naming the round it ``dropped``."""
        settings, admission = self.config.round, self.config.admission
        opened = dict(
            pid=os.getpid(), host=socket.gethostname(), versions=snapshot.versions[-1], rounds=state.rounds,
            candidates=settings.candidates, max_rounds=settings.max_rounds, max_trials=settings.max_trials,
            patience=settings.patience, t_threshold=admission.t_threshold, max_correlation=admission.max_correlation,
            min_coverage=admission.min_coverage, min_monotonicity=admission.min_monotonicity,
            trial_exponent=admission.trial_exponent,
        )
        if len(snapshot.versions) == 1:
            return Start(**opened)
        return Resume(**opened, dropped=state.rounds + 1 if dropped else None)

    def _stop(self, state: _State, records: list[TrialRecord]) -> Stop | None:
        """The first stop condition that holds before the next round, or None."""
        settings = self.config.round
        if state.rounds >= settings.max_rounds:
            return Stop("max_rounds", f"the round limit of {settings.max_rounds} rounds is reached")
        if len(records) + settings.candidates > settings.max_trials:
            return Stop(
                "max_trials",
                f"another round of {settings.candidates} candidates would pass the trial limit of "
                f"{settings.max_trials} ({len(records)} trials so far)",
            )
        recent = range(state.rounds - settings.patience + 1, state.rounds + 1)
        if state.rounds >= settings.patience and not any(r.status == "accepted" and r.round in recent for r in records):
            return Stop("patience", f"no candidate was accepted in the last {settings.patience} rounds")
        return None

    def _context(
        self, records: list[TrialRecord], state: _State, reference: list, known: list, evaluator: Evaluator,
        accepted: list[TrialRecord], recent_rounds: int, prompts: Prompts, lineage: dict[str, Lineage] | None = None,
    ) -> dict[str, str]:
        """The sections the planner and the proposer read; see ``_feedback``.

        The trials of the last ``recent_rounds`` rounds are the recent
        history, one line each; the older ones are counted in the earlier
        summary, by mechanism type in the order of ``prompts``. Each accepted
        factor carries its ``lineage`` (by trial id) when given. With
        ``context.feedback`` off only the seed directions and the families.
        """
        given = {"seed": _seed(self.config.directions), "families": _feedback.families(reference, known)}
        if not self.config.context.feedback:
            return given
        labels = [_feedback.accepted_label(r) for r in accepted]
        first_recent = _feedback.first_recent_round(state.rounds, recent_rounds)
        trials = _feedback.named(records)
        return {
            **given,
            "memory": state.memory,
            "earlier_summary": _feedback.earlier_summary(
                [r for r in records if r.round < first_recent], prompts.mechanism_types,
            ),
            "recent_history": _feedback.history(
                [(t, r) for t, r in trials if r.round >= first_recent], earlier=trials,
            ),
            "last_round": _feedback.last_round([(t, r) for t, r in trials if r.round == state.rounds], trials),
            "best": _feedback.best(trials, evaluator.contributions(labels) if labels else {}, lineage),
            **({"themes": _feedback.themes(
                trials, *evaluator.tried(), self.config.context.theme_correlation, self.config.context.send_back_repeats,
            )} if self.config.context.theme_correlation is not None else {}),
        }

    def _round(
        self, number: int, proposals: list[Proposal], researcher: Researcher, gate: Gate, zoo: Zoo, evaluator: Evaluator,
        earlier: list[TrialRecord], events: Events, confirming: Evaluator | None = None,
        vocabulary: Vocabulary | None = None,
    ) -> list[TrialRecord]:
        """Review, evaluate and admit one round's proposals; return its records in proposal order.

        A candidate the reviewer finds inconsistent is rewritten once
        (``Researcher.revise``); its record is that of the rewrite, failed if
        the parser or the Gate refuses it and rejected if the reviewer still
        finds it inconsistent. The candidates the reviewer finds consistent are tested together, so
        each is judged at the evaluated-trial count after the round: the
        earlier rounds' plus every candidate of the round that ran, whatever
        its place in the proposal order. They are admitted in that order,
        and the accepted ones are added to ``zoo``. With a ``confirming``
        evaluator (of the confirmation segment) every candidate is evaluated
        there first, against the pool before the round, and one accepted
        joins its pool after the round. The evaluation's start
        and each outcome go to ``events``, and each evaluated candidate's
        daily IC to ``ic/round_NN.json`` (``_keep_daily_ic``).
        """
        evaluated = sum(r.evaluated for r in earlier)
        records = [self._failed(number, p) if p.checked is None else None for p in proposals]
        gated = [i for i, p in enumerate(proposals) if p.checked is not None]
        verdicts = researcher.review(number, [proposals[i] for i in gated])
        passed = [i for i, verdict in zip(gated, verdicts) if verdict.consistent]
        sent_back = [(i, verdict) for i, verdict in zip(gated, verdicts) if not verdict.consistent]
        rewrites = dict(zip(
            [i for i, _ in sent_back],
            researcher.revise(number, [(proposals[i], verdict) for i, verdict in sent_back], gate, zoo),
        ))
        if self.config.context.send_back_repeats:
            repeats = self._renew(number, proposals, passed, researcher, gate, zoo, evaluator, earlier)
            passed = [i for i in passed if i not in repeats]
            rewrites |= repeats
        proposals = [rewrites.get(i, proposal) for i, proposal in enumerate(proposals)]
        judged, refused = self._judge_rewrites(number, rewrites, researcher)
        passed = sorted(passed + judged)
        for i, record in refused.items():
            records[i] = record

        daily = {}

        def admit(position: int, metrics: Metrics, ran: int) -> str | None:
            proposal = proposals[passed[position]]
            daily[passed[position]] = metrics.daily
            record = self._admit(
                number, proposal, metrics, evaluated + ran, confirmations[position] if confirmations else None,
                lambda: _coverage_cause(evaluator, proposal.checked.graph, gate.vocabulary, self.config.admission),
            )
            records[passed[position]] = record
            events.write(AdmissionOutcome(
                round=number, slot=passed[position], status=record.status, stage=record.stage,
                reasons=list(record.reasons), threshold=_admission.t_threshold(self.config.admission, evaluated + ran),
                metrics=record.metrics,
            ))
            if record.status != "accepted":
                return None
            return zoo.add(proposal.candidate.description, _feedback.accepted_source(number), proposal.checked.graph).label

        events.write(EvaluationStart(round=number, slots=list(passed)))
        graphs = [proposals[i].checked.graph for i in passed]
        confirmations = confirming.evaluate(graphs) if confirming is not None and graphs else None
        tried = [_feedback.trial_id(number, i) for i in passed] if self.config.context.theme_correlation is not None else None
        results = evaluator.evaluate(graphs, admit, tried)
        if confirming is not None:
            _join(confirming, [r for r in records if r is not None and r.status == "accepted"], vocabulary)
        _keep_daily_ic(Path(self.config.run_dir), number, daily)
        for i, result in zip(passed, results):
            if isinstance(result, CompileFailure):
                records[i] = TrialRecord(
                    status="failed", stage="evaluate", reasons=[result.error], **self._common(number, proposals[i]),
                )
                events.write(AdmissionOutcome(
                    round=number, slot=i, status="failed", stage="evaluate", reasons=[result.error], threshold=None,
                    metrics=None,
                ))
        return records

    def _renew(
        self, number: int, proposals: list[Proposal], passed: list[int], researcher: Researcher, gate: Gate, zoo: Zoo,
        evaluator: Evaluator, earlier: list[TrialRecord],
    ) -> dict[int, Proposal]:
        """The rewrites, by slot, of the new directions' candidates in ``passed`` that measure what an earlier trial measured.

        Before anything is scored, a new direction's candidate whose factor
        is correlated with an earlier evaluated trial's at
        ``context.theme_correlation`` or more in absolute value
        (``Evaluator.nearest_tried``, which computes each new direction's
        candidate once more) is rewritten once (``Researcher.renew``), with
        that trial named and no statistic but the correlation, and a rewrite
        the parser or the Gate refuses is fixed up to ``round.retries`` times;
        the rewrites are judged as the reviewer's are (``_judge_rewrites``).
        """
        threshold = self.config.context.theme_correlation
        new = [i for i in passed if proposals[i].direction.action == "new"]
        nearest = evaluator.nearest_tried([proposals[i].checked.graph for i in new])
        by_id = dict(_feedback.named(earlier))
        repeated = [
            (i, name, correlation) for i, found in zip(new, nearest) if not isinstance(found, CompileFailure)
            for name, correlation in [found] if name and abs(correlation) >= threshold
        ]
        rewrites = researcher.renew(
            number, [(proposals[i], _repeats(name, correlation, by_id[name], threshold)) for i, name, correlation in repeated],
            gate, zoo, self.config.round.retries,
        )
        return {i: rewrite for (i, _, _), rewrite in zip(repeated, rewrites)}

    def _judge_rewrites(
        self, number: int, rewrites: dict[int, Proposal], researcher: Researcher,
    ) -> tuple[list[int], dict[int, TrialRecord]]:
        """Of the rewrites, by slot, the slots to evaluate and the records of the others.

        A rewrite the parser or the Gate refused is failed; the reviewer
        judges the others in one more call, and an inconsistent one is
        rejected at review; a consistent one is evaluated with the rest.
        """
        refused = {i: self._failed(number, p) for i, p in rewrites.items() if p.checked is None}
        checked = [i for i, p in rewrites.items() if p.checked is not None]
        judged = []
        for i, verdict in zip(checked, researcher.review(number, [rewrites[i] for i in checked])):
            if verdict.consistent:
                judged.append(i)
            else:
                refused[i] = TrialRecord(
                    status="rejected", stage="review", reasons=[verdict.refusal], **self._common(number, rewrites[i]),
                )
        return judged, refused

    @staticmethod
    def _failed(number: int, proposal: Proposal) -> TrialRecord:
        """The record of a candidate the parser or the Gate still refused after its retries."""
        last = proposal.attempts[-1]
        item = proposal.item
        return TrialRecord(
            round=number, candidate=item if isinstance(item, dict) else {"raw": item}, status="failed",
            stage=last.stage, reasons=list(last.reasons), attempts=_attempts(proposal),
            direction=proposal.direction.to_dict(),
        )

    def _admit(
        self, number: int, proposal: Proposal, metrics: Metrics, trials: int,
        confirmation: Metrics | CompileFailure | None = None, coverage_cause: Callable[[], str] | None = None,
    ) -> TrialRecord:
        """Decide one evaluated candidate, the ``trials``-th evaluated trial of the campaign.

        With its ``confirmation`` (its evaluation on the confirmation
        segment) a candidate passing every other check is refused unless it
        holds there (``_admission.confirmed``); a confirmation that did not
        compile does not hold. A refusal for coverage goes on with what
        ``coverage_cause`` says of where the candidate loses its values.
        """
        admission = self.config.admission
        reasons = _admission.reasons(metrics, trials, admission)
        if reasons and coverage_cause is not None and _admission.reason_kind(reasons[0]) == "coverage":
            reasons = [reasons[0] + coverage_cause()]
        kept = confirmation.to_dict() if isinstance(confirmation, Metrics) else None
        if not reasons and confirmation is not None and (
            kept is None or not _admission.confirmed(metrics.ic_mean, confirmation.partial_ic_t, admission)
        ):
            reasons = [_admission.NOT_CONFIRMED]
        return TrialRecord(
            status="rejected" if reasons else "accepted", stage="admission", reasons=reasons,
            metrics=metrics.to_dict(), confirmation=kept, **self._common(number, proposal),
        )

    @staticmethod
    def _common(number: int, proposal: Proposal) -> dict:
        """The record fields of a candidate the Gate passed."""
        checked = proposal.checked
        return dict(
            round=number, candidate=proposal.candidate.to_dict(), fingerprint=checked.fingerprint,
            stats=checked.stats, attempts=_attempts(proposal), direction=proposal.direction.to_dict(),
            graph=checked.graph.to_json(),
        )


class _Loaded(NamedTuple):
    """What ``Campaign._load`` read: the datasets, their columns, and the files and settings of the snapshot."""

    prices: object
    intraday: object
    columns: object
    intraday_columns: object
    files: SearchFiles
    settings: dict


class _ParentSettings(NamedTuple):
    """The Lineage, ``round.max_children``, Picked-parent and Trial graph settings a run uses.

    They are read from the settings the run's snapshot is opened with
    (``Campaign._search``): the config's on a new run, the recorded run's in
    a replay, so a replay builds on parents as the recorded run did and
    ``check`` judges the templates as ``run`` does.
    """

    lineage: bool
    max_children: int | None
    picked_parents: int
    fertile_children: int
    trial_graph: TrialGraphConfig

    @classmethod
    def of(cls, settings: dict) -> "_ParentSettings":
        """The settings ``settings`` (a snapshot's) record, each left out when it was off or its default."""
        context, rounds = settings.get("context", {}), settings.get("round", {})
        return cls(
            bool(context.get("lineage", False)), rounds.get("max_children"), rounds.get("picked_parents", 0),
            context.get("fertile_children", ContextConfig.fertile_children),
            TrialGraphConfig(**settings.get("trial_graph", {})),
        )


def _claim(k: int, quota: dict[str, int], trials: dict[str, TrialRecord], graph: _trial_graph.TrialGraph) -> list[dict]:
    """The round's picked slots: the Trial graph's pick order, each trial given a slot of its mechanism type.

    A trial takes one slot of its type (the type of the slot it filled)
    while ``quota`` leaves that type one unclaimed, until ``k`` are claimed.
    Each is the ``Pick`` as a dict with the type.
    """
    claimed, left = [], dict(quota)
    for pick in graph.pick_order():
        record = trials[pick.trial]
        kind = (record.direction or record.candidate).get("mechanism_type")
        if left.get(kind, 0) > 0:
            left[kind] -= 1
            claimed.append({"trial": pick.trial, "mechanism_type": kind, **{
                name: getattr(pick, name) for name in ("quality", "fertility", "novelty", "front")
            }})
            if len(claimed) == k:
                break
    return claimed


#: What a part ``Campaign.check`` could not check gives in place of its value.
_FAILED = object()


class _Problems(Exception):
    """Several problems of one part of a check."""

    def __init__(self, problems: list[str]):
        self.problems = tuple(problems)
        super().__init__("; ".join(problems))


def _problems(exc: Exception) -> tuple[str, ...]:
    """The problems an exception reports: each of a ``VocabularyError``'s or a ``PromptError``'s, or its message."""
    problems = getattr(exc, "problems", None)
    if problems:
        return tuple(problems)
    return (str(exc) or type(exc).__name__,)


def _reading(component):
    """``component`` (or a list of components), checked to read only stores that exist.

    Raises
    ------
    _Problems
        Naming every store it reads that does not exist.
    """
    components = component if isinstance(component, list) else [component]
    missing = [
        use.path for item in components for use in walk_store_uses(item)
        if use.relation == READS and use.path is not None and not Path(use.path).exists()
    ]
    if missing:
        raise _Problems([f"the store {path} it reads does not exist" for path in dict.fromkeys(missing)])
    return component


def _parts(names: list[str]) -> str:
    """``names`` joined for a sentence."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _coverage_cause(evaluator: Evaluator, graph: Graph, vocabulary: Vocabulary, admission: AdmissionConfig) -> str:
    """Where ``graph`` loses its values on ``evaluator``'s segment, continuing its coverage refusal (``_admission.coverage_cause``)."""
    daily, trace = evaluator.trace_coverage(graph)
    return _admission.coverage_cause(trace, lambda node: render(daily, node, vocabulary), admission.min_coverage)


def _keep_daily_ic(run_dir: Path, number: int, daily: dict[int, DailyIC | None]) -> None:
    """Keep round ``number``'s daily IC and partial IC by slot in ``ic/round_NN.json``, apart from the ledger.

    The days are the label's, shared by the round's candidates; NaN (an unscored day) is written as null.
    A round run again after a resume writes its file anew.
    """
    kept = {slot: series for slot, series in daily.items() if series is not None}
    if not kept:
        return
    (run_dir / "ic").mkdir(exist_ok=True)
    (run_dir / "ic" / f"round_{number:02d}.json").write_text(json.dumps({
        "days": list(next(iter(kept.values())).days),
        "slots": {
            str(slot): {"ic": [v if math.isfinite(v) else None for v in s.ic],
                        "partial_ic": [v if math.isfinite(v) else None for v in s.partial_ic]}
            for slot, s in sorted(kept.items())
        },
    }))


def _keep_config(config: CampaignConfig, run_dir: Path, opening: int) -> None:
    """Keep the config text of the run's ``opening``-th start or resume as ``config/open-<opening>.yaml``.

    A resume may raise a limit, so every opening keeps its own; a config made in code has no text to keep.
    """
    if config.source is not None:
        (run_dir / "config").mkdir(exist_ok=True)
        (run_dir / "config" / f"open-{opening}.yaml").write_text(config.source)


def _written(ledger: Ledger, log: LLMLog) -> tuple[str, int]:
    """What the run has written to its ledger and its LLM log: the ledger's text and the number of call files."""
    return ledger.path.read_text() if ledger.path.exists() else "", len(log.files())


def _seed(directions) -> str:
    """The seed directions as the planner reads them."""
    return "\n".join(f"- {direction}" for direction in directions)


def _restore(
    records: list[TrialRecord], zoo: Zoo, evaluator: Evaluator, confirming: Evaluator | None, vocabulary: Vocabulary,
    themes: bool,
) -> None:
    """Put the accepted factors of the rounds already written back into ``zoo`` and each segment's pool, in order.

    With ``themes`` the evaluated trials go back into the tried trials too.
    The Evaluators read the columns from the run's store (``Evaluator.join``),
    so a factor is computed only when the store lacks it: a run directory
    written before the store, say, whose columns are stored now.

    Raises
    ------
    RuntimeError
        If an accepted factor's graph no longer compiles or an evaluated
        trial's no longer computes (the code or the quantlab version changed
        under the run directory).
    """
    accepted = [r for r in records if r.status == "accepted"]
    graphs = [Graph.from_json(r.graph, vocabulary.classes) for r in accepted]
    labels = [zoo.add(r.candidate["description"], _feedback.accepted_source(r.round), g).label for r, g in zip(accepted, graphs)]
    tried = [
        (trial, Graph.from_json(r.graph, vocabulary.classes)) for trial, r in _feedback.named(records) if r.evaluated
    ] if themes else []
    broken = evaluator.join(pool=list(zip(labels, graphs)), tried=tried)
    if lost := [name for name in broken if name in labels]:
        raise RuntimeError(f"Campaign: cannot resume, accepted factor(s) no longer compile: {lost}")
    if broken:
        raise RuntimeError(f"Campaign: cannot resume, evaluated trial(s) no longer compute: {broken}")
    if confirming is not None:
        _join(confirming, accepted, vocabulary)


def _repeats(trial: str, correlation: float, earlier: TrialRecord, threshold: float) -> str:
    """Why a new direction's candidate is sent back: it measures what the earlier ``trial`` measured."""
    return (
        f"the research direction is new, but the factor measures what earlier trial {trial} measured: its "
        f"correlation with {trial}'s factor is {correlation:+.2f}, at least {threshold:g} in absolute value "
        f"({trial} computed: {earlier.candidate.get('description', '')}; {_feedback.outcome(earlier)}); write the "
        f"candidate for a mechanism no earlier trial measured, as the research direction asks"
    )


def _join(evaluator: Evaluator, accepted: list[TrialRecord], vocabulary: Vocabulary) -> None:
    """Add the ``accepted`` factors to ``evaluator``'s pool, in order: the confirmation segment's pool.

    A factor of the batch the evaluator computed last (the round's
    candidates) or of its store is not computed again (``Evaluator.join``).

    Raises
    ------
    RuntimeError
        If an accepted factor's graph does not compile there.
    """
    labels = [_feedback.accepted_label(r) for r in accepted]
    graphs = [Graph.from_json(r.graph, vocabulary.classes) for r in accepted]
    broken = evaluator.join(pool=list(zip(labels, graphs)))
    if broken:
        raise RuntimeError(f"Campaign: accepted factor(s) do not compile on the confirmation segment: {broken}")


def _store_key(config: CampaignConfig) -> dict:
    """What an Evaluator's stored columns depend on besides its segment: the data, the scoring and the warm-up."""
    return json.loads(json.dumps(
        {"data": asdict(config.data), "scoring": asdict(config.scoring), "warmup_bars": config.evaluator.warmup_bars},
        default=str,
    ))


def _columns(prices) -> set[str] | None:
    """The variables of the price dataset's panel; None when it reads no store (a panel held in memory).

    The names are the panel's, as candidate code reads them: a merged or renaming input (a prefix, a
    shared name) gives its columns under the names its panel holds, not its store's.
    """
    if not any(Path(path).exists() for path in prices.stores()):
        return None
    return set(prices.head(0).collect_schema().names()) - {"timestamp", "symbol"}


def _intraday_columns(intraday) -> set[str] | None:
    """The intraday dataset's columns and the session inputs added to them; None without a dataset or a store."""
    if intraday is None:
        return None
    columns = _columns(intraday)
    return None if columns is None else columns | set(SESSION_INPUTS)


def _write_zoo(path: Path, zoo: Zoo) -> None:
    """Write which reference and known factors the originality check covers and which it cannot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"covered": list(zoo.covered), "not_covered": list(zoo.not_covered)}, indent=2) + "\n")


def _attempts(proposal: Proposal) -> list[dict]:
    """The refused versions of a proposal as the ledger stores them."""
    return [{**asdict(a), "reasons": list(a.reasons)} for a in proposal.attempts]
