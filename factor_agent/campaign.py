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
   call: one still inconsistent is rejected without being evaluated;
5. the Evaluator scores the rest on the training segment in one compile,
   admission decides (``_admission``) in proposal order, and an accepted
   candidate joins the zoo and the reference pool before the next is admitted;
6. the round's records go to the ledger and the state to ``state.json``.

Before anything else ``run`` checks the replaced prompt templates for
their placeholders and builds the ``Vocabulary``, checking every field's
column against the price dataset; every problem is reported at once,
before any LLM call and before anything is written, so a run refused for
its files starts cleanly once they are fixed. It then opens the run's
``RunSnapshot``: a first start copies the config's field, operator and
template files into the run directory, a resume fails if they changed
since and warns if the commits did; every commit is listed in
``state.json``. The run searches with the snapshot's copies. A replay (roles on the ``replay`` backend) searches with the
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
was written before.

Before each round the stop conditions are checked: the round limit, the
trial limit and ``patience`` rounds in a row without an acceptance; the
first that holds ends the run and is recorded. The library counts no
money: spending limits belong to the provider. Every call goes
to the run's LLM log (``llm.log``), which the ``replay`` backend serves
back to reproduce the run without a provider; on a resume the calls of the
dropped round are moved to ``llm_log/dropped/``, so the log holds exactly
the calls the ledger was written from. ``run`` resumes
a run directory from its state: the rounds written are kept, a round a
crash left half written is dropped and run again, and the zoo, the pool and
the graphs the Gate compares duplicates with are restored from the ledger.
The reference set and the known factors are read from the config on every
start, a resume included; neither is snapshotted.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import xarray as xr

from factor_agent import _admission, _feedback, _scheduler
from factor_agent.config import ROLES, CampaignConfig
from factor_agent.evaluator import CompileFailure, Evaluator, Metrics
from factor_agent.gate import Gate
from factor_agent.gate.graph import Graph
from factor_agent.gate.vocabulary import Vocabulary
from factor_agent.ledger import Ledger, TrialRecord
from factor_agent.llm.base import LLMBackend, make_backend
from factor_agent.llm.log import LLMLog, RecordingBackend
from factor_agent.prompts import Prompts, Templates
from factor_agent.researcher import Proposal, Researcher
from factor_agent.scoring import Scoring
from factor_agent.segments import code_version
from factor_agent.snapshot import RunSnapshot, SnapshotMismatch
from factor_agent.zoo import Zoo


@dataclass(frozen=True)
class Stop:
    """Why the campaign stopped: ``by`` is ``"max_rounds"``, ``"max_trials"`` or ``"patience"``."""

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
        prices = config.data.load("prices")
        snapshot = self._snapshot(run_dir, state_path, _columns(prices))
        recent_rounds = snapshot.settings["context"]["recent_rounds"]
        vocabulary = config.vocabulary(_columns(prices), snapshot.search_files)
        label = config.data.load("label")
        scoring = Scoring.from_config(config.scoring, config.data, njobs=config.evaluator.njobs)
        prompts = self._prompts(config.templates(snapshot.search_files), vocabulary, label, scoring)
        state = _State.read(state_path)
        state.versions = snapshot.versions
        ledger = Ledger(run_dir / "ledger.jsonl")
        ledger.keep(state.rounds)
        records = ledger.records()
        reference, known = config.load_factors()
        log = LLMLog(run_dir / "llm_log")
        log.drop_after(state.calls)
        researcher = Researcher(
            {
                role: (RecordingBackend(backend, log), config.roles[role].model)
                for role, backend in self._role_backends().items()
            },
            prompts,
        )
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
            period_lags=vocabulary.lag_columns,
        )
        zoo = Zoo.from_existing(reference, vocabulary, known=known)
        _write_zoo(run_dir / "zoo.json", zoo)
        gate = Gate(config.gate, vocabulary, seen=[Graph.from_json(r.graph, vocabulary.classes) for r in records if r.graph])
        _restore(records, zoo, evaluator, vocabulary)
        while (stop := self._stop(state, records)) is None:
            number = state.rounds + 1
            if _feedback.distill_due(state.rounds, state.memory_round, config.round.memory_every):
                since = [r for r in records if r.round > state.memory_through]
                memory = researcher.distill(number, state.memory, _feedback.history(since, statistics=False))
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
            context = self._context(records, state, reference, known, evaluator, accepted, recent_rounds, prompts)
            directions, issues = researcher.plan(number, quota, context, [_feedback.accepted_label(r) for r in accepted])
            proposals = researcher.propose(number, directions, context, gate, zoo, config.round.retries)
            round_records = self._round(number, proposals, researcher, gate, zoo, evaluator, records)
            ledger.append(round_records)
            records += round_records
            state.rounds, state.calls = number, len(log.files())
            state.log.append({
                "round": number, "quota": quota, "plan_issues": issues,
                "accepted": sum(r.status == "accepted" for r in round_records),
            })
            state.write(state_path)
        state.stop = asdict(stop)
        state.write(state_path)
        return CampaignResult(
            records=records, ledger_path=ledger.path, stop=stop, rounds=state.rounds,
            memory=state.memory, not_covered=zoo.not_covered,
        )

    def _snapshot(self, run_dir: Path, state_path: Path, columns: list[str]) -> RunSnapshot:
        """Open the run's snapshot: of the config's files, or, in a replay, of the replayed run's snapshot.

        The replaced templates and the vocabulary (against the price
        dataset's ``columns``) are checked first, so broken files are never
        snapshotted and a run refused for them starts cleanly once they are fixed.

        Raises
        ------
        PromptError
            If a replaced template lacks a placeholder its role needs or uses one it may not.
        VocabularyError
            With every problem of the field and operator files.
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
            files, settings = recorded.search_files, recorded.settings
        else:
            files, settings = self.config.search_files(), {"context": asdict(self.config.context)}
        self.config.templates(files)
        self.config.vocabulary(columns, files)
        return RunSnapshot.open(run_dir, files, code_version(), settings)

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
        accepted: list[TrialRecord], recent_rounds: int, prompts: Prompts,
    ) -> dict[str, str]:
        """The sections the planner and the proposer read; see ``_feedback``.

        The trials of the last ``recent_rounds`` rounds are the recent
        history, one line each; the older ones are counted in the earlier
        summary, by mechanism type in the order of ``prompts``.
        """
        labels = [_feedback.accepted_label(r) for r in accepted]
        first_recent = _feedback.first_recent_round(state.rounds, recent_rounds)
        return {
            "seed": _seed(self.config.directions),
            "families": _feedback.families(reference, known),
            "memory": state.memory,
            "earlier_summary": _feedback.earlier_summary(
                [r for r in records if r.round < first_recent], prompts.mechanism_types,
            ),
            "recent_history": _feedback.history([r for r in records if r.round >= first_recent]),
            "last_round": _feedback.last_round([r for r in records if r.round == state.rounds]),
            "best": _feedback.best(accepted, evaluator.contributions(labels) if labels else {}),
        }

    def _round(
        self, number: int, proposals: list[Proposal], researcher: Researcher, gate: Gate, zoo: Zoo, evaluator: Evaluator,
        earlier: list[TrialRecord],
    ) -> list[TrialRecord]:
        """Review, evaluate and admit one round's proposals; return its records in proposal order.

        A candidate the reviewer finds inconsistent is rewritten once
        (``Researcher.revise``); its record is that of the rewrite, failed if
        the parser or the Gate refuses it and rejected if the reviewer still
        finds it inconsistent. The candidates the reviewer finds consistent are tested together, so
        each is judged at the evaluated-trial count after the round: the
        earlier rounds' plus every candidate of the round that ran, whatever
        its place in the proposal order. They are admitted in that order,
        and the accepted ones are added to ``zoo``.
        """
        evaluated = sum(r.evaluated for r in earlier)
        records = [self._failed(number, p) if p.checked is None else None for p in proposals]
        gated = [i for i, p in enumerate(proposals) if p.checked is not None]
        verdicts = researcher.review(number, [proposals[i] for i in gated])
        passed = [i for i, verdict in zip(gated, verdicts) if verdict.consistent]
        sent_back = [(i, verdict) for i, verdict in zip(gated, verdicts) if not verdict.consistent]
        proposals = list(proposals)
        rewrites = researcher.revise(number, [(proposals[i], verdict) for i, verdict in sent_back], gate, zoo)
        for (i, _), rewrite in zip(sent_back, rewrites):
            proposals[i] = rewrite
            if rewrite.checked is None:
                records[i] = self._failed(number, rewrite)
        rewritten = [i for i, _ in sent_back if proposals[i].checked is not None]
        for i, verdict in zip(rewritten, researcher.review(number, [proposals[i] for i in rewritten])):
            if verdict.consistent:
                passed.append(i)
            else:
                records[i] = TrialRecord(
                    status="rejected", stage="review", reasons=[verdict.refusal],
                    **self._common(number, proposals[i]),
                )
        passed.sort()

        def admit(position: int, metrics: Metrics, ran: int) -> str | None:
            proposal = proposals[passed[position]]
            record = self._admit(number, proposal, metrics, evaluated + ran)
            records[passed[position]] = record
            if record.status != "accepted":
                return None
            return zoo.add(proposal.candidate.description, _feedback.accepted_source(number), proposal.checked.graph).label

        results = evaluator.evaluate([proposals[i].checked.graph for i in passed], admit)
        for i, result in zip(passed, results):
            if isinstance(result, CompileFailure):
                records[i] = TrialRecord(
                    status="failed", stage="evaluate", reasons=[result.error], **self._common(number, proposals[i]),
                )
        return records

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

    def _admit(self, number: int, proposal: Proposal, metrics: Metrics, trials: int) -> TrialRecord:
        """Decide one evaluated candidate, the ``trials``-th evaluated trial of the campaign."""
        reasons = _admission.reasons(metrics, trials, self.config.admission)
        return TrialRecord(
            status="rejected" if reasons else "accepted", stage="admission", reasons=reasons,
            metrics=metrics.to_dict(), **self._common(number, proposal),
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


def _seed(directions) -> str:
    """The seed directions as the planner reads them."""
    return "\n".join(f"- {direction}" for direction in directions)


def _restore(records: list[TrialRecord], zoo: Zoo, evaluator: Evaluator, vocabulary: Vocabulary) -> None:
    """Put the factors accepted in the rounds already written back into ``zoo`` and the reference pool.

    Raises
    ------
    RuntimeError
        If an accepted factor's graph no longer compiles (the code or the
        quantlab version changed under the run directory).
    """
    accepted = [r for r in records if r.status == "accepted"]
    if not accepted:
        return
    graphs = [Graph.from_json(r.graph, vocabulary.classes) for r in accepted]
    labels = [zoo.add(r.candidate["description"], _feedback.accepted_source(r.round), g).label for r, g in zip(accepted, graphs)]
    results = evaluator.evaluate(graphs, lambda position, metrics, ran: labels[position])
    broken = [label for label, result in zip(labels, results) if isinstance(result, CompileFailure)]
    if broken:
        raise RuntimeError(f"Campaign: cannot resume, accepted factor(s) no longer compile: {broken}")


def _columns(prices) -> set[str] | None:
    """The variables of the stores the price dataset reads; None when it reads none (a panel held in memory)."""
    stores = [path for path in prices.stores() if Path(path).exists()]
    if not stores:
        return None
    columns: set[str] = set()
    for path in stores:
        with xr.open_zarr(path) as store:
            columns |= set(store.data_vars)
    return columns


def _write_zoo(path: Path, zoo: Zoo) -> None:
    """Write which reference and known factors the originality check covers and which it cannot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"covered": list(zoo.covered), "not_covered": list(zoo.not_covered)}, indent=2) + "\n")


def _attempts(proposal: Proposal) -> list[dict]:
    """The refused versions of a proposal as the ledger stores them."""
    return [{**asdict(a), "reasons": list(a.reasons)} for a in proposal.attempts]
