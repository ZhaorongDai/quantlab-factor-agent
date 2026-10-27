"""Read a run directory as the frontend shows it, without writing to it.

``list_runs(root)`` finds every run directory under a folder: a directory
holding a ``snapshot.json``, which a campaign writes on its first start,
before its first round. ``RunView.read(run_dir)`` reads one run. Callers
never see the run directory's files; this module alone knows them, and
reads them while a campaign may be writing them:

- The state file decides how many rounds are written. The campaign appends
  a round's trials to the ledger before it rewrites the state, so ledger
  rounds beyond the state's count are not read until the state catches up,
  and a ledger line a crash cut short is skipped.
- NaN and infinite values in the ledger (a correlation with an empty pool,
  say) read as ``None``, so a view converts to strict JSON.
- A resumed run keeps the stop of its previous open in its state file until
  it stops again. A stop counts only when the state file was written after
  the run's latest open and no LLM call ended after it: a resumed run that
  has asked the LLM since, or has written no state since it was opened, has
  not stopped.
- The admission settings are not in the run directory: an accepted factor's
  t threshold is the one its round's refusals quote (one per round, since a
  round's candidates are judged at the same trial count), carried over from
  the round before when the round quotes none, and the correlation limit is
  the one a refusal quotes. Either is ``None`` while no refusal quotes it.
- An Acceptance report is ``acceptance/<segment>[_<digest>].json``; one that
  records opens of the hold-out is a hold-out report. The opens are the hold-out
  lock's. A report written before the contribution table or the selection
  flag existed reads with ``None`` for them: nothing it lacks is made up.
- A collision is read from a refusal's text: the Gate's rewrite, refinement
  of parameters only or shared subtree is a code collision with each factor
  it names, admission's correlation above the limit a ranking collision. The
  Gate's refusal of a duplicate of the run's own earlier candidate is not.
- The diagnosis is read from the trials by rules private to this module,
  their thresholds fixed here, each finding counting trials (not attempts:
  a trial is counted once whichever of its versions met the problem).
- A trial the Evaluator scored is placed on the admission map by its
  partial-IC t in its direction (the sign of its training IC, as admission
  reads it) and its largest absolute correlation with an existing factor.
  The trials it never scored are grouped by the stage that refused them and
  the reason: reasons that differ only in their numbers, or in the reply or
  code they quote, are one reason.
- The memory's versions are the distiller's files ``memory/round_NN.md``,
  each written before round ``NN``, so after round ``NN - 1``. A version
  after the first is compared with the one before it line by line.
"""

import difflib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Stop:
    """Why a run stopped, as the campaign recorded it: ``by`` is the stop condition, ``reason`` its sentence."""

    by: str
    reason: str


@dataclass(frozen=True)
class Collision:
    """A trial's refusal for being too like an existing factor: a reference, known or accepted one.

    ``label`` names the factor as the Gate and the Evaluator do; ``name`` is
    how a page names it, as ``Neighbour.name``. ``kind`` is ``"code"`` (the
    Gate found a rewrite, the same graph with other parameters, or a too
    large shared subtree) or ``"ranking"`` (admission found a correlation
    above the limit). ``accepted_round`` is the round that accepted the
    factor, ``None`` for one of the user's own factors; ``known`` is true
    for a known factor, false for a reference or accepted one.
    """

    label: str
    name: str
    kind: str
    accepted_round: int | None
    known: bool


@dataclass(frozen=True)
class Trial:
    """One trial of a written round, as the ledger holds it, with NaN read as ``None``.

    ``slot`` is the trial's place in its round, from 0, in the order of the
    round's research directions. ``status``, ``stage``, ``reasons``,
    ``candidate``, ``direction``, ``attempts``, ``stats`` and ``metrics`` are
    the trial record's (``factor_agent.ledger.TrialRecord``); ``metrics`` is
    ``None`` for a trial the Evaluator never scored. ``collisions`` holds
    every collision of the trial's versions, each factor and kind once.
    """

    round: int
    slot: int
    status: str
    stage: str
    reasons: tuple[str, ...]
    candidate: dict
    direction: dict | None
    attempts: tuple[dict, ...]
    stats: dict | None
    metrics: dict | None
    collisions: tuple[Collision, ...] = ()


@dataclass(frozen=True)
class Neighbour:
    """The factor closest to an accepted one in ranking, and their correlation.

    ``label`` names it as the Evaluator does. ``name`` is how a page names
    it: a factor accepted earlier in the run as ``"round N: "`` and the first
    words of its description, any other by its label. ``accepted_round`` is
    N for a factor accepted in the run, ``None`` for a reference or known
    factor. ``correlation`` is the raw cross-sectional correlation, which
    admission compares in absolute value with its limit.
    """

    label: str
    name: str
    accepted_round: int | None
    correlation: float | None


@dataclass(frozen=True)
class Factor:
    """An accepted factor, written up with the facts that say how far to trust it.

    ``label`` is the name the Evaluator and Acceptance give it,
    ``"<description> (accepted in round N)"``. ``round``, ``slot`` and
    ``mechanism_type`` place it in the run; ``hypothesis``, ``description``
    and ``code`` are its candidate's. ``factor_direction`` is +1 when a
    higher value went with a higher training return, -1 when with a lower
    one. ``ic_mean``, ``ic_t``, ``partial_ic_mean`` and ``partial_ic_t`` are
    the training IC and the partial IC and their Newey-West t, in the
    factor's direction, as admission judged them; ``metrics`` holds every
    statistic as the ledger records it, the raw values among them.
    ``decile_returns`` and ``coverage`` are the Evaluator's.
    """

    label: str
    round: int
    slot: int
    mechanism_type: str | None
    hypothesis: str
    description: str
    code: str
    factor_direction: int
    ic_mean: float | None
    ic_t: float | None
    partial_ic_mean: float | None
    partial_ic_t: float | None
    threshold: float | None
    monotonicity: float | None
    nearest: Neighbour | None
    nearest_known: Neighbour | None
    decile_returns: tuple[float | None, ...]
    coverage: float | None
    metrics: dict


@dataclass(frozen=True)
class Placed:
    """An evaluated trial as the admission map places it, with its statistics in its direction.

    ``round`` and ``slot`` name the trial, ``status`` is accepted or rejected.
    ``factor_direction`` is +1 when a higher value went with a higher training
    return, -1 otherwise. ``ic_mean``, ``ic_t``, ``partial_ic_mean`` and
    ``partial_ic_t`` are in that direction; the trial's ``metrics`` keep the
    raw values. ``largest_correlation`` is the larger absolute correlation
    with the nearest reference or accepted factor and the nearest known one,
    ``None`` when there was none to compare with. ``threshold`` is the t
    threshold of its round as the refusals quote it (``None`` while none does).
    """

    round: int
    slot: int
    status: str
    factor_direction: int
    ic_mean: float | None
    ic_t: float | None
    partial_ic_mean: float | None
    partial_ic_t: float | None
    largest_correlation: float | None
    threshold: float | None


@dataclass(frozen=True)
class TrialRef:
    """A trial named by its round and its slot in the round."""

    round: int
    slot: int


@dataclass(frozen=True)
class Refusal:
    """The trials one stage refused for the same reason before the Evaluator scored them.

    ``stage`` is ``"parse"``, ``"gate"``, ``"review"`` or ``"evaluate"`` (code
    the Evaluator could not build). ``reason`` is the first trial's final
    reason; the others' may differ in the numbers, reply or code they quote.
    ``rounds`` are the rounds the trials belong to, the earliest first.
    """

    stage: str
    reason: str
    rounds: tuple[int, ...]
    trials: tuple[TrialRef, ...]


@dataclass(frozen=True)
class MemoryLine:
    """One line of a memory version compared with the version before it.

    ``change`` is ``"kept"``, ``"added"`` (not in the version before) or
    ``"removed"`` (in the version before, not in this one).
    """

    change: str
    line: str


@dataclass(frozen=True)
class MemoryVersion:
    """One version of the memory the distiller wrote.

    ``round`` is the round after which it was written. ``changes`` holds the
    lines of this version and of the one before, in order, each marked kept,
    added or removed; it is ``None`` for the run's first version.
    """

    round: int
    text: str
    changes: tuple[MemoryLine, ...] | None


@dataclass(frozen=True)
class Contribution:
    """One accepted factor's contribution to the model in one mode, from an Acceptance report's contribution table.

    ``factor`` is the factor's label. ``mode`` is ``"drop_one"`` (the whole
    selection against the selection without it) or ``"add_one"`` (the
    baseline with only it against the baseline). ``difference`` holds, per
    compared statistic, with the factor minus without it, seed-averaged;
    ``agreeing`` the seeds whose own difference has the sign of that mean.
    """

    factor: str
    mode: str
    difference: dict[str, float | None]
    agreeing: dict[str, int]


@dataclass(frozen=True)
class Comparison:
    """One Acceptance report: the user's model with and without accepted factors, on validation or the hold-out.

    ``name`` is the report's file name without ``.json`` (``validation``, or
    ``validation_<digest>`` for a selection). ``holdout`` is true for a
    report on the hold-out. ``factors`` holds the labels of the factors
    added to the model. ``selection`` is true for a selection of fewer than
    every accepted factor, ``None`` for a report written before the flag.
    ``baseline`` and ``with_accepted`` hold each compared statistic averaged
    over ``seeds``, ``difference`` the second minus the first. ``verdict``
    is ``"improves"`` when the rank IC and the top-k excess return both
    rise with the factors, ``"does_not_improve"`` when neither does,
    ``"mixed"`` otherwise, ``None`` when either is undefined. ``forced`` is
    true for a hold-out opened again past its lock. ``contribution`` is the
    report's contribution table, the largest rank IC difference first:
    empty when no contribution was asked for, ``None`` for a report written
    before the table. ``written`` is the report file's time (UTC).
    """

    name: str
    segment: str
    holdout: bool
    factors: tuple[str, ...]
    selection: bool | None
    seeds: tuple[int, ...]
    top_k: int | None
    train_window: tuple[str, ...]
    test_window: tuple[str, ...]
    baseline: dict[str, float | None]
    with_accepted: dict[str, float | None]
    difference: dict[str, float | None]
    verdict: str | None
    forced: bool
    contribution: tuple[Contribution, ...] | None
    code_version: str
    written: datetime


@dataclass(frozen=True)
class HoldoutOpen:
    """One open of the hold-out, as the hold-out lock records it: ``forced`` for every open after the first."""

    opened_at: str
    code_version: str
    forced: bool


@dataclass(frozen=True)
class AcceptanceReports:
    """What Acceptance has said of a run's accepted factors.

    ``validation`` is the validation report on every accepted factor, or
    when there is none the latest validation report of a selection;
    ``holdout`` is the report of the latest open of the hold-out. Either is
    ``None`` while Acceptance has not written it. ``reports`` holds every
    report, the earliest written first; ``holdout_opens`` every open the
    hold-out lock records, which may include an open that wrote no report.
    """

    validation: Comparison | None
    holdout: Comparison | None
    reports: tuple[Comparison, ...]
    holdout_opens: tuple[HoldoutOpen, ...]


@dataclass(frozen=True)
class Finding:
    """One finding of a run's diagnosis: a thing seen in its trials and what to do about it.

    ``rule`` names the rule that found it: ``"unreadable_replies"``,
    ``"known_collisions"`` (one of the user's own factors),
    ``"accepted_collisions"`` (factors the run accepted), ``"coverage_refusals"``,
    ``"graph_build_failures"`` or ``"below_threshold"``. ``trials`` counts the
    trials it concerns. ``seen`` and ``do`` are sentences for the page.
    """

    rule: str
    trials: int
    seen: str
    do: str


@dataclass(frozen=True)
class RunSummary:
    """What the run list shows of a run.

    ``name`` is the run directory's path relative to the listed folder
    (the folder's own name when the folder is the run directory).
    ``code_version`` is the commit of this package the run was last opened
    with, ``"unknown"`` when it could not be read. ``rounds`` counts the
    written rounds, ``trials`` and ``accepted`` their trials and accepted
    ones. ``stop`` is ``None`` while the run has not stopped, or crashed
    without stopping. ``updated`` is the time of the run's last write (UTC).
    """

    name: str
    code_version: str
    rounds: int
    trials: int
    accepted: int
    stop: Stop | None
    updated: datetime


@dataclass(frozen=True)
class RunView:
    """A read-only snapshot of one run: its summary, the trials of its written rounds, its accepted factors and its memory.

    ``factors`` holds the accepted factors in the order they were accepted
    (by round, then slot). ``correlation_limit`` is the largest absolute
    correlation with an existing factor that admission allows, as the run's
    latest refusal for it quotes it, ``None`` while no refusal quotes it: the
    run directory does not record the campaign's admission settings.
    ``evaluated`` places every trial the Evaluator scored, in round and slot
    order, and ``threshold`` is the t threshold of the latest round as its or
    an earlier round's refusals quote it, ``None`` while none does.
    ``never_evaluated`` groups the trials it never scored, by stage (parse,
    Gate, review, then evaluation) and then by the first round of each reason.
    ``memory`` holds the versions of the memory by the round after which
    they were written, the earliest first; it is empty while the distiller
    has written none. ``acceptance`` holds the run's Acceptance reports and
    hold-out opens. ``diagnosis`` holds at most three findings, the one
    concerning the most trials first; it is empty when no rule holds.
    ``now`` is the time the view describes (UTC).
    """

    summary: RunSummary
    trials: tuple[Trial, ...]
    factors: tuple[Factor, ...]
    correlation_limit: float | None
    evaluated: tuple[Placed, ...]
    threshold: float | None
    never_evaluated: tuple[Refusal, ...]
    memory: tuple[MemoryVersion, ...]
    acceptance: AcceptanceReports
    diagnosis: tuple[Finding, ...]
    now: datetime

    @classmethod
    def read(cls, run_dir: str | Path, now: datetime | None = None, name: str | None = None) -> "RunView":
        """Read the run in ``run_dir``.

        Parameters
        ----------
        run_dir : str or Path
            A run directory: one holding a ``snapshot.json``.
        now : datetime, optional
            The time the view describes; the current time by default.
        name : str, optional
            The run's name in the view; the directory's name by default.

        Returns
        -------
        RunView

        Raises
        ------
        FileNotFoundError
            When ``run_dir`` holds no ``snapshot.json``.
        """
        run_dir = Path(run_dir)
        if not (run_dir / "snapshot.json").is_file():
            raise FileNotFoundError(f"web: {run_dir} is not a run directory: it holds no snapshot.json")
        state = _json(run_dir / "state.json") or {}
        rounds = state.get("rounds", 0)
        trials = tuple(_trials(run_dir / "ledger.jsonl", rounds))
        versions = (_json(run_dir / "snapshot.json") or {}).get("versions") or [{}]
        summary = RunSummary(
            name=name or run_dir.name,
            code_version=versions[-1].get("factor_agent") or "unknown",
            rounds=rounds,
            trials=len(trials),
            accepted=sum(t.status == "accepted" for t in trials),
            stop=_stop(run_dir, state, versions[-1].get("opened_at")),
            updated=_time(max(p.stat().st_mtime for p in _written(run_dir))),
        )
        return cls(summary=summary, trials=trials, factors=_factors(trials), correlation_limit=_correlation_limit(trials),
                   **_trials_page(trials), memory=_memory(run_dir / "memory"), acceptance=_acceptance(run_dir),
                   diagnosis=_diagnosis(trials),
                   now=now or datetime.now(timezone.utc))


def list_runs(root: str | Path) -> list[RunSummary]:
    """The runs under ``root``, the latest written first.

    Parameters
    ----------
    root : str or Path
        A folder of run directories, at any depth, or a run directory itself.

    Returns
    -------
    list of RunSummary
        One per run directory, named by its path relative to ``root``.
    """
    root = Path(root)
    found = sorted({p.parent for p in root.rglob("snapshot.json") if p.is_file()})
    summaries = [RunView.read(d, name=d.relative_to(root).as_posix() if d != root else root.name).summary for d in found]
    return sorted(summaries, key=lambda s: s.updated, reverse=True)


def _loads(text: str):
    # The ledger and the Acceptance reports write NaN where a value is undefined.
    return json.loads(text, parse_constant=lambda _: None)


def _json(path: Path):
    try:
        return _loads(path.read_text())
    except FileNotFoundError:
        return None


def _trials(ledger: Path, rounds: int) -> list[Trial]:
    """The trials of the first ``rounds`` rounds; later rounds and a line a crash cut short are skipped."""
    try:
        lines = ledger.read_text().splitlines()
    except FileNotFoundError:
        return []
    trials, slots = [], {}
    for line in lines:
        try:
            record = _loads(line)
        except json.JSONDecodeError:
            continue
        if record["round"] > rounds:
            continue
        slot = slots[record["round"]] = slots.get(record["round"], -1) + 1
        trials.append(Trial(
            round=record["round"], slot=slot, status=record["status"], stage=record["stage"],
            reasons=tuple(record["reasons"]), candidate=record["candidate"], direction=record.get("direction"),
            attempts=tuple(record.get("attempts") or ()), stats=record.get("stats"), metrics=record.get("metrics"),
            collisions=_trial_collisions(record),
        ))
    return trials


#: The partial-IC t threshold an admission refusal quotes.
_THRESHOLD = re.compile(r"is below the threshold ([\d.]+) at \d+ evaluated trial")


#: The correlation limit an admission refusal quotes.
_LIMIT = re.compile(r"^cross-sectional correlation .* is above ([\d.]+) in absolute value$", re.DOTALL)


def _correlation_limit(trials: tuple[Trial, ...]) -> float | None:
    """The correlation limit the latest refusal quoting it gives, ``None`` when none does."""
    quoted = [found.group(1) for trial in trials for reason in trial.reasons if (found := _LIMIT.match(reason))]
    return float(quoted[-1]) if quoted else None


def _thresholds(trials: tuple[Trial, ...]) -> dict[int, float | None]:
    """Each round's threshold as its refusals quote it, the previous round's when it quotes none."""
    quoted = {}
    for trial in trials:
        for reason in trial.reasons:
            if found := _THRESHOLD.search(reason):
                quoted[trial.round] = float(found.group(1))
    thresholds, last = {}, None
    for number in sorted({trial.round for trial in trials}):
        thresholds[number] = last = quoted.get(number, last)
    return thresholds


#: An accepted factor's label: its description and the round that accepted it.
_ACCEPTED = re.compile(r"^(.*) \(accepted in round (\d+)\)$", re.DOTALL)

#: How many words of an earlier accepted factor's description name it.
_FIRST_WORDS = 6


def _neighbour(label: str | None, correlation: float | None) -> Neighbour | None:
    """The nearest factor the Evaluator recorded, ``None`` when there was none to compare with."""
    if not label:
        return None
    if accepted := _ACCEPTED.match(label):
        words = accepted.group(1).split()
        first = " ".join(words[:_FIRST_WORDS]) + ("…" if len(words) > _FIRST_WORDS else "")
        number = int(accepted.group(2))
        return Neighbour(label=label, name=f"round {number}: {first}", accepted_round=number, correlation=correlation)
    return Neighbour(label=label, name=label, accepted_round=None, correlation=correlation)


def _ranks(values: list[float]) -> list[float]:
    """Each value's rank from 1, ties given their mean rank."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for i in order[start:end + 1]:
            ranks[i] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def _monotonicity(deciles: tuple[float | None, ...], sign: int) -> float | None:
    """The Spearman correlation of the decile number with the decile returns, signed by ``sign``, as admission's."""
    if len(deciles) < 2 or any(r is None for r in deciles) or len(set(deciles)) == 1:
        return None
    ranks, n = _ranks(list(deciles)), len(deciles)
    mean = (n + 1) / 2
    spread = sum((r - mean) ** 2 for r in ranks) * sum((i + 1 - mean) ** 2 for i in range(n))
    return sign * sum((r - mean) * (i + 1 - mean) for i, r in enumerate(ranks)) / spread ** 0.5


def _factors(trials: tuple[Trial, ...]) -> tuple[Factor, ...]:
    thresholds = _thresholds(trials)
    factors = []
    for trial in trials:
        if trial.status != "accepted":
            continue
        metrics, candidate = trial.metrics, trial.candidate
        sign = 1 if (metrics.get("ic_mean") or 0) > 0 else -1
        deciles = tuple(metrics.get("decile_returns") or ())
        oriented = {key: None if metrics.get(key) is None else sign * metrics[key]
                    for key in ("ic_mean", "ic_t", "partial_ic_mean", "partial_ic_t")}
        factors.append(Factor(
            label=f"{candidate['description']} (accepted in round {trial.round})",
            round=trial.round, slot=trial.slot, mechanism_type=(trial.direction or {}).get("mechanism_type"),
            hypothesis=candidate.get("hypothesis", ""), description=candidate["description"], code=candidate["code"],
            factor_direction=sign, **oriented, threshold=thresholds[trial.round],
            monotonicity=_monotonicity(deciles, sign),
            nearest=_neighbour(metrics.get("nearest"), metrics.get("nearest_correlation")),
            nearest_known=_neighbour(metrics.get("nearest_known"), metrics.get("nearest_known_correlation")),
            decile_returns=deciles, coverage=metrics.get("coverage"), metrics=metrics,
        ))
    return tuple(factors)


#: The refusals that name the factors a candidate collided with, by kind; group 1 holds the names.
_COLLISIONS = (
    ("code", re.compile(r"^the factor rewrites (.+?(?:\)| more)): ", re.DOTALL)),
    ("code", re.compile(r"^the factor is (.+?) with other windows or parameter values only; ", re.DOTALL)),
    ("code", re.compile(r"^the factor shares a \d+-node subtree with (.+?), more than the limit of \d+: ", re.DOTALL)),
    ("ranking", re.compile(
        r"^cross-sectional correlation [-+]?[\d.]+ with (.+?)(?:, sampled on [^,]*,)? is above [\d.]+ in absolute value$",
        re.DOTALL)),
)

#: One label in the Gate's list of them: a reference, known or accepted factor's label ends in its parentheses.
_LABEL = re.compile(r"(.+?) \((accepted in round \d+|[\w.]+(?:, a known factor)?)\)(?:, |(?: and \d+ more)?$)", re.DOTALL)


def _collisions(reason: str) -> list[Collision]:
    """The collisions a refusal names, none for any other refusal."""
    for kind, pattern in _COLLISIONS:
        if found := pattern.match(reason):
            names = found.group(1)
            labels = [names] if kind == "ranking" else [f"{m[1]} ({m[2]})" for m in _LABEL.finditer(names)]
            return [_collision(label, kind) for label in labels]
    return []


def _collision(label: str, kind: str) -> Collision:
    neighbour = _neighbour(label, None)
    return Collision(label=label, name=neighbour.name, kind=kind, accepted_round=neighbour.accepted_round,
                     known=label.endswith(", a known factor)"))


def _trial_collisions(record: dict) -> tuple[Collision, ...]:
    """Every collision of a trial's versions, each factor and kind once."""
    reasons = [r for attempt in record.get("attempts") or () for r in attempt.get("reasons") or ()]
    found = {}
    for reason in reasons + list(record["reasons"]):
        for collision in _collisions(reason):
            found.setdefault((collision.label, collision.kind), collision)
    return tuple(found.values())


#: Findings shown at most.
_FINDINGS = 3

#: The share of trials above which unreadable replies, coverage refusals and graph build failures are findings.
_UNREADABLE_SHARE, _COVERAGE_SHARE, _BUILD_SHARE = 0.20, 0.15, 0.15

#: The number of trials above which collisions with one of the user's factors, or with accepted ones, are a finding.
_COLLIDING_TRIALS = 5


def _refusals(trial: Trial) -> list[tuple[str, str]]:
    """Every (stage, reason) the trial's versions were refused with, its final refusal among them."""
    found = [(attempt.get("stage", ""), r) for attempt in trial.attempts for r in attempt.get("reasons") or ()]
    return found + [(trial.stage, r) for r in trial.reasons]


def _share(count: int, total: int) -> str:
    return f"{count} of {total} trials ({count / total:.0%})"


def _diagnosis(trials: list[Trial]) -> tuple[Finding, ...]:
    """The findings of the rules that hold, the most trials first, at most three."""
    n = len(trials)
    if not n:
        return ()
    findings = []
    unreadable = sum(any(stage == "parse" for stage, _ in _refusals(t)) for t in trials)
    if unreadable > _UNREADABLE_SHARE * n:
        findings.append(Finding(
            "unreadable_replies", unreadable,
            f"The LLM's replies could not be read as candidates in {_share(unreadable, n)}.",
            "Change the model, or raise its output limit if its replies are cut off."))
    own, accepted = {}, {}
    for trial in trials:
        for collision in {c.label: c for c in trial.collisions}.values():
            (own if collision.accepted_round is None else accepted).setdefault(collision.label, []).append(trial)
    if own:
        label, colliding = max(own.items(), key=lambda item: len(item[1]))
        if len(colliding) > _COLLIDING_TRIALS:
            findings.append(Finding(
                "known_collisions", len(colliding),
                f"Candidates collided with your factor {label} in {_share(len(colliding), n)}.",
                "Steer the research directions away from it, or check that your known factors are the ones you mean."))
    merged = {id(t) for colliding in accepted.values() for t in colliding}
    if len(merged) > _COLLIDING_TRIALS:
        names = ", ".join(f"{_neighbour(label, None).name} ({len(colliding)})"
                          for label, colliding in sorted(accepted.items(), key=lambda item: -len(item[1])))
        findings.append(Finding(
            "accepted_collisions", len(merged),
            f"Candidates collided with factors the run accepted earlier in {_share(len(merged), n)}: {names}.",
            "Nothing to fix: these are the novelty checks at work, refusing what the run already found. "
            "If it goes on, give later rounds other research directions."))
    coverage = sum(any(stage == "admission" and r.startswith("coverage ") for stage, r in _refusals(t)) for t in trials)
    if coverage > _COVERAGE_SHARE * n:
        findings.append(Finding(
            "coverage_refusals", coverage,
            f"Candidates had a value on too few members in {_share(coverage, n)}.",
            "Check that your data's fields have values for the members and dates the run trains on."))
    builds = sum(any(r.startswith(("graph build failed", "graph build crashed")) for _, r in _refusals(t)) for t in trials)
    if builds > _BUILD_SHARE * n:
        findings.append(Finding(
            "graph_build_failures", builds,
            f"Candidate code failed to build a graph in {_share(builds, n)}.",
            "Check the operator file's signatures and descriptions, or try a model that writes the operators correctly."))
    evaluated = [t for t in trials if t.metrics is not None]
    below = sum(any(_THRESHOLD.search(r) for r in t.reasons) for t in evaluated)
    if not any(t.status == "accepted" for t in trials) and below > len(evaluated) / 2:
        findings.append(Finding(
            "below_threshold", below,
            f"Nothing was accepted: {below} of {len(evaluated)} evaluated trials fell below the partial-IC t threshold.",
            "This is the statistical bar, not an error: more rounds or other research directions may clear it."))
    return tuple(sorted(findings, key=lambda f: -f.trials)[:_FINDINGS])


#: The order of the stages that refuse a trial before the Evaluator scores it.
_STAGES = ("parse", "gate", "review", "evaluate")

#: What a reason quotes after its message: a reply, code or a value.
_QUOTED = re.compile(r": ['\"`].*$", re.DOTALL)

#: A number in a reason.
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?")


def _trials_page(trials: tuple[Trial, ...]) -> dict:
    """The admission map's trials, the latest threshold and the trials never evaluated, as ``RunView`` fields."""
    thresholds = _thresholds(trials)
    evaluated = []
    for trial in trials:
        if not trial.metrics:
            continue
        metrics = trial.metrics
        sign = 1 if (metrics.get("ic_mean") or 0) > 0 else -1
        correlations = [abs(c) for c in (metrics.get("nearest_correlation"), metrics.get("nearest_known_correlation"))
                        if c is not None]
        evaluated.append(Placed(
            round=trial.round, slot=trial.slot, status=trial.status, factor_direction=sign,
            **{key: None if metrics.get(key) is None else sign * metrics[key]
               for key in ("ic_mean", "ic_t", "partial_ic_mean", "partial_ic_t")},
            largest_correlation=max(correlations) if correlations else None, threshold=thresholds[trial.round],
        ))
    groups: dict[tuple[str, str], list[Trial]] = {}
    for trial in trials:
        if trial.metrics or trial.status == "accepted":
            continue
        reason = trial.reasons[0] if trial.reasons else ""
        groups.setdefault((trial.stage, _NUMBER.sub("#", _QUOTED.sub("", reason))), []).append(trial)
    never = [Refusal(stage=stage, reason=members[0].reasons[0] if members[0].reasons else "",
                     rounds=tuple(sorted({t.round for t in members})),
                     trials=tuple(TrialRef(t.round, t.slot) for t in members))
             for (stage, _), members in groups.items()]
    never.sort(key=lambda g: (_STAGES.index(g.stage) if g.stage in _STAGES else len(_STAGES), g.trials[0].round,
                              g.trials[0].slot))
    return {"evaluated": tuple(evaluated), "threshold": thresholds[max(thresholds)] if thresholds else None,
            "never_evaluated": tuple(never)}


def _memory(folder: Path) -> tuple[MemoryVersion, ...]:
    """The memory's versions, the earliest first, each but the first compared with the one before."""
    files = {}
    for path in folder.glob("round_*.md"):
        if match := re.fullmatch(r"round_(\d+)\.md", path.name):
            files[int(match[1]) - 1] = path.read_text().rstrip("\n")
    versions, before = [], None
    for number, text in sorted(files.items()):
        versions.append(MemoryVersion(round=number, text=text, changes=None if before is None else _changes(before, text)))
        before = text
    return tuple(versions)


def _changes(before: str, after: str) -> tuple[MemoryLine, ...]:
    old, new = before.splitlines(), after.splitlines()
    lines = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes():
        if tag == "equal":
            lines += [MemoryLine("kept", line) for line in new[j1:j2]]
            continue
        lines += [MemoryLine("removed", line) for line in old[i1:i2]]
        lines += [MemoryLine("added", line) for line in new[j1:j2]]
    return tuple(lines)


def _verdict(difference: dict) -> str | None:
    rank_ic, excess = difference.get("rank_ic"), difference.get("top_k_excess")
    if rank_ic is None or excess is None:
        return None
    if rank_ic > 0 and excess > 0:
        return "improves"
    return "does_not_improve" if rank_ic <= 0 and excess <= 0 else "mixed"


def _comparison(path: Path) -> Comparison | None:
    report = _json(path)
    if not isinstance(report, dict) or "difference" not in report:
        return None
    mean, difference = report.get("mean") or {}, report["difference"]
    table = report.get("contribution")
    return Comparison(
        name=path.stem, segment=report.get("segment", path.stem), holdout=bool(report.get("holdout_opens")),
        factors=tuple(report.get("accepted") or ()), selection=report.get("selection"),
        seeds=tuple(report.get("seeds") or ()), top_k=report.get("top_k"),
        train_window=tuple(report.get("train_window") or ()), test_window=tuple(report.get("test_window") or ()),
        baseline=mean.get("baseline") or {}, with_accepted=mean.get("with_accepted") or {}, difference=difference,
        verdict=_verdict(difference), forced=bool(report.get("forced")),
        contribution=None if table is None else tuple(
            Contribution(factor=row["factor"], mode=row["mode"], difference=row["difference"], agreeing=row["agreeing"])
            for row in table
        ),
        code_version=(report.get("code_version") or {}).get("factor_agent") or "unknown",
        written=_time(path.stat().st_mtime),
    )


def _acceptance(run_dir: Path) -> AcceptanceReports:
    """The Acceptance reports, the earliest written first, and the hold-out lock's opens."""
    found = [(path.stat().st_mtime_ns, path) for path in (run_dir / "acceptance").glob("*.json")]
    reports = tuple(c for _, path in sorted(found) if (c := _comparison(path)))
    validations = [c for c in reports if not c.holdout]
    validation = next((c for c in validations if c.name == c.segment), validations[-1] if validations else None)
    holdouts = [c for c in reports if c.holdout]
    lock = _json(run_dir / "holdout.lock") or {}
    opens = tuple(
        HoldoutOpen(opened_at=o["opened_at"], code_version=(o.get("code_version") or {}).get("factor_agent") or "unknown",
                    forced=bool(o.get("forced")))
        for o in lock.get("opens") or ()
    )
    return AcceptanceReports(validation=validation, holdout=holdouts[-1] if holdouts else None, reports=reports,
                             holdout_opens=opens)


def _calls(run_dir: Path) -> list[Path]:
    return list((run_dir / "llm_log").glob("*.json"))


def _written(run_dir: Path) -> list[Path]:
    """The run's files that change while it runs."""
    files = [run_dir / name for name in ("snapshot.json", "state.json", "ledger.jsonl")]
    return [p for p in files if p.exists()] + _calls(run_dir)


def _stop(run_dir: Path, state: dict, opened_at: str | None) -> Stop | None:
    """The state's stop, unless it was recorded before the latest open (see the module's notes)."""
    if not state.get("stop"):
        return None
    written = (run_dir / "state.json").stat().st_mtime_ns
    if opened_at and written < datetime.fromisoformat(opened_at).timestamp() * 1e9:
        return None
    if any(p.stat().st_mtime_ns > written for p in _calls(run_dir)):
        return None
    return Stop(by=state["stop"]["by"], reason=state["stop"]["reason"])


def _time(seconds: float) -> datetime:
    return datetime.fromtimestamp(seconds, timezone.utc)
