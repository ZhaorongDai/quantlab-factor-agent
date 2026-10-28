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
- The events file (``factor_agent.events``) tells what the run is doing:
  its phase, its calls and their tokens, its process, and the settings it
  was opened with. A run a campaign without events wrote has none of this
  but the calls of its round in progress, counted from its call files.
- The round in progress is replayed from the events since the run was
  last opened: its calls, and a pending trial per slot of its plan. A
  round a resume dropped left its events before the resume, so it is not
  among them. A run without events has no round in progress to show.
- An LLM call is read by the number of its call file, from the file
  itself: tokens, truncation and the model's reasoning are the provider's
  reply's; what it concerned (round, slots) only the events file says.
- A resumed run keeps the stop of its previous open in its state file until
  it stops again. With events, a stop counts only when a stop event follows
  the latest start or resume. Without, it counts only when the state file
  was written after the run's latest open and no LLM call ended after it: a
  resumed run that has asked the LLM since, or has written no state since
  it was opened, has not stopped.
- A round's t threshold (one per round, since a round's candidates are
  judged at the same trial count) is the one its admission events give.
  Without them it is the one its refusals quote, carried over from the
  round before when the round quotes none. The correlation limit is the
  one the run was last opened with, or without events the one a refusal
  quotes. Either is ``None`` while nothing gives it.
- The process that opened the run last is looked for only on the host it
  ran on; on another host it is unknown, never gone.
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
import socket
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psutil

from factor_agent import events
from factor_agent.llm.base import Reply


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
class Health:
    """What a run is doing now, as its events file and its process tell.

    ``phase`` is ``"planning"`` (a round is written, or the run was just
    opened, and no call of the next round has been sent), ``"asking"`` (the
    LLM is being asked for the round in progress) or ``"evaluating"`` (the
    Evaluator scores the round's candidates); ``None`` when the run has
    stopped or crashed, or has no events file. ``round`` is the round in
    progress and ``max_rounds`` the round limit, ``None`` when the run has
    stopped or the limit is unknown. ``process`` is ``"running"`` or
    ``"gone"`` when the campaign's process is checked on this host, and
    ``"unknown"`` when the campaign ran on another host or the run has no
    events file; ``crashed`` is true when the process is gone and the run
    recorded no stop. ``last_call`` is when the latest LLM call came back
    (UTC). ``stalled`` is true when the run looks stuck: no call was sent
    or came back for ``STALL_ASKING`` while asking, or the evaluation has
    lasted ``STALL_EVALUATING``. ``calls`` counts the calls of the round in
    progress by role, since the run was last opened; a run without an
    events file has only these counts, from its call files. ``tokens`` are
    the tokens of every call of the run, a round a resume dropped
    included, and ``round_tokens`` the same by round; ``None`` and empty
    without an events file. ``time_left`` estimates the seconds until the
    round or trial limit stops the run, from the mean length of its last
    ``RECENT_ROUNDS`` rounds; ``None`` while no round was timed, and once
    the run has stopped or crashed.
    """

    phase: str | None
    round: int | None
    max_rounds: int | None
    process: str
    crashed: bool
    last_call: datetime | None
    stalled: bool
    calls: dict[str, int]
    time_left: float | None
    tokens: "Tokens | None"
    round_tokens: tuple["RoundTokens", ...]


@dataclass(frozen=True)
class Tokens:
    """Tokens of LLM calls as the providers report them; a count no reply reported is ``None``.

    ``cache_hits`` are input tokens the provider read from its cache, and
    ``reasoning`` output tokens the model spent reasoning.
    """

    input: int | None
    output: int | None
    cache_hits: int | None
    reasoning: int | None


@dataclass(frozen=True)
class RoundTokens:
    """The tokens of a round's calls."""

    round: int
    tokens: Tokens


@dataclass(frozen=True)
class Check:
    """One check of a pending trial's candidate: the parser and the Gate on one version of it, or the reviewer.

    ``by`` is ``"gate"`` or ``"reviewer"``. For the Gate, ``step`` is the
    version's: ``"proposed"`` (the first), ``"fixed"`` (sent back by the
    parser or the Gate) or ``"rewritten"`` (sent back by the reviewer), and
    ``stage`` is ``"parse"`` or ``"gate"`` when it refused the version,
    ``None`` when it passed it. ``passed`` is true when the Gate passed the
    version or the reviewer found it consistent. ``reasons`` are the Gate's
    reasons for a refusal, or the reviewer's reason for its verdict either
    way; ``hints`` are the Gate's.
    """

    by: str
    step: str | None
    passed: bool
    stage: str | None
    reasons: tuple[str, ...]
    hints: tuple[str, ...]
    time: datetime


@dataclass(frozen=True)
class PendingTrial:
    """A pending trial: what is known so far of one slot of the round in progress.

    ``stage`` is the furthest it went: ``"planned"`` (its research direction
    is chosen), ``"proposed"``, ``"fixed"`` (after the Gate refused it),
    ``"rewritten"`` (after the reviewer found it inconsistent), and
    ``"consistent"`` or ``"inconsistent"`` as the reviewer last found it.
    ``direction`` is its research direction as the plan gives it. ``latest``
    is its latest version as the proposer wrote it, ``None`` before the
    first. ``checks`` are the Gate's outcomes and the reviewer's verdicts
    so far, the earliest first. It becomes a trial when its round is written.
    """

    slot: int
    stage: str
    direction: dict
    latest: object
    checks: tuple[Check, ...]


@dataclass(frozen=True)
class RoundCall:
    """An LLM call of the round in progress, as the call timeline shows it.

    ``number`` is its call file's number, by which ``read_call`` reads it;
    ``None`` while it has not come back, or when it failed (``error`` says
    how). ``group`` is its proposal group and ``slots`` the slots it
    concerns; both are empty for the planner, whose call concerns the round.
    ``sent`` and ``returned`` are when it was sent and came back (UTC).
    """

    number: int | None
    role: str
    model: str
    group: int | None
    slots: tuple[int, ...]
    sent: datetime
    returned: datetime | None
    error: str | None


@dataclass(frozen=True)
class RunningRound:
    """The round in progress, as its events tell since the run was last opened.

    ``started`` is when the round before it was written, or the run was
    opened when it is the first since. ``calls`` are its LLM calls in the
    order they were sent, and ``evaluation`` when the Evaluator started on
    it (``None`` before). ``pending`` holds a pending trial per slot of its
    plan, in slot order; it is empty until the plan is made.
    """

    round: int
    started: datetime
    calls: tuple[RoundCall, ...]
    evaluation: datetime | None
    pending: tuple[PendingTrial, ...]


@dataclass(frozen=True)
class LLMCall:
    """One LLM call of a run, in full, as its call file holds it.

    ``number`` is the number of its call file. ``round``, ``group`` and
    ``slots`` are the round, proposal group and slots it concerned, as the
    events file says (``None`` and empty without one). ``time`` is when it
    came back (UTC) and ``seconds`` how long it took (``None`` without an
    events file). ``tokens``, ``truncated`` (cut off at the output limit,
    ``None`` when the provider's reply does not say) and ``empty`` are read
    from the provider's reply. ``reasoning`` is the model's reasoning when
    the provider returns it, ``None`` otherwise.
    """

    number: int
    role: str
    model: str
    round: int | None
    group: int | None
    slots: tuple[int, ...]
    time: datetime
    seconds: float | None
    tokens: Tokens
    truncated: bool | None
    empty: bool
    prompt: str
    reply: str
    reasoning: str | None


@dataclass(frozen=True)
class RunView:
    """A read-only snapshot of one run: its summary, the trials of its written rounds, its accepted factors and its memory.

    ``factors`` holds the accepted factors in the order they were accepted
    (by round, then slot). ``correlation_limit`` is the largest absolute
    correlation with an existing factor that admission allows: the one the
    run was last opened with, or for a run without events the one its
    latest refusal for it quotes, ``None`` while nothing gives it.
    ``evaluated`` places every trial the Evaluator scored, in round and slot
    order, and ``threshold`` is the t threshold of the latest round (see the
    module's notes), ``None`` while nothing gives it.
    ``never_evaluated`` groups the trials it never scored, by stage (parse,
    Gate, review, then evaluation) and then by the first round of each reason.
    ``memory`` holds the versions of the memory by the round after which
    they were written, the earliest first; it is empty while the distiller
    has written none. ``acceptance`` holds the run's Acceptance reports and
    hold-out opens. ``diagnosis`` holds at most three findings, the one
    concerning the most trials first; it is empty when no rule holds.
    ``health`` is what the run is doing now, and ``running`` the round in
    progress: ``None`` when the run has stopped, or has no events file (its
    health then counts the round's calls by role). ``now`` is the time the
    view describes (UTC).
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
    health: Health
    running: RunningRound | None
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
        now = now or datetime.now(timezone.utc)
        state = _json(run_dir / "state.json") or {}
        rounds = state.get("rounds", 0)
        trials = tuple(_trials(run_dir / "ledger.jsonl", rounds))
        versions = (_json(run_dir / "snapshot.json") or {}).get("versions") or [{}]
        replayed = _Events(events.read(run_dir / events.FILE))
        summary = RunSummary(
            name=name or run_dir.name,
            code_version=versions[-1].get("factor_agent") or "unknown",
            rounds=rounds,
            trials=len(trials),
            accepted=sum(t.status == "accepted" for t in trials),
            stop=_stop(run_dir, state, versions[-1].get("opened_at"), replayed),
            updated=_time(max(p.stat().st_mtime for p in _written(run_dir))),
        )
        return cls(summary=summary, trials=trials, factors=_factors(trials, replayed),
                   correlation_limit=_correlation_limit(trials, replayed),
                   **_trials_page(trials, replayed), memory=_memory(run_dir / "memory"),
                   acceptance=_acceptance(run_dir), diagnosis=_diagnosis(trials),
                   health=_health(run_dir, state, summary, replayed, now),
                   running=_running(summary, replayed), now=now)


def run_version(run_dir: str | Path) -> str:
    """A text that changes whenever the run in ``run_dir`` writes, cheap to read: a page polls it before reading the run.

    It is made of the modification times and sizes of the files a campaign
    writes while it runs, and the number of its call files.
    """
    run_dir = Path(run_dir)
    parts = []
    for name in ("snapshot.json", "state.json", "ledger.jsonl", events.FILE, "llm_log", "memory", "acceptance",
                 "holdout.lock"):
        try:
            stat = (run_dir / name).stat()
        except FileNotFoundError:
            parts.append("-")
            continue
        parts.append(f"{stat.st_mtime_ns}:{stat.st_size}")
    return " ".join(parts)


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
    summaries = [RunView.read(d, name=name).summary for name, d in _run_dirs(root).items()]
    return sorted(summaries, key=lambda s: s.updated, reverse=True)


def find_run(root: str | Path, name: str) -> Path | None:
    """The run directory ``list_runs(root)`` names ``name``, ``None`` when it names none so."""
    return _run_dirs(Path(root)).get(name)


def read_call(run_dir: str | Path, number: int) -> LLMCall | None:
    """The LLM call of the run in ``run_dir`` whose call file has ``number``, ``None`` when it has none.

    A call of a round a resume dropped is not read: its file is no longer
    in the run's sequence of calls.
    """
    run_dir = Path(run_dir)
    files = list((run_dir / "llm_log").glob(f"{number:06d}-*.json")) if number > 0 else []
    if not files:
        return None
    path = files[0]
    try:
        call = _loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    request, reply = call.get("request") or {}, call.get("reply") or {}
    text = reply.get("text") or ""
    facts = events.reply_facts(Reply(text=text, usage=reply.get("usage") or {}, model_id=reply.get("model_id") or "",
                                     raw=reply.get("raw") or ""))
    end = next((e for e in reversed(events.read(run_dir / events.FILE))
                if isinstance(e, events.CallEnd) and e.file == path.name), None)
    return LLMCall(
        number=number, role=request.get("role") or path.stem.partition("-")[2], model=request.get("model") or "",
        round=end.round if end else None, group=end.group if end else None, slots=tuple(end.slots) if end else (),
        time=end.time if end else _time(path.stat().st_mtime), seconds=end.seconds if end else None,
        tokens=Tokens(facts["input_tokens"], facts["output_tokens"], facts["cache_hits"], facts["reasoning"]),
        truncated=facts["truncated"], empty=facts["empty"], prompt=request.get("prompt") or "", reply=text,
        reasoning=_reasoning(reply.get("raw") or ""),
    )


def _reasoning(raw: str) -> str | None:
    """The model's reasoning in a provider's raw reply: an OpenAI-compatible message's, or an Anthropic message's thinking."""
    try:
        value = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    choices = value.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        text = message.get("reasoning_content") or message.get("reasoning") if isinstance(message, dict) else None
        return text if isinstance(text, str) and text else None
    blocks = value.get("content")
    if isinstance(blocks, list):
        thinking = [b.get("thinking") for b in blocks if isinstance(b, dict) and b.get("type") == "thinking"]
        text = "\n\n".join(t for t in thinking if isinstance(t, str) and t)
        return text or None
    return None


def _run_dirs(root: Path) -> dict[str, Path]:
    """The run directories under ``root`` by their names."""
    found = sorted({p.parent for p in root.rglob("snapshot.json") if p.is_file()})
    return {d.relative_to(root).as_posix() if d != root else root.name: d for d in found}


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


def _correlation_limit(trials: tuple[Trial, ...], replayed: "_Events") -> float | None:
    """The correlation limit the run was last opened with, else the latest refusal quoting it, else ``None``."""
    if replayed.opened is not None:
        return replayed.opened.max_correlation
    quoted = [found.group(1) for trial in trials for reason in trial.reasons if (found := _LIMIT.match(reason))]
    return float(quoted[-1]) if quoted else None


def _thresholds(trials: tuple[Trial, ...], replayed: "_Events") -> dict[int, float | None]:
    """Each round's threshold as its admission events give it, else as its refusals quote it, else the previous round's.

    The admission events of a round a resume dropped come before those of
    the round run again under the same number, so the later ones count.
    """
    quoted = {}
    for trial in trials:
        for reason in trial.reasons:
            if found := _THRESHOLD.search(reason):
                quoted[trial.round] = float(found.group(1))
    for event in replayed.all:
        if isinstance(event, events.AdmissionOutcome) and event.threshold is not None:
            quoted[event.round] = event.threshold
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


def _factors(trials: tuple[Trial, ...], replayed: "_Events") -> tuple[Factor, ...]:
    thresholds = _thresholds(trials, replayed)
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


def _trials_page(trials: tuple[Trial, ...], replayed: "_Events") -> dict:
    """The admission map's trials, the latest threshold and the trials never evaluated, as ``RunView`` fields."""
    thresholds = _thresholds(trials, replayed)
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
    files = [run_dir / name for name in ("snapshot.json", "state.json", "ledger.jsonl", events.FILE)]
    return [p for p in files if p.exists()] + _calls(run_dir)


class _Events:
    """A run's events, and those since its latest opening (``current``), the opening first."""

    def __init__(self, replayed: list[events.Event]):
        self.all = replayed
        openings = [i for i, event in enumerate(replayed) if isinstance(event, events.Start)]
        self.current = replayed[openings[-1]:] if openings else []
        self.opened: events.Start | None = self.current[0] if self.current else None


#: The events of each phase of a round: the first a phase starts with, the later ones it goes on with.
_PHASES = {
    events.Start: "planning", events.Resume: "planning", events.RoundEnd: "planning",
    events.CallStart: "asking", events.CallEnd: "asking", events.Plan: "asking", events.GateOutcome: "asking",
    events.ReviewVerdict: "asking",
    events.EvaluationStart: "evaluating", events.AdmissionOutcome: "evaluating",
}


#: How long a run asking the LLM may go without a call sent or back before it looks stuck.
STALL_ASKING = timedelta(minutes=15)

#: How long an evaluation may last before the run looks stuck.
STALL_EVALUATING = timedelta(minutes=30)


def _health(run_dir: Path, state: dict, summary: RunSummary, replayed: _Events, now: datetime) -> Health:
    opened = replayed.opened
    process = _process(opened)
    crashed = summary.stop is None and process == "gone"
    phase = None
    if opened is not None and summary.stop is None and not crashed:
        phase = _PHASES.get(type(replayed.current[-1]))
    running = None if summary.stop else summary.rounds + 1
    ends = [e for e in replayed.all if isinstance(e, events.CallEnd) and e.error is None]
    if opened is None:
        files = sorted(_calls(run_dir))
        last_call = _time(max(p.stat().st_mtime for p in files)) if files else None
        roles = [] if running is None else [p.stem.partition("-")[2] for p in files[state.get("calls", 0):]]
    else:
        last_call = ends[-1].time if ends else None
        roles = [e.role for e in replayed.current if isinstance(e, events.CallEnd) and e.file and e.round == running]
    by_round = {}
    for end in ends:
        by_round.setdefault(end.round, []).append(end)
    return Health(
        phase=phase, round=running, max_rounds=opened.max_rounds if opened else None, process=process,
        crashed=crashed, last_call=last_call, stalled=_stalled(phase, replayed, now),
        calls={role: roles.count(role) for role in dict.fromkeys(roles)},
        time_left=None if running is None or crashed else _time_left(summary, replayed, now),
        tokens=_tokens(ends) if opened else None,
        round_tokens=tuple(RoundTokens(number, _tokens(calls)) for number, calls in sorted(by_round.items())),
    )


#: How many of the latest timed rounds the time left is estimated from.
RECENT_ROUNDS = 3


def _time_left(summary: RunSummary, replayed: _Events, now: datetime) -> float | None:
    """The seconds the rounds the limits still allow would take at the recent rounds' pace, the round begun included.

    A round is timed from the end of the round before, or the opening of
    the run when it was the first after it, to its own end.
    """
    opened = replayed.opened
    if opened is None:
        return None
    lengths, since = [], None
    for event in replayed.all:
        if isinstance(event, (events.Start, events.RoundEnd)):
            if isinstance(event, events.RoundEnd) and since is not None:
                lengths.append((event.time - since).total_seconds())
            since = event.time
    if not lengths:
        return None
    rounds = min(opened.max_rounds - summary.rounds, (opened.max_trials - summary.trials) // max(opened.candidates, 1))
    pace = sum(lengths[-RECENT_ROUNDS:]) / len(lengths[-RECENT_ROUNDS:])
    return max(0.0, pace * rounds - (now - since).total_seconds())


def _tokens(ends: list[events.CallEnd]) -> Tokens:
    """The tokens of the calls that ``ends`` end, each count ``None`` when no call reported it."""
    def total(key):
        counts = [getattr(end, key) for end in ends if getattr(end, key) is not None]
        return sum(counts) if counts else None
    return Tokens(total("input_tokens"), total("output_tokens"), total("cache_hits"), total("reasoning"))


def _stalled(phase: str | None, replayed: _Events, now: datetime) -> bool:
    if phase == "asking":
        calls = [e.time for e in replayed.current if isinstance(e, (events.CallStart, events.Start))]
        return now - calls[-1] > STALL_ASKING
    if phase == "evaluating":
        started = [e.time for e in replayed.current if isinstance(e, events.EvaluationStart)]
        return now - started[-1] > STALL_EVALUATING
    return False


def _running(summary: RunSummary, replayed: _Events) -> RunningRound | None:
    """The round in progress, from the events since the run was last opened; a round a resume dropped is not among them."""
    if replayed.opened is None or summary.stop is not None:
        return None
    number = summary.rounds + 1
    started = replayed.opened.time
    calls: dict[int, RoundCall] = {}
    evaluation = None
    directions: list[dict] = []
    stages: dict[int, str] = {}
    latest: dict[int, object] = {}
    checks: dict[int, list[Check]] = {}
    for event in replayed.current:
        if isinstance(event, events.RoundEnd):
            started = event.time
        if getattr(event, "round", None) != number:
            continue
        if isinstance(event, events.CallEnd):
            sent = calls[event.call].sent if event.call in calls else event.time - timedelta(seconds=event.seconds)
            calls[event.call] = RoundCall(
                number=_call_number(event.file), role=event.role, model=event.model, group=event.group,
                slots=tuple(event.slots), sent=sent, returned=event.time, error=event.error,
            )
        elif isinstance(event, events.CallStart):
            calls[event.call] = RoundCall(number=None, role=event.role, model=event.model, group=event.group,
                                          slots=tuple(event.slots), sent=event.time, returned=None, error=None)
        elif isinstance(event, events.EvaluationStart):
            evaluation = event.time
        elif isinstance(event, events.Plan):
            directions = event.directions
            stages = dict.fromkeys(range(len(directions)), "planned")
        elif isinstance(event, events.GateOutcome):
            stages[event.slot], latest[event.slot] = event.step, event.item
            checks.setdefault(event.slot, []).append(Check(
                by="gate", step=event.step, passed=event.passed, stage=event.stage, reasons=tuple(event.reasons),
                hints=tuple(event.hints), time=event.time,
            ))
        elif isinstance(event, events.ReviewVerdict):
            stages[event.slot] = "consistent" if event.consistent else "inconsistent"
            checks.setdefault(event.slot, []).append(Check(
                by="reviewer", step=None, passed=event.consistent, stage=None, reasons=(event.reason,), hints=(),
                time=event.time,
            ))
    pending = tuple(
        PendingTrial(slot=slot, stage=stages[slot], direction=direction, latest=latest.get(slot),
                     checks=tuple(checks.get(slot, ())))
        for slot, direction in enumerate(directions)
    )
    return RunningRound(round=number, started=started, calls=tuple(calls.values()), evaluation=evaluation,
                        pending=pending)


def _call_number(file: str | None) -> int | None:
    """The number of the call file named ``file``, as ``<number>-<role>.json``."""
    return int(file.partition("-")[0]) if file else None


def _process(opened: "events.Start | None") -> str:
    """Whether the process that opened the run last still runs, when it ran on this host."""
    if opened is None or opened.host != socket.gethostname():
        return "unknown"
    try:
        started = psutil.Process(opened.pid).create_time()
    except psutil.NoSuchProcess:
        return "gone"
    except psutil.Error:
        return "unknown"
    # A process started after the run was opened has only been given the same id.
    return "running" if started <= opened.time.timestamp() + 1 else "gone"


def _stop(run_dir: Path, state: dict, opened_at: str | None, replayed: _Events) -> Stop | None:
    """The state's stop, unless it was recorded before the latest open (see the module's notes)."""
    if not state.get("stop"):
        return None
    if replayed.opened is not None:
        if not any(isinstance(event, events.Stopped) for event in replayed.current):
            return None
        return Stop(by=state["stop"]["by"], reason=state["stop"]["reason"])
    written = (run_dir / "state.json").stat().st_mtime_ns
    if opened_at and written < datetime.fromisoformat(opened_at).timestamp() * 1e9:
        return None
    if any(p.stat().st_mtime_ns > written for p in _calls(run_dir)):
        return None
    return Stop(by=state["stop"]["by"], reason=state["stop"]["reason"])


def _time(seconds: float) -> datetime:
    return datetime.fromtimestamp(seconds, timezone.utc)
