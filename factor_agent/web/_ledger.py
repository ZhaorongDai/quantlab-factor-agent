"""The trials of a run's written rounds, the collisions their refusals name, and the limits admission judged them by."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from factor_agent import events
from factor_agent.web._files import Replayed, loads


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


def _records(ledger: Path, rounds: int):
    """The ledger's records of the first ``rounds`` rounds; later rounds and a line a crash cut short are skipped."""
    try:
        lines = ledger.read_text().splitlines()
    except FileNotFoundError:
        return
    for line in lines:
        try:
            record = loads(line)
        except json.JSONDecodeError:
            continue
        if record["round"] <= rounds:
            yield record


def statuses(ledger: Path, rounds: int) -> list[str]:
    """The status of every trial of the first ``rounds`` rounds, in ledger order: what a run's summary counts."""
    return [record["status"] for record in _records(ledger, rounds)]


def read_trials(ledger: Path, rounds: int) -> tuple[Trial, ...]:
    """The trials of the first ``rounds`` rounds, each given its slot in its round."""
    trials, slots = [], {}
    for record in _records(ledger, rounds):
        slot = slots[record["round"]] = slots.get(record["round"], -1) + 1
        trials.append(Trial(
            round=record["round"], slot=slot, status=record["status"], stage=record["stage"],
            reasons=tuple(record["reasons"]), candidate=record["candidate"], direction=record.get("direction"),
            attempts=tuple(record.get("attempts") or ()), stats=record.get("stats"), metrics=record.get("metrics"),
            collisions=_trial_collisions(record),
        ))
    return tuple(trials)


#: The partial-IC t threshold an admission refusal quotes.
THRESHOLD = re.compile(r"is below the threshold ([\d.]+) at \d+ evaluated trial")

#: The correlation limit an admission refusal quotes.
_LIMIT = re.compile(r"^cross-sectional correlation .* is above ([\d.]+) in absolute value$", re.DOTALL)


def correlation_limit(trials: tuple[Trial, ...], replayed: Replayed) -> float | None:
    """The correlation limit the run was last opened with, else the latest refusal quoting it, else ``None``."""
    if replayed.opened is not None:
        return replayed.opened.max_correlation
    quoted = [found.group(1) for trial in trials for reason in trial.reasons if (found := _LIMIT.match(reason))]
    return float(quoted[-1]) if quoted else None


def thresholds(trials: tuple[Trial, ...], replayed: Replayed) -> dict[int, tuple[float | None, str | None]]:
    """Each round's threshold and where it comes from: ``"admission"``, ``"refusals"`` or ``"carried_over"``.

    A round's admission events give it; without them its refusals quote it;
    when neither does it is the round before's, carried over. The admission
    events of a round a resume dropped come before those of the round run
    again under the same number, so the later ones count. A round before
    any threshold has ``(None, None)``.
    """
    given = {}
    for trial in trials:
        for reason in trial.reasons:
            if found := THRESHOLD.search(reason):
                given[trial.round] = (float(found.group(1)), "refusals")
    for event in replayed.all:
        if isinstance(event, events.AdmissionOutcome) and event.threshold is not None:
            given[event.round] = (event.threshold, "admission")
    found, last = {}, None
    for number in sorted({trial.round for trial in trials}):
        if number in given:
            found[number] = last = given[number]
        else:
            found[number] = (None, None) if last is None else (last[0], "carried_over")
    return found


#: An accepted factor's label: its description and the round that accepted it.
_ACCEPTED = re.compile(r"^(.*) \(accepted in round (\d+)\)$", re.DOTALL)

#: How many words of an earlier accepted factor's description name it.
_FIRST_WORDS = 6


def neighbour(label: str | None, correlation: float | None) -> Neighbour | None:
    """The nearest factor the Evaluator recorded, ``None`` when there was none to compare with."""
    if not label:
        return None
    if accepted := _ACCEPTED.match(label):
        words = accepted.group(1).split()
        first = " ".join(words[:_FIRST_WORDS]) + ("…" if len(words) > _FIRST_WORDS else "")
        number = int(accepted.group(2))
        return Neighbour(label=label, name=f"round {number}: {first}", accepted_round=number, correlation=correlation)
    return Neighbour(label=label, name=label, accepted_round=None, correlation=correlation)


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
    """The collision of ``kind`` with the factor ``label`` names."""
    named = neighbour(label, None)
    return Collision(label=label, name=named.name, kind=kind, accepted_round=named.accepted_round,
                     known=label.endswith(", a known factor)"))


def _trial_collisions(record: dict) -> tuple[Collision, ...]:
    """Every collision of a trial's versions, each factor and kind once."""
    reasons = [r for attempt in record.get("attempts") or () for r in attempt.get("reasons") or ()]
    found = {}
    for reason in reasons + list(record["reasons"]):
        for collision in _collisions(reason):
            found.setdefault((collision.label, collision.kind), collision)
    return tuple(found.values())


def in_direction(metrics: dict) -> tuple[int, dict[str, float | None]]:
    """A scored trial's direction, +1 or -1 by the sign of its training IC, and its ICs and t in that direction."""
    sign = 1 if (metrics.get("ic_mean") or 0) > 0 else -1
    return sign, {key: None if metrics.get(key) is None else sign * metrics[key]
                  for key in ("ic_mean", "ic_t", "partial_ic_mean", "partial_ic_t")}
