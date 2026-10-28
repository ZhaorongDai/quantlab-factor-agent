"""The trials page: the evaluated trials on the admission map, and the others grouped by what refused them."""

import re
from dataclasses import dataclass

from factor_agent.web._files import Replayed
from factor_agent.web._ledger import Trial, in_direction, monotonicity, thresholds


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
    threshold of its round, as ``Factor.threshold`` (``None`` while nothing gives it).
    ``monotonicity`` is its decile monotonicity in its direction, as admission's.
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
    monotonicity: float | None


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
class TrialsPage:
    """The admission map's trials, the latest round's threshold and the trials never evaluated, as ``RunView`` holds them."""

    evaluated: tuple[Placed, ...]
    threshold: float | None
    never_evaluated: tuple[Refusal, ...]


#: The order of the stages that refuse a trial before the Evaluator scores it.
_STAGES = ("parse", "gate", "review", "evaluate")

#: What a reason quotes after its message: a reply, code or a value.
_QUOTED = re.compile(r": ['\"`].*$", re.DOTALL)

#: A number in a reason.
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?")


def trials_page(trials: tuple[Trial, ...], replayed: Replayed) -> TrialsPage:
    """What the trials page shows of ``trials``."""
    by_round = thresholds(trials, replayed)
    evaluated = []
    for trial in trials:
        if not trial.metrics:
            continue
        metrics = trial.metrics
        sign, oriented = in_direction(metrics)
        correlations = [abs(c) for c in (metrics.get("nearest_correlation"), metrics.get("nearest_known_correlation"))
                        if c is not None]
        evaluated.append(Placed(
            round=trial.round, slot=trial.slot, status=trial.status, factor_direction=sign,
            ic_mean=oriented["ic_mean"], ic_t=oriented["ic_t"], partial_ic_mean=oriented["partial_ic_mean"],
            partial_ic_t=oriented["partial_ic_t"], largest_correlation=max(correlations) if correlations else None,
            threshold=by_round[trial.round][0],
            monotonicity=monotonicity(tuple(metrics.get("decile_returns") or ()), sign),
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
    return TrialsPage(evaluated=tuple(evaluated), threshold=by_round[max(by_round)][0] if by_round else None,
                      never_evaluated=tuple(never))
