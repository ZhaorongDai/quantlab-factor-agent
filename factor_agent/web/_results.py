"""The results page's accepted factors, each with the facts that say how far to trust it."""

from dataclasses import dataclass

from factor_agent.web._files import Replayed
from factor_agent.web._ledger import Neighbour, Trial, in_direction, monotonicity, neighbour, thresholds


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
    ``threshold`` is its round's t threshold and ``threshold_source`` where
    it comes from: ``"admission"`` (the admission events of its round),
    ``"refusals"`` (what its round's refusals quote, for a run without
    events) or ``"carried_over"`` (neither gives it, so it is an earlier
    round's); both are ``None`` while nothing gives a threshold.
    ``decile_returns`` and ``coverage`` are the Evaluator's. ``intraday``
    is its trial's (``Trial.intraday``): its graph holds a session aggregation.
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
    threshold_source: str | None
    monotonicity: float | None
    nearest: Neighbour | None
    nearest_known: Neighbour | None
    decile_returns: tuple[float | None, ...]
    coverage: float | None
    metrics: dict
    intraday: bool = False


def factors(trials: tuple[Trial, ...], replayed: Replayed) -> tuple[Factor, ...]:
    """The accepted factors, in the order they were accepted."""
    by_round = thresholds(trials, replayed)
    found = []
    for trial in trials:
        if trial.status != "accepted":
            continue
        metrics, candidate = trial.metrics, trial.candidate
        sign, oriented = in_direction(metrics)
        deciles = tuple(metrics.get("decile_returns") or ())
        threshold, source = by_round[trial.round]
        found.append(Factor(
            label=f"{candidate['description']} (accepted in round {trial.round})",
            round=trial.round, slot=trial.slot, mechanism_type=(trial.direction or {}).get("mechanism_type"),
            hypothesis=candidate.get("hypothesis", ""), description=candidate["description"], code=candidate["code"],
            factor_direction=sign, ic_mean=oriented["ic_mean"], ic_t=oriented["ic_t"],
            partial_ic_mean=oriented["partial_ic_mean"], partial_ic_t=oriented["partial_ic_t"],
            threshold=threshold, threshold_source=source, monotonicity=monotonicity(deciles, sign),
            nearest=neighbour(metrics.get("nearest"), metrics.get("nearest_correlation")),
            nearest_known=neighbour(metrics.get("nearest_known"), metrics.get("nearest_known_correlation")),
            decile_returns=deciles, coverage=metrics.get("coverage"), metrics=metrics,
            intraday=trial.intraday,
        ))
    return tuple(found)
