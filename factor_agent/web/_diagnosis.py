"""The diagnosis: at most three findings in a run's trials, each with what to do about it.

The rules and their thresholds are fixed here. Each finding counts trials,
not attempts: a trial is counted once whichever of its versions met the
problem.
"""

from dataclasses import dataclass

from factor_agent.web._ledger import THRESHOLD, Trial, neighbour


@dataclass(frozen=True)
class Finding:
    """One finding of a run's diagnosis: a thing seen in its trials and what to do about it.

    ``rule`` names the rule that found it: ``"unreadable_replies"``,
    ``"known_collisions"`` (one of the user's known factors),
    ``"accepted_collisions"`` (factors the run accepted), ``"coverage_refusals"``,
    ``"graph_build_failures"`` or ``"below_threshold"``. ``trials`` counts the
    trials it concerns. ``seen`` and ``do`` are sentences for the page.
    """

    rule: str
    trials: int
    seen: str
    do: str


#: Findings shown at most.
_FINDINGS = 3

#: The share of trials above which unreadable replies, coverage refusals and graph build failures are findings.
_UNREADABLE_SHARE, _COVERAGE_SHARE, _BUILD_SHARE = 0.20, 0.15, 0.15

#: The number of trials above which collisions with one known factor, or with accepted ones, are a finding.
_COLLIDING_TRIALS = 5


def _refusals(trial: Trial) -> list[tuple[str, str]]:
    """Every (stage, reason) the trial's versions were refused with, its final refusal among them."""
    found = [(attempt.get("stage", ""), r) for attempt in trial.attempts for r in attempt.get("reasons") or ()]
    return found + [(trial.stage, r) for r in trial.reasons]


def _share(count: int, total: int) -> str:
    """``count`` of ``total`` trials, with the percentage."""
    return f"{count} of {total} trials ({count / total:.0%})"


def diagnosis(trials: tuple[Trial, ...]) -> tuple[Finding, ...]:
    """The findings of the rules that hold, the most trials first, at most three.

    Collisions with a reference factor are no finding: the reference pool is
    there to be compared with, while a known factor the run keeps rewriting
    is one the user may not mean to name, and an accepted one shows the
    novelty checks at work.
    """
    n = len(trials)
    if not n:
        return ()
    findings = []
    unreadable = sum(any(stage == "parse" for stage, _ in _refusals(t)) for t in trials)
    if unreadable > _UNREADABLE_SHARE * n:
        findings.append(Finding(
            "unreadable_replies", unreadable,
            f"Unreadable LLM replies: {_share(unreadable, n)}.",
            "Change the model or raise its output limit."))
    known, accepted = {}, {}
    for trial in trials:
        for collision in {c.label: c for c in trial.collisions}.values():
            if collision.known:
                known.setdefault(collision.label, []).append(trial)
            elif collision.accepted_round is not None:
                accepted.setdefault(collision.label, []).append(trial)
    if known:
        label, colliding = max(known.items(), key=lambda item: len(item[1]))
        if len(colliding) > _COLLIDING_TRIALS:
            findings.append(Finding(
                "known_collisions", len(colliding),
                f"Collided with your known factor {label}: {_share(len(colliding), n)}.",
                "Steer the research directions away from it, or check your known factors."))
    merged = {id(t) for colliding in accepted.values() for t in colliding}
    if len(merged) > _COLLIDING_TRIALS:
        names = ", ".join(f"{neighbour(label, None).name} ({len(colliding)})"
                          for label, colliding in sorted(accepted.items(), key=lambda item: -len(item[1])))
        findings.append(Finding(
            "accepted_collisions", len(merged),
            f"Collided with factors the run accepted: {_share(len(merged), n)} ({names}).",
            "Expected: the novelty checks at work. If it persists, change the research directions."))
    coverage = sum(any(stage == "admission" and r.startswith("coverage ") for stage, r in _refusals(t)) for t in trials)
    if coverage > _COVERAGE_SHARE * n:
        findings.append(Finding(
            "coverage_refusals", coverage,
            f"Too few members covered: {_share(coverage, n)}.",
            "Check your data's fields and dates."))
    builds = sum(any(r.startswith(("graph build failed", "graph build crashed")) for _, r in _refusals(t)) for t in trials)
    if builds > _BUILD_SHARE * n:
        findings.append(Finding(
            "graph_build_failures", builds,
            f"Code failed to build: {_share(builds, n)}.",
            "Check the operator file, or try another model."))
    evaluated = [t for t in trials if t.metrics is not None]
    below = sum(any(THRESHOLD.search(r) for r in t.reasons) for t in evaluated)
    if not any(t.status == "accepted" for t in trials) and below > len(evaluated) / 2:
        findings.append(Finding(
            "below_threshold", below,
            f"Nothing accepted: {below} of {len(evaluated)} evaluated trials below the t threshold.",
            "The statistical bar, not an error: run more rounds or change the research directions."))
    return tuple(sorted(findings, key=lambda f: -f.trials)[:_FINDINGS])
