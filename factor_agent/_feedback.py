"""The context the planner and the proposer read: what is known, assembled from the ledger.

It shows only training-segment statistics: each candidate's IC and its
Newey-West t, its residual (partial) IC, what the pool explains of it and the
pool factors most correlated with it, the order of its deciles' returns (as
ranks), the nearest existing factor and the correlation with it, the nearest
known factor and its sampled correlation, the rejection reasons, and each
accepted factor's marginal weight and residual IC in the accepted pool, and,
when given, each accepted factor's and each parent's Lineage (``_lineage``). It
never shows a return, a date, or advice to change windows or parameters. The first round
has no history; it sees the data, the seed directions and the existing
factor families (the reference set's, then the known factors', marked as
known) by name and description, without any statistic.

Every trial is named by its id, ``T<round>-<slot>`` (``named``), by which a
research direction names the trial it builds on (``parent``).
"""

import math
import re

from factor_agent._admission import direction, reason_kind
from factor_agent._lineage import Lineage
from factor_agent.evaluator import Contribution
from factor_agent.ledger import TrialRecord
from factor_agent.zoo import owner

_DECIMAL = re.compile(r"[-+]?\d+\.\d+")
_TRIAL_ID = re.compile(r"T(\d+)-(\d+)")


def first_recent_round(rounds: int, recent_rounds: int) -> int:
    """The first round of the recent history after ``rounds`` rounds; the rounds before it are the earlier summary's."""
    return rounds - recent_rounds + 1


def distill_due(rounds: int, memory_round: int, memory_every: int) -> bool:
    """Whether the distiller is asked after ``rounds`` rounds, when it was last asked after ``memory_round``."""
    return rounds - memory_round >= memory_every


def accepted_source(number: int) -> str:
    """The zoo source of a factor accepted in round ``number``."""
    return f"accepted in round {number}"


def accepted_label(record: TrialRecord) -> str:
    """The name an accepted factor has in the zoo, the pool and the prompts (``ZooEntry.label``)."""
    return f"{record.candidate['description']} ({accepted_source(record.round)})"


#: How the families section marks a known factor's class.
KNOWN_MARK = "known, not in the reference set, refused when rewritten"


def families(reference, known=()) -> str:
    """One line per factor of the reference set, then per known factor: its owner's class, doc line and outputs.

    The owner (``zoo.owner``) is the factor a wrapper such as
    ``NeutralizedFactor`` shows; the doc line is the first line of its
    class docstring. A known factor's class is marked with ``KNOWN_MARK``.
    """
    lines = []
    for factor, mark in [(f, "") for f in reference] + [(f, f" ({KNOWN_MARK})") for f in known]:
        names = factor.get_factor_names()
        kind = type(owner(factor))
        doc = (kind.__doc__ or "").strip().splitlines()
        shown = ", ".join(names[:12]) + (", ..." if len(names) > 12 else "")
        lines.append(f"- {kind.__name__}{mark}: {doc[0] if doc else 'no description'} ({len(names)} outputs: {shown})")
    return "\n".join(lines)


def trial_id(number: int, slot: int) -> str:
    """The id of the trial of round ``number`` in ``slot`` (from 0): ``T<round>-<slot from 1, two digits>``."""
    return f"T{number}-{slot + 1:02d}"


def named(records: list[TrialRecord]) -> list[tuple[str, TrialRecord]]:
    """``records`` with their ids; a round's records are in slot order, as the ledger holds them."""
    slots: dict[int, int] = {}
    pairs = []
    for r in records:
        slots[r.round] = slots.get(r.round, -1) + 1
        pairs.append((trial_id(r.round, slots[r.round]), r))
    return pairs


def history(
    trials: list[tuple[str, TrialRecord]], statistics: bool = True, earlier: list[tuple[str, TrialRecord]] = (),
) -> str:
    """One line per trial (``named``). Without ``statistics`` no metric is shown and the reasons' decimals are masked.

    ``earlier`` holds the run's trials (``named``), by which a trial's parent
    is shown with its residual t, the measure its child is compared on.
    """
    by_id = dict(earlier)
    lines = []
    for trial, r in trials:
        line = (
            f"- {trial}, round {r.round}, {_direction(r, by_id if statistics else {})}: "
            f"{r.candidate.get('description', '(unparsed)')} -> {outcome(r)}"
        )
        if r.evaluated:
            line += f"; {direction_text(r.metrics['ic_mean'])}"
        if statistics and r.evaluated:
            m = r.metrics
            line += (
                f"; training IC {_num(m['ic_mean'], 4)} (t {_num(m['ic_t'], 2)}), "
                f"residual IC {_num(m['partial_ic_mean'], 4)} (t {_num(m['partial_ic_t'], 2)})"
            )
            share = m.get("pool_share", float("nan"))
            if share is not None and math.isfinite(share) and share > 0:
                line += f", {share:.0%} explained by the pool"
        if not statistics and r.reasons:
            line += "; " + _DECIMAL.sub("#", " ".join(r.reasons))
        lines.append(line)
    return "\n".join(lines)


def earlier_summary(records: list[TrialRecord], mechanism_types: tuple[str, ...] = ()) -> str:
    """The trials of ``records`` counted by mechanism type: trials, evaluated, accepted and each kind of refusal.

    The types come in the order of ``mechanism_types``, then any other in
    name order; a type without trials is left out. A refusal is counted by
    the stage that refused it (parse, gate, review inconsistent, compile) and,
    at admission, by the kind of its reason (``_admission.reason_kind``).
    """
    if not records:
        return ""
    by_type: dict[str, list[TrialRecord]] = {}
    for r in records:
        by_type.setdefault((r.direction or {}).get("mechanism_type", "?"), []).append(r)
    order = [t for t in mechanism_types if t in by_type] + sorted(set(by_type) - set(mechanism_types))
    rounds = sorted({r.round for r in records})
    lines = [f"The trials of rounds {rounds[0]} to {rounds[-1]}, by mechanism type:"]
    for kind in order:
        trials = by_type[kind]
        refused: dict[str, int] = {}
        for r in trials:
            if r.status != "accepted":
                label = _refusal(r)
                refused[label] = refused.get(label, 0) + 1
        line = (
            f"- {kind}: {len(trials)} trials, {sum(r.evaluated for r in trials)} evaluated, "
            f"{sum(r.status == 'accepted' for r in trials)} accepted"
        )
        if refused:
            line += "; refused: " + ", ".join(f"{label} {count}" for label, count in refused.items())
        lines.append(line)
    return "\n".join(lines)


#: Groups ``themes`` lists, the largest first; the rest are counted.
THEMES_LISTED = 30


def theme_groups(names: list[str], matrix, threshold: float) -> list[list[str]]:
    """The tried trials ``names`` in groups, by the matrix of their correlations (rows and columns in ``names``' order).

    The trials are taken in the order they were tried, by their ids
    (``trial_id``: by round, then by slot), whatever the order of ``names``.
    Each joins the group of the earlier trial it is most correlated with,
    when that correlation is at least ``threshold`` in absolute value, and
    opens a group of its own otherwise. The groups are in the order they
    were opened, each in the order its trials were tried.
    """
    order = sorted(range(len(names)), key=lambda k: _trial_order(names[k]))
    group_of: dict[int, int] = {}
    groups: list[list[str]] = []
    for position, j in enumerate(order):
        earlier = [(abs(float(matrix[i][j])), i) for i in order[:position]]
        best = max(((c, i) for c, i in earlier if math.isfinite(c)), default=(0.0, -1))
        if best[0] >= threshold:
            group_of[j] = group_of[best[1]]
            groups[group_of[j]].append(names[j])
        else:
            group_of[j] = len(groups)
            groups.append([names[j]])
    return groups


def _trial_order(trial: str) -> tuple[int, int]:
    """The round and the slot of a trial id (``trial_id``)."""
    number, slot = _TRIAL_ID.fullmatch(trial).groups()
    return int(number), int(slot)



def themes(
    trials: list[tuple[str, TrialRecord]], names: list[str], matrix, threshold: float, sent_back: bool = False,
) -> str:
    """The evaluated trials (of ``named`` ``trials``) grouped by what their factors measure (``theme_groups``).

    ``names`` and ``matrix`` are the tried trials and their correlations
    (``Evaluator.tried``). Every group of two trials or more and every group
    holding an accepted factor is listed, the largest first and at most
    ``THEMES_LISTED``: the trial that opened it, its trials, how many were
    accepted, its largest residual t in its trial's direction, the last
    round it was tried in and its refusals by kind; the other groups are
    counted. With ``sent_back`` the planner is told that a new direction's
    candidate falling in a group is sent back to be rewritten.
    """
    if not names:
        return ""
    by_id = dict(trials)
    groups = [[t for t in group if t in by_id] for group in theme_groups(names, matrix, threshold)]
    groups = [g for g in groups if g]
    shown = [g for g in groups if len(g) > 1 or any(by_id[t].status == "accepted" for t in g)]
    shown.sort(key=lambda g: -len(g))
    lines = [
        f"Each evaluated trial is grouped with the earlier trial its factor is most correlated with (the mean "
        f"daily correlation on a sample of the training dates), when that correlation is at least {threshold:g} "
        f"in absolute value; a trial without one opens a group. A trial of a group measures what the trial that "
        f"opened it measures, whatever its hypothesis says. A new research direction is a mechanism no group "
        f"measures" + (": a new direction's candidate that falls in a group is sent back to be rewritten." if sent_back else ".")
    ]
    for group in shown[:THEMES_LISTED]:
        records = [by_id[t] for t in group]
        first = records[0]
        accepted = [t for t in group if by_id[t].status == "accepted"]
        best_t, best_id = max(
            ((direction(r.metrics["ic_mean"]) * r.metrics["partial_ic_t"], t) for t, r in zip(group, records)
             if math.isfinite(r.metrics["partial_ic_t"])),
            default=(float("nan"), ""),
        )
        refused: dict[str, int] = {}
        for r in records:
            if r.status != "accepted":
                refused[_refusal(r)] = refused.get(_refusal(r), 0) + 1
        line = (
            f"- opened by {group[0]} ({first.candidate.get('description', '')}): {len(group)} trial(s) "
            f"({', '.join(group)}), {len(accepted)} accepted{' (' + ', '.join(accepted) + ')' if accepted else ''}"
        )
        if best_id:
            line += f"; largest residual t in its direction {_num(best_t, 2)} ({best_id})"
        line += f"; last tried in round {max(r.round for r in records)}"
        if refused:
            line += "; refused: " + ", ".join(f"{label} {count}" for label, count in refused.items())
        lines.append(line)
    if len(shown) > THEMES_LISTED:
        lines.append(f"- {len(shown) - THEMES_LISTED} smaller group(s) of several trials, not listed")
    if len(groups) > len(shown):
        lines.append(f"- {len(groups) - len(shown)} group(s) of a single refused trial, not listed")
    return "\n".join(lines)


def _refusal(r: TrialRecord) -> str:
    """The kind of refusal of a trial that was not accepted."""
    if r.stage == "admission":
        return f"admission ({reason_kind(r.reasons[0]) if r.reasons else 'other'})"
    return {"review": "review inconsistent", "evaluate": "compile"}.get(r.stage, r.stage)


def last_round(trials: list[tuple[str, TrialRecord]], earlier: list[tuple[str, TrialRecord]] = ()) -> str:
    """Every candidate of a round (``named``) with its outcome, its statistics and why it was refused.

    ``earlier`` holds the run's trials (``named``), by which a parent is
    shown with its residual t and an accepted factor is named by its id.
    """
    by_id = dict(earlier)
    blocks = []
    for i, (trial, r) in enumerate(trials, 1):
        lines = [
            f"{i}. {trial} [{_direction(r, by_id)}] {r.candidate.get('description', '(unparsed)')}",
            f"   hypothesis: {r.candidate.get('hypothesis', '')}",
            f"   outcome: {outcome(r)}",
        ]
        blocks.append("\n".join(lines + _statistics(r, earlier)))
    return "\n".join(blocks)


def parent(trial: str, r: TrialRecord, earlier: list[tuple[str, TrialRecord]] = (), lineage: Lineage | None = None) -> str:
    """The trial a research direction builds on, as its proposer reads it: what it computed, its code, statistics and reasons.

    With its ``lineage`` the block ends with how the trials built on it did.
    """
    code = "\n".join("     " + line for line in r.candidate.get("code", "").splitlines())
    lines = [
        f"{trial}, round {r.round}, {r.candidate.get('mechanism_type', '?')}: {r.candidate.get('description', '')}",
        f"   hypothesis: {r.candidate.get('hypothesis', '')}",
        f"   outcome: {outcome(r)}",
        "   code:", code,
    ]
    return "\n".join(lines + _statistics(r, earlier) + ([f"   {lineage.text()}"] if lineage is not None else []))


def _statistics(r: TrialRecord, earlier: list[tuple[str, TrialRecord]] = ()) -> list[str]:
    """A trial's training statistics, what the pool explains of it, its factor direction and why it was refused, as lines.

    An accepted factor of ``earlier`` (``named``) is named by its id.
    """
    lines = []
    if r.evaluated:
        m = r.metrics
        nearest = (
            f"nearest existing factor {m['nearest']}, correlation {_num(m['nearest_correlation'], 2)}"
            if m["nearest"] else "no existing factor to compare with"
        )
        if m["nearest_known"]:
            nearest += (
                f"; nearest known factor {m['nearest_known']}, "
                f"sampled correlation {_num(m['nearest_known_correlation'], 2)}"
            )
        lines.append(
            f"   training IC {_num(m['ic_mean'], 4)} (Newey-West t {_num(m['ic_t'], 2)}); "
            f"residual IC {_num(m['partial_ic_mean'], 4)} (Newey-West t {_num(m['partial_ic_t'], 2)}); {nearest}"
        )
        overlap = _overlap(m, {accepted_label(e): t for t, e in earlier if e.status == "accepted"})
        if overlap:
            lines.append(overlap)
        order = _decile_order(m.get("decile_returns") or [])
        if order:
            lines.append(
                f"   the deciles' returns ranked from 1 (lowest) to 10, lowest factor values first: {order} "
                f"(rising in the factor's direction is 1 to 10, or 10 to 1 for direction -1)"
            )
        lines.append(f"   {direction_text(m['ic_mean'])}")
    if r.reasons:
        lines.append("   reasons: " + " | ".join(r.reasons))
    return lines


def accepted(records: list[TrialRecord]) -> str:
    """The accepted factors as the distiller reads them: name, mechanism type, hypothesis and direction, no statistic."""
    return "\n".join(
        f"- {accepted_label(r)} [{r.candidate['mechanism_type']}]: {r.candidate['hypothesis']}; "
        f"{direction_text(r.metrics['ic_mean'])}"
        for r in records if r.status == "accepted"
    )


def best(
    trials: list[tuple[str, TrialRecord]], contributions: dict[str, Contribution], lineage: dict[str, Lineage] | None = None,
) -> str:
    """The accepted factors (of ``named`` trials): id, name, mechanism, code, and marginal weight and residual IC in the accepted pool.

    With ``lineage`` (by trial id, ``_lineage.lineages``) each also carries its Lineage.
    """
    blocks = []
    for trial, r in trials:
        if r.status != "accepted":
            continue
        label = accepted_label(r)
        part = contributions.get(label)
        code = "\n".join("     " + line for line in r.candidate["code"].splitlines())
        lines = [
            f"- {trial}: {label} [{r.candidate['mechanism_type']}]: {r.candidate['hypothesis']}",
            f"  {direction_text(r.metrics['ic_mean'])}", "  code:", code,
        ]
        if part is not None:
            lines.append(
                f"  in the accepted pool: marginal weight {_num(part.weight, 3)}, residual IC {_num(part.residual_ic, 4)}"
            )
        if lineage and trial in lineage:
            lines.append(f"  {lineage[trial].text()}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def direction_text(ic_mean: float) -> str:
    """The factor direction the training IC shows, in words."""
    sign = direction(ic_mean)
    if sign == 0:
        return "no factor direction (the training IC is zero or not a number)"
    return f"factor direction {sign:+d} (a higher value went with a {'higher' if sign > 0 else 'lower'} return)"


def _decile_order(returns: list[float]) -> str:
    """The rank of each decile's mean return among the deciles, lowest factor values first, or "" when one is missing.

    The ranks show the shape (rising, U-shaped, broken at one end) that the
    monotonicity check reads, and no return.
    """
    if len(returns) != 10 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in returns):
        return ""
    order = sorted(range(10), key=lambda i: returns[i])
    return " ".join(str(order.index(i) + 1) for i in range(10))


def _overlap(metrics: dict, ids: dict[str, str]) -> str:
    """What the pool explains of a candidate and the pool factors most correlated with it (accepted ones by ``ids``), or ""."""
    share, overlaps = metrics.get("pool_share"), metrics.get("overlaps") or []
    if share is None or not math.isfinite(share) or not overlaps:
        return ""
    named = ", ".join(f"{ids.get(name, name)} {_num(correlation, 2)}" for name, correlation in overlaps)
    return (
        f"   {max(share, 0):.0%} of its daily cross-section is explained by the reference set and the accepted "
        f"factors (the part the residual IC leaves out); most correlated: {named}"
    )


def _direction(r: TrialRecord, by_id: dict[str, TrialRecord] | None = None) -> str:
    """The trial's mechanism type and action, with its parent and, from ``by_id``, the parent's residual t."""
    d = r.direction or {}
    text = f"{d.get('mechanism_type', '?')}, {d.get('action', '?')}"
    if not d.get("parent"):
        return text
    earlier = (by_id or {}).get(d["parent"])
    if earlier is None or not earlier.evaluated:
        return f"{text} of {d['parent']}"
    return f"{text} of {d['parent']} (whose residual t was {_num(earlier.metrics['partial_ic_t'], 2)})"


def outcome(r: TrialRecord) -> str:
    """A trial's outcome in words: accepted, or its status at the stage that decided it."""
    return "accepted" if r.status == "accepted" else f"{r.status} at {r.stage}"



def _num(value: float, digits: int) -> str:
    return f"{value:+.{digits}f}" if math.isfinite(value) else "n/a"
