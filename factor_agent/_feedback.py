"""The context the planner and the proposer read: what is known, assembled from the ledger.

It shows only training-segment statistics: each candidate's IC and its
Newey-West t, its residual (partial) IC, the nearest existing factor and the
correlation with it, the rejection reasons, and each accepted factor's
marginal weight and residual IC in the accepted pool. It never shows a
return, a date, or advice to change windows or parameters. The first round
has no history; it sees the data, the seed directions and the existing
factor families by name and description, without any statistic.
"""

import math
import re

from factor_agent._admission import direction
from factor_agent.evaluator import Contribution
from factor_agent.ledger import TrialRecord
from factor_agent.zoo import owner

_DECIMAL = re.compile(r"[-+]?\d+\.\d+")


def accepted_source(number: int) -> str:
    """The zoo source of a factor accepted in round ``number``."""
    return f"accepted in round {number}"


def accepted_label(record: TrialRecord) -> str:
    """The name an accepted factor has in the zoo, the pool and the prompts (``ZooEntry.label``)."""
    return f"{record.candidate['description']} ({accepted_source(record.round)})"


def families(reference) -> str:
    """One line per reference factor: its owner's class, what its docstring says it is, and its outputs.

    The owner (``zoo.owner``) is the factor a wrapper such as ``NeutralizedFactor`` shows.
    """
    lines = []
    for factor in reference:
        names = factor.get_factor_names()
        kind = type(owner(factor))
        doc = (kind.__doc__ or "").strip().splitlines()
        shown = ", ".join(names[:12]) + (", ..." if len(names) > 12 else "")
        lines.append(f"- {kind.__name__}: {doc[0] if doc else 'no description'} ({len(names)} outputs: {shown})")
    return "\n".join(lines)


def history(records: list[TrialRecord], statistics: bool = True) -> str:
    """One line per trial. Without ``statistics`` no metric is shown and the reasons' decimals are masked."""
    lines = []
    for r in records:
        line = f"- round {r.round}, {_direction(r)}: {r.candidate.get('description', '(unparsed)')} -> {_outcome(r)}"
        if r.evaluated:
            line += f"; {direction_text(r.metrics['ic_mean'])}"
        if statistics and r.evaluated:
            m = r.metrics
            line += f"; training IC {_num(m['ic_mean'], 4)}, residual IC {_num(m['partial_ic_mean'], 4)}"
        if not statistics and r.reasons:
            line += "; " + _DECIMAL.sub("#", " ".join(r.reasons))
        lines.append(line)
    return "\n".join(lines)


def last_round(records: list[TrialRecord]) -> str:
    """Every candidate of a round with its outcome, its statistics and why it was refused."""
    blocks = []
    for i, r in enumerate(records, 1):
        lines = [
            f"{i}. [{_direction(r)}] {r.candidate.get('description', '(unparsed)')}",
            f"   hypothesis: {r.candidate.get('hypothesis', '')}",
            f"   outcome: {_outcome(r)}",
        ]
        if r.evaluated:
            m = r.metrics
            nearest = (
                f"nearest existing factor {m['nearest']}, correlation {_num(m['nearest_correlation'], 2)}"
                if m["nearest"] else "no existing factor to compare with"
            )
            lines.append(
                f"   training IC {_num(m['ic_mean'], 4)} (Newey-West t {_num(m['ic_t'], 2)}); "
                f"residual IC {_num(m['partial_ic_mean'], 4)} (Newey-West t {_num(m['partial_ic_t'], 2)}); {nearest}"
            )
            lines.append(f"   {direction_text(m['ic_mean'])}")
        if r.reasons:
            lines.append("   reasons: " + " | ".join(r.reasons))
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def best(records: list[TrialRecord], contributions: dict[str, Contribution]) -> str:
    """The accepted factors: name, mechanism, code, and marginal weight and residual IC in the accepted pool."""
    blocks = []
    for r in records:
        if r.status != "accepted":
            continue
        label = accepted_label(r)
        part = contributions.get(label)
        code = "\n".join("     " + line for line in r.candidate["code"].splitlines())
        lines = [
            f"- {label} [{r.candidate['mechanism_type']}]: {r.candidate['hypothesis']}",
            f"  {direction_text(r.metrics['ic_mean'])}", "  code:", code,
        ]
        if part is not None:
            lines.append(
                f"  in the accepted pool: marginal weight {_num(part.weight, 3)}, residual IC {_num(part.residual_ic, 4)}"
            )
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def direction_text(ic_mean: float) -> str:
    """The factor direction the training IC shows, in words."""
    sign = direction(ic_mean)
    if sign == 0:
        return "no factor direction (the training IC is zero or not a number)"
    return f"factor direction {sign:+d} (a higher value went with a {'higher' if sign > 0 else 'lower'} return)"


def _direction(r: TrialRecord) -> str:
    d = r.direction or {}
    text = f"{d.get('mechanism_type', '?')}, {d.get('action', '?')}"
    return f"{text} of {d['parent']}" if d.get("parent") else text


def _outcome(r: TrialRecord) -> str:
    return "accepted" if r.status == "accepted" else f"{r.status} at {r.stage}"


def _num(value: float, digits: int) -> str:
    return f"{value:+.{digits}f}" if math.isfinite(value) else "n/a"
