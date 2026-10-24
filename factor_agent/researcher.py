"""The Researcher: the LLM roles of a round, from prompt to checked candidates."""

import json
import re
from dataclasses import dataclass

from factor_agent import _prompts
from factor_agent.candidate import MECHANISM_TYPES, Candidate
from factor_agent.gate import Gate, Passed
from factor_agent.gate.vocabulary import describe
from factor_agent.llm.base import LLMBackend, Request

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class Attempt:
    """One refused version of a candidate: what the proposer wrote, where it failed and why.

    ``stage`` is ``"parse"`` (not a valid candidate) or ``"gate"``.
    """

    item: object
    stage: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Proposal:
    """One candidate slot of a round after its retries.

    ``checked`` is set when the last version passed the Gate; otherwise the
    slot is a failed trial and the last of ``attempts`` says why. ``item``
    is the last version the proposer wrote, ``candidate`` its parsed form
    (None if it did not parse), and ``attempts`` every refused version in order.
    """

    item: object
    candidate: Candidate | None
    checked: Passed | None
    attempts: tuple[Attempt, ...]


class Researcher:
    """Ask the proposer for candidates, gate them and send the refused ones back to be fixed.

    Parameters
    ----------
    proposer : LLMBackend
        The proposer role's backend.
    model : str
        The model the proposer requests.
    """

    def __init__(self, proposer: LLMBackend, model: str):
        self.proposer = proposer
        self.model = model

    def propose(self, n: int, gate: Gate, retries: int) -> list[Proposal]:
        """Ask for ``n`` candidates and return one ``Proposal`` per item of the reply.

        Every candidate the parser or ``gate`` refuses is sent back with its
        reasons, all of a round's in one request, up to ``retries`` times.

        Raises
        ------
        ValueError
            If the first reply holds no JSON array.
        """
        prompt = _prompts.PROPOSE.format(n=n, mechanism_types=", ".join(MECHANISM_TYPES), **describe())
        items = _json_array(self._ask(prompt))
        attempts: list[list[Attempt]] = [[] for _ in items]
        proposals: list[Proposal | None] = [None] * len(items)
        pending = list(range(len(items)))
        for retry in range(retries + 1):
            for slot in pending:
                proposals[slot] = self._check(items[slot], gate, attempts[slot])
            pending = [slot for slot in pending if proposals[slot].checked is None]
            if not pending or retry == retries:
                break
            fixed = self._ask_to_fix([(items[slot], attempts[slot][-1]) for slot in pending])
            for slot, item in zip(pending, fixed):
                items[slot] = item
        return proposals

    def _check(self, item: object, gate: Gate, attempts: list[Attempt]) -> Proposal:
        """Parse and gate one item; a refusal is appended to ``attempts``."""
        if isinstance(item, _Missing):
            attempts.append(Attempt(item.previous, "parse", (item.reason,)))
            return Proposal(item.previous, None, None, tuple(attempts))
        try:
            candidate = Candidate.from_dict(item)
        except ValueError as exc:
            attempts.append(Attempt(item, "parse", (str(exc),)))
            return Proposal(item, None, None, tuple(attempts))
        checked = gate.check(candidate)
        if isinstance(checked, Passed):
            return Proposal(item, candidate, checked, tuple(attempts))
        attempts.append(Attempt(item, "gate", checked.reasons))
        return Proposal(item, candidate, None, tuple(attempts))

    def _ask_to_fix(self, failures: list[tuple[object, Attempt]]) -> list:
        """Send the refused items back; return one fixed item per failure, in order."""
        items = [item.previous if isinstance(item, _Missing) else item for item, _ in failures]
        blocks = [
            _prompts.FAILURE.format(
                number=i + 1,
                item=json.dumps(item, indent=2),
                reasons="\n".join(f"- {reason}" for reason in attempt.reasons),
            )
            for i, (item, (_, attempt)) in enumerate(zip(items, failures))
        ]
        text = self._ask(_prompts.RETRY.format(failures="\n".join(blocks), n=len(failures), **describe()))
        try:
            fixed = _json_array(text)
        except ValueError as exc:
            return [_Missing(str(exc), item) for item in items]
        short = f"the reply to the fix request had {len(fixed)} items for {len(failures)} candidates"
        return [fixed[i] if i < len(fixed) else _Missing(short, item) for i, item in enumerate(items)]

    def _ask(self, prompt: str) -> str:
        """Send ``prompt`` to the proposer and return the reply's text."""
        return self.proposer.complete(Request(role="proposer", model=self.model, prompt=prompt)).text


@dataclass(frozen=True)
class _Missing:
    """No usable item came back for a candidate the proposer was asked to fix; ``previous`` is its last version."""

    reason: str
    previous: object


def _json_array(text: str) -> list:
    """Return the JSON array in ``text``: a fenced block if there is one, else the whole text."""
    fenced = _FENCE.search(text)
    body = fenced.group(1) if fenced else text
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"the proposer's reply is not JSON ({exc}): {text[:200]!r}") from exc
    if not isinstance(value, list):
        raise ValueError(f"the proposer's reply is a JSON {type(value).__name__}, not an array")
    return value
