"""The Researcher: the LLM roles of a round, from prompt to parsed candidates."""

import json
import re
from dataclasses import dataclass

from factor_agent import _prompts
from factor_agent.candidate import MECHANISM_TYPES, Candidate
from factor_agent.gate.vocabulary import BUILTINS, DERIVED_INPUTS, INPUT_COLUMNS, OPERATORS
from factor_agent.llm.base import LLMBackend, Request

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class Unparsed:
    """A reply item that is not a valid candidate, with the item and why."""

    item: object
    reason: str


class Researcher:
    """Ask the proposer for candidates and parse its reply.

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

    def propose(self, n: int) -> list[Candidate | Unparsed]:
        """Ask for ``n`` candidates; return one entry per item of the reply.

        Raises
        ------
        ValueError
            If the reply holds no JSON array.
        """
        prompt = _prompts.PROPOSE.format(
            n=n,
            inputs=", ".join([*INPUT_COLUMNS, *DERIVED_INPUTS]),
            operators=", ".join(sorted(OPERATORS)),
            builtins=", ".join(sorted(BUILTINS)),
            mechanism_types=", ".join(MECHANISM_TYPES),
        )
        reply = self.proposer.complete(Request(role="proposer", model=self.model, prompt=prompt))
        return [self._parse_item(item) for item in _json_array(reply.text)]

    @staticmethod
    def _parse_item(item: object) -> Candidate | Unparsed:
        try:
            return Candidate.from_dict(item)
        except ValueError as exc:
            return Unparsed(item, str(exc))


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
