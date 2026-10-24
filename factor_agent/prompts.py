"""Prompt templates. They describe the data, the vocabulary and the training-segment feedback, never dates or returns.

Each prompt that asks a role to answer about a set of things (the quota,
the directions, the refused candidates, the candidates to judge) ends with
them as one fenced JSON block.

The templates are built in (``TEMPLATES``) and any of them may be replaced
from a file (``Templates.load``). A template names what it is filled with
as ``{placeholder}``; ``PLACEHOLDERS`` lists, per template, the ones it must
use and the ones it may. Every template may use ``market`` (the config's
description of the data) and ``horizon`` (the bars the label spans), and
every one but ``data``, which is the same in every round, ``round``.
``Prompts`` fills a template with those and with the data
section, itself the ``data`` template filled with the market, the
described inputs of the vocabulary, the scoring's sentence and the horizon.
"""

import string
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from factor_agent.gate import _dimension
from factor_agent.gate.vocabulary import Vocabulary

DATA = """\
The data: {market}.
{scoring}
The label a factor is scored against is the return over the next {horizon} bars.
Only a training period is ever evaluated; its dates are not disclosed.

The inputs, by family:
{inputs}"""

VOCABULARY = """\
Inputs: {inputs}
Operators:
{operators}
Parameter values (windows, lags and constants must be one of these; constants may be negative):
{parameters}
Builtins: {builtins}"""

CODE_RULES = """\
Write each factor as Python code over KunQuant operators. The code may assign variables, define
functions, and use for/while/if and `with Builder(loop):` blocks for window loops. It must not use
import, class, attribute access, file or network access, or names outside the vocabulary below.
It must end by assigning exactly one operator to `alpha`. Arithmetic operators (+, -, *, /) and
comparisons work on operators; a comparison is only a Select condition."""

FIELDS = """\
- "hypothesis": the economic mechanism, in one or two sentences
- "mechanism_type": the direction's mechanism type
- "expected_sign": 1 if a higher value predicts a higher return, -1 if lower
- "horizon_rationale": why the effect plays out over about {horizon} bars
- "description": what the code computes, in words
- "code": the Python code"""

PLAN = """\
You are the research planner of a search for cross-sectional factors, each from a stated economic
mechanism, on {market}.
Plan the directions of this round.

Round: {round}

{data}

{context}

The mechanism types:
{mechanisms}

Plan exactly the number of directions given below for each mechanism type. Each direction is one
of these actions:
- "new": a mechanism not tried before;
- "mutation": a mechanism orthogonal to one that works (an accepted factor named in "parent"),
  exploiting a different economic channel;
- "refinement": a different expression of the same mechanism as an accepted factor named in
  "parent"; only changing its windows or parameters is refused;
- "reversal": a mechanism that predicts the direction the data showed for a refuted hypothesis
  named in "parent"; code that only negates, rescales or ranks the refuted factor is refused.
"parent" is an accepted factor's name exactly as listed under the current best mechanisms for a
mutation or a refinement, a refuted hypothesis's name exactly as listed under the refuted
hypotheses for a reversal, or "" for a new direction.

Answer with a JSON array of objects with the fields "mechanism_type", "action", "parent" and
"direction" (the idea to pursue, in one or two sentences). Directions per mechanism type:
```json
{quota}
```
"""

PROPOSE = """\
You are a quantitative researcher. Write one cross-sectional factor for each research direction
below, from a stated economic mechanism, on {market}.
Each factor predicts the return over the next {horizon} bars.

Round: {round}

{data}

{context}

{code_rules}

{vocabulary}

Answer with a JSON array of {n} objects, one per direction and in the same order, each with exactly
these fields:
{fields}

The directions:
```json
{directions}
```
"""

RETRY = """\
The candidates below were refused before evaluation, for the reasons given with each. Fix each one
so that it passes: correct the code, or the fields, so that it still expresses its stated mechanism
and follows its direction. If a candidate duplicates an earlier one, propose a different mechanism
of the same type instead.
The factors are on {market}.
Each predicts the return over the next {horizon} bars.

Round: {round}

{code_rules}

{vocabulary}

Answer with a JSON array of {n} objects, one per candidate below and in the same order, each with
exactly the fields of the original request:
{fields}

The refused candidates:
```json
{failures}
```
"""

REVIEW = """\
You are the reviewer of proposed factors on {market}.
Each is meant to predict the return over the next {horizon} bars.
For each candidate below judge, strictly, whether the hypothesis, the description and the code
agree: the description says what the code computes, and the code computes what the hypothesis
says drives returns, with the expected sign. Do not rewrite or improve any candidate; only judge
it. A candidate whose code does something its text does not say, or leaves out what its
hypothesis needs, is inconsistent.

Round: {round}

Answer with a JSON array of objects, one per candidate, each with the fields "number" (the
candidate's), "consistent" (true or false) and "reason" (one sentence).

The candidates:
```json
{candidates}
```
"""

DISTILL = """\
You keep the memory of a search for cross-sectional factors on {market}.
The factors predict the return over the next {horizon} bars.
Below is every trial so far with its outcome. Distil it into a short memory for the researchers
who plan and write the next rounds, in three parts:
- "Working mechanism patterns": the kinds of mechanism and construction that were accepted;
- "Failure constraints": what keeps being refused and why, as rules to follow;
- "Directional findings": the mechanisms the data showed working the other way round (a trial
  rejected with the opposite sign), said as the direction they do work in, such as "volume surges
  are followed by outperformance".
Write no numbers, no statistics and no factor code; describe patterns in words. Answer with the
memory as plain text.

Round: {round}

{history}
"""

#: The context sections in prompt order, by key and heading; an empty section is left out.
SECTIONS = (
    ("seed", "Seed research directions"),
    ("families", "Existing factor families (candidates must add to them, not rewrite them)"),
    ("memory", "Memory distilled from earlier rounds"),
    ("history", "All history (every earlier trial)"),
    ("last_round", "Last round's feedback (training period only)"),
    ("best", "Current best mechanisms (the accepted factors)"),
    ("refuted", "Refuted hypotheses (the training IC had the opposite sign to the one stated)"),
)

#: The built-in templates, by the name a config's ``prompts`` replaces them under.
TEMPLATES = {
    "data": DATA, "plan": PLAN, "propose": PROPOSE, "retry": RETRY, "review": REVIEW, "distill": DISTILL,
}

#: What every template may use.
_COMMON = frozenset({"market", "horizon"})

#: Per template, the placeholders it must use and the ones it may use besides ``_COMMON``.
PLACEHOLDERS = {
    "data": (frozenset({"market", "inputs", "scoring", "horizon"}), frozenset()),
    "plan": (frozenset({"data", "context", "quota"}), frozenset({"round", "mechanisms"})),
    "propose": (
        frozenset({"data", "context", "code_rules", "vocabulary", "fields", "n", "directions"}),
        frozenset({"round", "mechanisms"}),
    ),
    "retry": (
        frozenset({"code_rules", "vocabulary", "fields", "n", "failures"}), frozenset({"round", "data", "mechanisms"}),
    ),
    "review": (frozenset({"candidates"}), frozenset({"round", "data", "mechanisms"})),
    "distill": (frozenset({"history"}), frozenset({"round", "data", "mechanisms"})),
}


class PromptError(ValueError):
    """Replaced templates are wrong; the message lists every problem."""

    def __init__(self, problems: list[str]):
        self.problems = tuple(problems)
        super().__init__("the prompt templates have problems:\n" + "\n".join(f"- {p}" for p in problems))


class Templates:
    """The templates of a campaign: the built-in ones, any of them replaced from a file."""

    def __init__(self, texts: Mapping[str, str]):
        self._texts = {**TEMPLATES, **texts}

    @classmethod
    def load(cls, replaced: Mapping[str, str | Path]) -> "Templates":
        """The built-in templates with those of ``replaced`` (template name to file) read from their files.

        Raises
        ------
        PromptError
            Naming every unknown template, unreadable file, malformed template, and
            placeholder a template lacks or may not use.
        """
        problems, texts = [], {}
        for name, path in replaced.items():
            if name not in TEMPLATES:
                problems.append(f"{name}: not a template; the templates are {list(TEMPLATES)}")
                continue
            try:
                texts[name] = Path(path).read_text()
            except OSError as exc:
                problems.append(f"{name} ({path}): cannot be read ({exc})")
                continue
            problems += [f"{name} ({path}): {problem}" for problem in _placeholder_problems(name, texts[name])]
        if problems:
            raise PromptError(problems)
        return cls(texts)

    def format(self, name: str, **values) -> str:
        """Template ``name`` filled with ``values``; values it does not name are ignored."""
        return self._texts[name].format(**values)


@dataclass(frozen=True)
class Prompts:
    """What the Researcher's prompts say: the templates and what fills every one of them.

    ``market`` is the config's description of the data, ``horizon`` the
    bars the label spans, ``scoring`` the scoring's sentence
    (``Scoring.describe``), ``vocabulary`` what candidates write with and
    ``mechanisms`` the mechanism types with their one-line descriptions, in order.
    """

    templates: Templates
    market: str
    horizon: int
    scoring: str
    vocabulary: Vocabulary
    mechanisms: Mapping[str, str]

    @property
    def mechanism_types(self) -> tuple[str, ...]:
        """The names of the mechanism types, in order: what a round is planned over and a candidate belongs to."""
        return tuple(self.mechanisms)

    def render(self, name: str, **values) -> str:
        """Template ``name`` filled with ``values`` and every value common to the templates."""
        common = {"market": self.market, "horizon": self.horizon}
        common["data"] = self.templates.format(
            "data", **common, scoring=self.scoring, inputs=inputs(self.vocabulary),
        )
        common.update(
            mechanisms="\n".join(f"- {kind}: {text}" for kind, text in self.mechanisms.items()),
            code_rules=CODE_RULES, vocabulary=VOCABULARY.format(**self.vocabulary.prompt()),
            fields=FIELDS.format(horizon=self.horizon),
        )
        return self.templates.format(name, **common, **values)


def inputs(vocabulary: Vocabulary) -> str:
    """The fields of ``vocabulary`` by family, each with its unit (a derived field's follows from its formula)."""
    lines = []
    for family, description in vocabulary.families.items():
        lines.append(f"- {family}" + (f" ({description}):" if description else ":"))
        for field in vocabulary.fields.values():
            if field.family == family:
                unit = f" ({_dimension.name(field.dimension)})" if field.dimension is not None else ""
                lines.append(f"  - {field.name}{unit}: {field.description}")
    return "\n".join(lines)


def _placeholder_problems(name: str, text: str) -> list[str]:
    """What is wrong with the placeholders of template ``name``'s ``text``."""
    try:
        used = {field for _, field, _, _ in string.Formatter().parse(text) if field is not None}
    except ValueError as exc:
        return [f"is not a valid template ({exc}); a literal brace is written twice"]
    required, optional = PLACEHOLDERS[name]
    problems = [f"lacks the placeholder {{{p}}}" for p in sorted(required - used)]
    unknown = sorted(used - required - optional - _COMMON)
    if unknown:
        problems.append(
            f"uses the unknown placeholder(s) {['{' + p + '}' for p in unknown]}; it may use "
            f"{sorted(required | optional | _COMMON)}"
        )
    return problems
