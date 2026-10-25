"""Prompt templates. They describe the data, the vocabulary and the training-segment feedback, never dates or returns.

Each role's prompt opens with the role, its task and who uses its answer,
holds the long inputs in the middle, each inside an XML tag (``<quota>``,
``<directions>``, ``<refused_candidates>``, ``<candidates>``, ``<previous_memory>``, ``<history>``,
and the context sections under their keys), and ends by restating the task
and the shape of the answer. A fenced ``json`` block holds only that shape,
the last thing in the prompt, so no input reads like an answer to copy.

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
Write each factor as Python code over KunQuant operators. The code runs in a sandbox that builds a
KunQuant graph from it, so it uses only Python's own constructs and the names listed in <vocabulary>:
it may assign variables, define functions, and use for/while/if and `with Builder(loop):` blocks for
window loops. The sandbox refuses import, class, attribute access and file or network access, because
the graph is all that is evaluated. The code ends by assigning one operator to `alpha`, the factor's
value. Arithmetic operators (+, -, *, /) and comparisons work on operators; a comparison serves only
as the condition of a Select.
A factor's value should be comparable across the cross-section, so write it as a pure number: a
ratio, a return, a rank or a z-score. An output that carries a unit (a price, a dollar amount, a share
count) ranks the cross-section partly by that level, and that ranking is what the scoring compares,
unless the mechanism is about that level, as a price impact per dollar traded is.
Leave a missing value missing: do not wrap the value in SetInfOrNanToValue to fill it. The scoring
skips a symbol without a value that day, so a missing value needs no fill; a filled value is ranked as
real data, so a symbol without the data is sorted as if its value were the fill and the factor partly
sorts the cross-section by whether the item is reported; and coverage is measured before any fill, so a fill does
not help a factor pass the coverage rule. Fill only when the mechanism says what a missing value means,
as an unreported R&D expense is zero, and state that reason in the description."""

FIELDS = """\
- "hypothesis": the economic mechanism that makes the factor predict the return, in one or two sentences
- "mechanism_type": the mechanism type of its research direction
- "horizon_rationale": why the effect plays out over about {horizon} bars
- "description": what the code computes, in words
- "code": the Python code, ending with the assignment to `alpha`"""

#: The shape of one candidate in the proposer's answer (braces doubled for ``str.format``).
_CANDIDATE_SKELETON = """\
```json
[
  {{
    "hypothesis": "<string>",
    "mechanism_type": "<string>",
    "horizon_rationale": "<string>",
    "description": "<string>",
    "code": "<string>"
  }}
]
```"""

PLAN = """\
You are the research planner of a search for cross-sectional factors on {market}, each built from a
stated economic mechanism. You plan this round's research directions; the proposer then writes one
candidate factor for each research direction you plan and decides how to compute it. So state each
research direction as the mechanism and the data it reads, and leave the formula, its windows and its
ratios to the proposer: a direction written as a known formula tends to come back as that textbook
factor, while a mechanism leaves room for a new way to measure it.

Round: {round}

<data>
{data}
</data>

{context}

The mechanism types:
<mechanism_types>
{mechanisms}
</mechanism_types>

Each research direction takes one of these actions:
- "new": a mechanism not tried before; its "parent" is "".
- "mutation": a mechanism orthogonal to an accepted factor, exploiting a different economic channel;
  its "parent" names that factor.
- "refinement": a different expression of the same mechanism as an accepted factor; its "parent"
  names that factor. Change how the mechanism is measured, because a version that changes only the
  windows or the parameters is refused.
A "parent" is written exactly as the accepted factor's name appears in <best>, because it is matched
against those names; a research direction whose parent matches none becomes a new one. Without a
<best> section no factor has been accepted yet, so every research direction is new.

<quota>
{quota}
</quota>

Now plan this round's research directions: for each mechanism type, as many as <quota> gives it.
Each research direction is an object with these fields:
- "mechanism_type": one of the mechanism types in <quota>
- "action": "new", "mutation" or "refinement"
- "parent": the accepted factor's name for a mutation or a refinement, "" for a new research direction
- "direction": the mechanism to pursue and the data it reads, in one or two sentences

Answer with a JSON array of research directions, one element per research direction, each with these
fields and no others:
```json
[
  {{
    "mechanism_type": "<string>",
    "action": "<new, mutation or refinement>",
    "parent": "<string>",
    "direction": "<string>"
  }}
]
```
"""

PROPOSE = """\
You are a quantitative researcher. You write one candidate factor for each research direction in
<directions>: a cross-sectional factor on {market}, built from a stated economic mechanism, that
predicts the return over the next {horizon} bars. Each candidate is checked by a static gate, judged
by a reviewer on whether its hypothesis, description and code agree, and then evaluated on a training
period.

Round: {round}

<data>
{data}
</data>

{context}

How to write the code:
{code_rules}

<vocabulary>
{vocabulary}
</vocabulary>

<directions>
{directions}
</directions>

Now write one candidate for each research direction in <directions> ({n} in all), in the same order.
Each candidate is an object with these fields:
{fields}

Answer with a JSON array of candidates, one per research direction ({n} in all). Element i is the
candidate for research direction i, because the candidates are matched to the research directions by
position; give each one these fields and no others:
""" + _CANDIDATE_SKELETON + "\n"

RETRY = """\
You are a quantitative researcher correcting candidate factors that were refused before evaluation.
Each entry in <refused_candidates> holds the research direction a candidate was written for
("direction"), the refused version ("refused_version"), why it was refused ("refused_because") and,
for some, how to write it instead ("how_to_write_it"). An entry the reviewer sent back also holds the
unit of the refused version's value ("output_unit"), such as "pure number" or "price"; a unit other
than a pure number is normalised, for example by dividing by a price, unless the mechanism is about
that level. Your corrected candidates
go through the same checks again, then to the reviewer and the evaluation on a training period.
The factors are on {market}; each predicts the return over the next {horizon} bars.

Round: {round}

How to write the code:
{code_rules}

<vocabulary>
{vocabulary}
</vocabulary>

<refused_candidates>
{failures}
</refused_candidates>

Now correct every entry in <refused_candidates> ({n} in all), in the same order. Change the code or the
fields so that the candidate passes and still expresses its mechanism and the idea of its research
direction. When the reason is that it duplicates an earlier candidate, write a different mechanism of
the same mechanism type instead.

Return each corrected candidate object itself, an object with these fields:
{fields}
Leave out "number", "direction", "output_unit", "refused_because" and "how_to_write_it", and write
the fields at the top level of the object rather than nested inside another object, because element i
of your array is read directly as the corrected candidate for entry i.

Answer with a JSON array of corrected candidates, one per entry ({n} in all). Element i answers entry
i of <refused_candidates>; give each one these fields and no others:
""" + _CANDIDATE_SKELETON + "\n"

REVIEW = """\
You are the reviewer of candidate factors on {market}, each meant to predict the return over the
next {horizon} bars. You judge, for each candidate in <candidates>, whether its hypothesis, its
description and its code agree; a candidate you find inconsistent is sent back once to be rewritten
with your reason and is dropped if it is still inconsistent, and one you find consistent is evaluated
on a training period.

Round: {round}

The data, with every field a candidate's code may read and what it holds:
<data>
{data}
</data>

Each candidate is listed with its "output_unit": the unit of its value, as the static gate infers it
from the units of the fields, such as "pure number" (a ratio, a return, a rank or a z-score), "price",
"money", "volume" or "1/dollars". It is "unknown" when the gate cannot tell; then judge the unit from
the code.

<candidates>
{candidates}
</candidates>

A candidate is inconsistent when any of these holds:
1. The code computes something the description does not say.
2. The description says something the code does not compute.
3. The code leaves out an input or a step the hypothesis needs.
4. What the code computes is not the quantity the hypothesis says drives the return.
5. The code does not use an input, a transformation or a quantity the hypothesis says the factor is
   built from. What the hypothesis mentions only as background, as a consequence or as a contrast
   does not count, because the factor need not be built from it. For example, a hypothesis that
   prices under-react to earnings surprises, with code that reads only price and volume, is
   inconsistent; a hypothesis that investors watch net income and overlook gross profitability,
   with code built from gross profit, is consistent, because net income is only the contrast.
6. The output has a unit other than a pure number ("output_unit"), and the hypothesis does not
   explain why the cross-section is ranked by that level, because such a value ranks it partly by
   price level or size. For example, the Amihud illiquidity, the absolute return per dollar traded
   (1/dollars), is consistent when its hypothesis is about the price impact per dollar traded; a
   slope of vwap described as a trend, without dividing by a price (price), is inconsistent.
7. The code fills missing values (SetInfOrNanToValue) and the description does not state the reason,
   what a missing value means in the mechanism, because a filled value is ranked as real data. For
   example, R&D expense filled with 0 is consistent when the description says an unreported R&D
   expense is zero; a ratio filled with 0 for no stated reason is inconsistent.
Otherwise it is consistent. Which way a factor predicts (its factor direction) is read from the data
afterwards, so a factor direction the text states, leaves out or gets the other way round is no reason
to find a candidate inconsistent. The length of the code is no reason to find a candidate consistent
or inconsistent: judge what the code computes. Your verdict is the whole answer, so judge each
candidate as it is written.

Now judge every candidate in <candidates>. Answer with a JSON array of verdicts, one per candidate:
element i is the verdict on candidate i, with that candidate's "number", a one-sentence "reason", and
then "consistent" (true or false). Give each verdict these fields and no others:
```json
[
  {{
    "number": <integer>,
    "reason": "<one sentence>",
    "consistent": <true or false>
  }}
]
```
"""

#: The parts of the memory, by heading, in order: what the distiller is asked for (``DISTILL``) and
#: what its reply is parsed by (``factor_agent.researcher``). A replaced ``distill`` template must name each.
MEMORY_PARTS = ("Working mechanism patterns", "Failure constraints", "Directional findings")
#: Items a part of the memory keeps at most.
MEMORY_ITEMS = 6

DISTILL = """\
You are the keeper of the memory of a search for cross-sectional factors on {market}; the factors
predict the return over the next {horizon} bars. The planner and the proposer of the next rounds read
your memory in their context, next to the latest feedback, to choose which mechanisms to pursue and
how to write them, so it carries the lessons that the list of trials is too long to show. You receive
the previous memory and the trials since it was written, and write the next memory from both.

Round: {round}

<previous_memory>
{memory}
</previous_memory>

<history>
{history}
</history>

""" + f"""Now fold the new trials in <history> into the previous memory, in three parts:
- {MEMORY_PARTS[0]}: the kinds of mechanism and construction that were accepted, each with
  why it worked.
- {MEMORY_PARTS[1]}: what keeps being refused, each as a rule to follow with its reason.
- {MEMORY_PARTS[2]}: which way the evaluated mechanisms worked, as the factor directions in
  <history> show them, such as "volume surges were followed by outperformance".
Keep what the new trials do not contradict, merge items that say the same thing, and drop a finding
the new trials overturned. Write at most {MEMORY_ITEMS} items per part, the most useful first; later items are cut. Describe every pattern in words, because the next rounds
should reason about mechanisms rather than tune to numbers; so the memory holds no numbers,
statistics or factor code.

Answer with the memory as plain text, without a code fence, in this shape:
```text
{MEMORY_PARTS[0]}
- <pattern>: <why it worked>
{MEMORY_PARTS[1]}
- <rule>: <why>
{MEMORY_PARTS[2]}
- <finding>
```
"""

#: The context sections in prompt order, by key and heading; an empty section is left out. Each is
#: shown under its heading inside the XML tag named by its key.
SECTIONS = (
    ("seed", "Seed research directions"),
    ("families", "Existing factor families (candidates must add to them, not rewrite them)"),
    ("memory", "Memory distilled from earlier rounds"),
    ("earlier_summary", "Earlier summary (the older trials, counted by mechanism type and outcome)"),
    ("recent_history", "Recent history (the trials of the last rounds, one line each)"),
    ("last_round", "Last round's feedback (training period only)"),
    ("best", "Current best mechanisms (the accepted factors)"),
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
    "review": (frozenset({"candidates", "data"}), frozenset({"round", "mechanisms"})),
    "distill": (frozenset({"memory", "history"}), frozenset({"round", "data", "mechanisms"})),
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
    if name == "distill":  # the distiller's reply is parsed by these headings, so it must be asked for them
        problems += [f"lacks the memory part heading {part!r}" for part in MEMORY_PARTS if part not in text]
    unknown = sorted(used - required - optional - _COMMON)
    if unknown:
        problems.append(
            f"uses the unknown placeholder(s) {['{' + p + '}' for p in unknown]}; it may use "
            f"{sorted(required | optional | _COMMON)}"
        )
    return problems
