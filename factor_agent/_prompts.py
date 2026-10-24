"""Prompt templates. They describe the data, the vocabulary and the training-segment feedback, never dates or returns.

Each prompt that asks a role to answer about a set of things (the quota,
the directions, the refused candidates, the candidates to judge) ends with
them as one fenced JSON block.
"""

DATA = """\
The data: adjusted daily bars of every US common stock. A factor is computed on every stock, ranked
across the whole market each day, neutralised against industry and the logarithm of market cap,
and scored on the members of the research universe that day against the open-to-open return over
the next five trading days. Only a training period is ever evaluated; its dates are not disclosed."""

VOCABULARY = """\
Inputs (adjusted daily bars): {inputs}
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
- "horizon_rationale": why the effect plays out over about five trading days
- "reference": the literature the mechanism comes from, or why there is none
- "description": what the code computes, in words
- "code": the Python code"""

PLAN = """\
You are the research planner of a search for daily cross-sectional factors of US common stocks,
each from a stated economic mechanism. Plan the directions of this round.

Round: {round}

{context}

Plan exactly the number of directions given below for each mechanism type. Each direction is one
of these actions:
- "new": a mechanism not tried before;
- "mutation": a mechanism orthogonal to one that works (an accepted factor named in "parent"),
  exploiting a different economic channel;
- "refinement": a different expression of the same mechanism as an accepted factor named in
  "parent"; only changing its windows or parameters is refused.
"parent" is an accepted factor's name exactly as listed under the current best mechanisms, or ""
for a new direction.

Answer with a JSON array of objects with the fields "mechanism_type", "action", "parent" and
"direction" (the idea to pursue, in one or two sentences). Directions per mechanism type:
```json
{quota}
```
"""

PROPOSE = """\
You are a quantitative researcher. Write one daily cross-sectional factor for US common stocks for
each research direction below, from a stated economic mechanism, to predict the open-to-open return
over the next five trading days.

Round: {round}

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
You are the reviewer of proposed stock factors. For each candidate below judge, strictly, whether
the hypothesis, the description and the code agree: the description says what the code computes,
and the code computes what the hypothesis says drives returns, with the expected sign. Do not
rewrite or improve any candidate; only judge it. A candidate whose code does something its text
does not say, or leaves out what its hypothesis needs, is inconsistent.

Round: {round}

Answer with a JSON array of objects, one per candidate, each with the fields "number" (the
candidate's), "consistent" (true or false) and "reason" (one sentence).

The candidates:
```json
{candidates}
```
"""

DISTILL = """\
You keep the memory of a search for daily cross-sectional stock factors. Below is every trial so
far with its outcome. Distil it into a short memory for the researchers who plan and write the
next rounds, in two parts:
- "Working mechanism patterns": the kinds of mechanism and construction that were accepted;
- "Failure constraints": what keeps being refused and why, as rules to follow.
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
)
