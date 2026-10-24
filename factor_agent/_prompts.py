"""Prompt templates. They describe the data and the vocabulary, never dates or results."""

PROPOSE = """\
You are a quantitative researcher. Propose {n} daily cross-sectional factors for US common stocks,
each from a stated economic mechanism, to predict the open-to-open return over the next five
trading days.

Write each factor as Python code over KunQuant operators. The code may assign variables, define
functions, and use for/while/if and `with Builder(loop):` blocks for window loops. It must not use
import, class, attribute access, file or network access, or names outside the vocabulary below.
It must end by assigning exactly one operator to `alpha`. Arithmetic operators (+, -, *, /) and
comparisons work on operators; a comparison is only a Select condition.

Inputs (adjusted daily bars): {inputs}
Operators:
{operators}
Parameter values (windows, lags and constants must be one of these; constants may be negative):
{parameters}
Builtins: {builtins}

Answer with a JSON array of {n} objects, each with exactly these fields:
- "hypothesis": the economic mechanism, in one or two sentences
- "mechanism_type": one of {mechanism_types}
- "expected_sign": 1 if a higher value predicts a higher return, -1 if lower
- "horizon_rationale": why the effect plays out over about five trading days
- "reference": the literature the mechanism comes from, or why there is none
- "description": what the code computes, in words
- "code": the Python code
"""

RETRY = """\
The candidates below were refused before evaluation, for the reasons given under each. Fix each one
so that it passes: correct the code, or the fields, so that it still expresses its stated mechanism.
If a candidate duplicates an earlier one, propose a different mechanism of the same type instead.

{failures}

The vocabulary is unchanged:
Inputs (adjusted daily bars): {inputs}
Operators:
{operators}
Parameter values (windows, lags and constants must be one of these; constants may be negative):
{parameters}
Builtins: {builtins}

Answer with a JSON array of {n} objects, one per candidate above and in the same order, each with
exactly the fields of the original request: "hypothesis", "mechanism_type", "expected_sign",
"horizon_rationale", "reference", "description", "code".
"""

FAILURE = """\
Candidate {number}:
{item}
Refused because:
{reasons}
"""
