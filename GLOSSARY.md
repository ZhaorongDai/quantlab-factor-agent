# quantlab-factor-agent

An LLM proposes daily stock factors from stated economic mechanisms; deterministic checks and a
training-period evaluation decide which ones add to the existing factor set.

## Language

**Input family**:
A group of data a candidate may read, opened or closed as a whole for a campaign: price and volume
(daily adjusted bars), or fundamentals (as-reported financial statement items and daily valuation
metrics).
_Avoid_: data source, dataset, data type

**Field**:
One named input of an input family that a candidate's code reads, with its unit and description.
_Avoid_: column, variable, feature

**Period lag**:
A field's value as of the k-th earlier reporting period (for financial statements, the k-th earlier
report), counted in periods, not in trading days.
_Avoid_: quarter lag, lag, BackRef, year-ago value

**Vocabulary**:
Everything a campaign's candidates may write with: the fields of its input families, the operators,
and the parameter values those operators accept.
_Avoid_: whitelist, registry, DSL

**Fix**:
A known way the operator library refuses a candidate's code, recognised by the text of its error,
with how to write it instead; it is added to the refusal the proposer is asked to fix.
_Avoid_: error library, example library, patch

**Factor direction**:
Which way a factor predicts: +1 when a higher value went with a higher return, -1 when with a lower
one. It is the sign of the factor's training IC, read from the data, never stated in advance.
_Avoid_: expected sign, sign of the hypothesis

**Research direction**:
One planned slot of a round: a mechanism type, an action (a new mechanism, or a mutation or refinement
of an accepted factor) and the idea the proposer is to write a candidate for.
_Avoid_: plan item, slot (for the idea itself)

**Scoring**:
How a factor's raw values become the values that are judged: ranked across the market each day, then
optionally neutralised and kept on the members of the day.
_Avoid_: normalisation, preprocessing

**Run snapshot**:
The copies, with their hashes, of the files a run searched with (field and operator files and the
replaced prompt templates), and the commits it ran on, kept in its run directory; a resume may not
change the files, and a replay or Acceptance reads the copies.
_Avoid_: checkpoint, cache, frozen config

**Memory**:
What the distiller writes every few rounds for the planner and the proposer: the mechanism patterns that
worked, the failures to avoid and the factor directions the data showed, in words. Each version is
written from the previous one and the trials since, and every version is kept.
_Avoid_: summary, notes

**Recent history**:
The trials of the last few rounds, one line each, as the planner reads them.
_Avoid_: full history, log

**Earlier summary**:
The trials before the recent history, counted by mechanism type and outcome rather than listed.
_Avoid_: old history, archive
