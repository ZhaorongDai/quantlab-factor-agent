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

**Scoring**:
How a factor's raw values become the values that are judged: ranked across the market each day, then
optionally neutralised and kept on the members of the day.
_Avoid_: normalisation, preprocessing

**Run snapshot**:
The copies, with their hashes, of the files a run searched with (field and operator files), and the
commits it ran on, kept in its run directory; a resume may not change the files, and a replay or
Acceptance reads the copies.
_Avoid_: checkpoint, cache, frozen config
