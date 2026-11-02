# Gate, review and evaluation

A candidate passes three judges in turn: the Gate (deterministic checks on its code and
graph), the reviewer (an LLM's consistency check) and the Evaluator with admission
(statistics on the training segment). Only admission accepts.

## The Gate

`Gate.check(candidate, zoo)` returns `Passed` with the built graph and its statistics, or
`Rejected` with the reasons. In order:

1. **Syntax**: the code uses only the allowed syntax and the vocabulary's names, and
   assigns `alpha`.
2. **Build**: the code builds its KunQuant graph in a sandboxed subprocess, within
   `gate.timeout_seconds` and `gate.memory_mb`. The time counts from when the sandbox
   is ready, so a slow start on a loaded machine does not count against the candidate; a
   sandbox not ready within 60 s stops the campaign as a broken environment.
   A KunQuant refusal comes back with the operator file's matching fixes.
3. **Stages**: a candidate of an [intraday family](intraday.md) keeps its intraday and
   daily stages apart, each operator in its stages and each window and lag in its stage's
   set (see [the rules](intraday.md#the-rules-the-gate-enforces)).
4. **Constraints**: every parameter is in its set, the graph has at most `gate.max_nodes`
   nodes, at most `gate.max_free_parameter_ratio` hand-set constants per operator and at
   most `gate.max_base_fields` input fields; the output's unit is inferred from the
   operators' dimension rules.
5. **Originality**: the graph is compared with every zoo factor (reference, known and
   accepted) after canonicalisation: a rewrite of one, a parameter-only variant of one, or
   a subtree larger than `gate.max_common_subtree_nodes` shared with one is refused, naming
   it. A graph an earlier candidate of the run already had is refused too.

A refused candidate goes back to the proposer with the reasons, up to `round.retries`
times; one still refused fails at stage `gate`.

## The review

The reviewer reads each gated candidate's hypothesis, description and code with the data
section's field descriptions and the output unit the Gate inferred, and judges whether they
describe the same factor: the inputs and transformations the hypothesis says the factor is
built from included, a unit-carrying value justified by the hypothesis, a fill justified
by the mechanism. An inconsistent candidate is sent back once to be rewritten, with the
reviewer's reason; the rewrite passes the Gate once (a refusal fails it) and is reviewed
once more, and one still inconsistent is rejected unevaluated.

## The Evaluator

The Evaluator is the only module that reads data. It compiles all of a round's surviving
graphs as one KunQuant graph, computes it on the whole-market dataset over the training
segment and its warm-up, and scores each candidate through the campaign's scoring: ranked
across the market, neutralised when `scoring.neutralize` is given, and masked to the day's
members. A batch that does not compile is bisected until the failing graphs are found;
they fail at stage `evaluate`. A round's intraday candidates first compute their session
columns on the intraday bars, in chunks of sessions; see
[evaluation and its cost](intraday.md#evaluation-and-its-cost).

On each scored date, only the members with a label are scored. The statistics:

| Statistic | Definition |
|---|---|
| IC | daily rank IC against the label; its mean and Newey-West t |
| partial IC | the rank IC of the residual of the candidate's normal scores on every pool factor's (with an intercept); its mean and t |
| correlation | the mean daily correlation of normal scores with every pool factor; the largest in absolute value is the nearest |
| nearest known | the same correlation with every known factor, on one scored date in five |
| pool share | the mean daily share of the candidate's normal-score variance the pool explains (the part the partial IC leaves out), with the three pool factors most correlated with it; the feedback names accepted ones by trial id |
| deciles | the mean label of each tenth of the daily cross section, lowest first |
| coverage | the mean daily share of the members on which the graph itself has a value, before any fill |

Candidates of one round are admitted in proposal order: an accepted candidate joins the
pool before the next is judged, so two near-duplicates of one round cannot both pass.

## Admission

The [five checks of admission](concepts.md#accepted) run in order and the first that fails
is the recorded reason. The t threshold is a Bonferroni correction anchored at
`admission.t_threshold`: since the direction is read from the data, each evaluated trial
counts as two tests, and the one-sided p-value of the anchor is divided by twice the trial
count. With the default 3.0 the threshold is about 3.21 at the first trial, 3.82 at 10,
4.35 at 100 and 4.73 at 600. `admission.trial_exponent` a, 1 by default, divides it by
twice the trial count to the a instead: at 0.5 a hundred trials count as ten, and the
threshold is 3.82 at 100 and 4.03 at 600, a weaker guard against a lucky draw among many.

## Reading the ledger

```python
import collections, json

rows = [json.loads(line) for line in open("ledger.jsonl")]
print(collections.Counter((r["status"], r["stage"]) for r in rows))
```

An evaluated trial has `metrics` (`ic_mean`, `ic_t`, `partial_ic_mean`, `partial_ic_t`,
`nearest`, `nearest_correlation`, `nearest_known`, `nearest_known_correlation`,
`decile_returns`, `coverage`, `n_days`, `start`, `end`); a refused one has `reasons`, and
`attempts` holds every refused version before the last.
