# Concepts

The vocabulary the rest of the user guide assumes: the research loop and its roles, the
factor sets a candidate is compared with, the segments, and what makes a factor accepted.
The project's [glossary](https://github.com/ZhaorongDai/quantlab-factor-agent/blob/main/GLOSSARY.md)
defines every term.

## Mechanism first

A candidate is a **hypothesis** about an economic mechanism (why some stocks earn more
over the next few days), a **description** of the factor the hypothesis implies, and
**code** that computes it, written with quantlab's KunQuant operators. Every candidate
belongs to a **mechanism type**; the default types are `behavioural_bias`,
`risk_compensation`, `liquidity`, `microstructure` and `information_diffusion`.

The agent does not search over formulas. It is asked for mechanisms, and the checks that
follow refuse what is merely a variant of an existing factor.

## The roles

| Role | Calls per round | Reads | Writes |
|---|---|---|---|
| planner | one | seed directions, factor families, memory, recent trials, last round's feedback | one direction per slot |
| proposer | one per proposal group, plus fixes | the directions, the data section, the vocabulary | candidates |
| reviewer | one per proposal group, plus one for rewrites | each candidate and the data section | a verdict per candidate |
| distiller | one every `round.memory_every` rounds | the trials since the last memory | `memory.md` |

Each role names its own backend and model, so the reviewer can be another provider's
model than the proposer's. See [LLM backends](llm-backends.md).

A round's slots are allocated by **mechanism type**: each type gets `round.min_per_type`
slots and the rest go by Thompson sampling on each type's admission rate so far. The
round's directions are split into `round.groups` **proposal groups**, contiguous runs in
mechanism-type order; each group is proposed, fixed and reviewed by calls of its own, and
the groups' calls of a stage are sent at once. The Gate still checks the whole round in
slot order, so a duplicate written in another group is refused.

## Factor sets

| Set | What it is | A candidate must be |
|---|---|---|
| reference set | the factors the agent is trying to add to (`reference`) | not a rewrite of one, correlated at most `max_correlation`, incremental (partial IC) |
| known factors | every other existing factor (`known`) | not a rewrite of one, correlated at most `max_correlation` on a sample of dates |
| accepted factors | what this campaign accepted so far | like the reference set |

The **zoo** holds the KunQuant graphs of all three; the Gate compares a candidate's graph
with them. The **reference pool** holds the scored values of the reference set and the
accepted factors; the Evaluator measures a candidate's correlation with it and regresses
the candidate on it for the partial IC. Known factors never enter the pool.

## Segments

| Segment | Used by | Shown to the LLM |
|---|---|---|
| training | the Evaluator, admission, the prompts' statistics | yes, as statistics only |
| validation | `Acceptance.validate()` | no |
| hold-out | `Acceptance.open_holdout()`, once | no |

Each segment drops the last bars whose label reads past its end (`purge_bars`, at least
the label's lookahead). The prompts carry no returns, dates or held-out metrics.

## Direction

A candidate states no direction. Its direction is the sign of its training IC, and every
statistic is read in that direction: a factor with a negative IC is judged, and exported,
as its negative. An exported factor always reads "a higher value, a higher return".

## Accepted

A candidate is accepted when, in order:

1. its coverage (the mean daily share of the members with a value, measured before any
   fill) is at least `admission.min_coverage`;
2. its training IC is a nonzero number, so it has a direction;
3. its correlation with every reference, accepted and (sampled) known factor is at most
   `admission.max_correlation` in absolute value;
4. its partial IC's Newey-West t clears a threshold that rises with the number of
   evaluated trials (a Bonferroni correction anchored at `admission.t_threshold`);
5. its decile returns rise with the decile: a Spearman correlation of at least
   `admission.min_monotonicity`.

The first check that fails is the reason recorded. See [evaluation](evaluation.md).
