# How the agent mines factors

The agent searches for factors the way a research team would: it proposes an economic
mechanism, writes the factor the mechanism implies, and keeps it only if the factor is new
and adds to what is already known. Language models do the proposing and one consistency
check. Everything that decides whether a factor is kept is deterministic code working on
statistics the models cannot influence. This page explains the mechanism without the
config keys; [concepts](concepts.md) and the pages it links give the details.

```{image} ../_static/mining-mechanism.svg
:alt: The four parts of the mechanism: the search by language models, the judges, learning from every trial, and the data the models may see
:width: 100%
```

## 1. The search: mechanisms, not formulas

A campaign runs in rounds of a fixed number of slots, twenty by default. Each slot belongs
to a **mechanism type**, a broad reason why some stocks earn more than others over the next
few days: behavioural biases, risk compensation, liquidity, market microstructure and the
slow diffusion of information. Every type gets a few slots each round, so no type is
abandoned early. The remaining slots go to the types that have been admitted most often so
far, by Thompson sampling: a type that keeps producing factors gets more attempts, and a
type that has done badly still gets some.

The **planner** writes one research direction per slot. A direction is either a new
mechanism, or builds on an earlier trial, named by its id (`T3-07` is round 3's seventh
slot):

- a **refinement** measures the parent's mechanism in another way, to remove what got it
  refused or to strengthen an accepted factor;
- a **mutation** follows a different channel that the parent's result points to.

The **proposer** turns each direction into a candidate: a hypothesis (why the mechanism
should predict returns), a description of the factor, and KunQuant code that computes it
from the fields the vocabulary opens. A round's directions are split into proposal groups
that are written in parallel, each by calls of its own.

The models are never asked to search over formulas. They are asked for mechanisms, the
Gate refuses what is merely a variant of an existing factor, and the reviewer checks that
the code measures what the mechanism says.

## 2. The judges: only code accepts

Every candidate passes four judges in turn. Each can refuse it, and the first refusal is
the recorded reason.

**The Gate** checks the code before any data is read. The code may use only the
vocabulary's fields and operators; it must build its factor graph in a sandboxed process
within a time and memory limit; its graph must stay within limits on size, hand-set
constants and the number of input fields. The Gate then compares the graph with every
factor in the **zoo** (the reference factors, other known factors and the factors accepted
so far) after putting both in a canonical form. A rewrite of an existing factor, the same
graph with other parameters, or a large shared subtree is refused, and the refusal names
the factor. A refused candidate goes back to the proposer with the reasons and may be fixed
twice.

**The reviewer**, a language model, reads the hypothesis, the description and the code
together and judges whether they describe the same factor. One that does not is sent back
once to be rewritten; a candidate still inconsistent is rejected without being evaluated.

**The Evaluator** is the only part that reads market data. It computes every surviving
candidate on the whole market over the training segment, scores it on each day's members of
the universe, and measures its rank IC, its **partial IC** (the IC of what is left after
regressing the candidate on every factor in the pool), its correlation with each existing
factor, its decile returns and its coverage.

**Admission** accepts a candidate only when, in this order:

1. it has a value for at least half the members on an average day;
2. its training IC is nonzero, which gives it a direction (a negative IC means the factor
   is used as its negative);
3. it is correlated at most 0.7 with every reference, accepted and known factor;
4. its partial IC is significant: its Newey-West t clears a threshold that rises with the
   number of trials evaluated so far;
5. its decile returns rise with the decile.

## 3. Learning: every trial shapes the next round

Every slot ends as a **trial** in the ledger, accepted, rejected or failed, with its
statistics or the reasons it was refused. The next round's planner and proposer read:

- the last round's statistics and rejection reasons, and a one-line summary of each trial
  of the last few rounds;
- the accepted factors, each with its weight and residual IC in the accepted set;
- for a candidate refused for low coverage, the step of its graph where the values were
  lost;
- a **memory**, a short text the distiller model rewrites every few rounds from the trials
  since the last one, about what worked and what did not;
- optionally, **themes**: the evaluated trials grouped by what their factors measure, each
  group with how often it was tried and accepted, so the planner can see which ideas are
  exhausted. A new direction whose candidate repeats an earlier group can be sent back
  once to be rewritten before it is evaluated.

An accepted factor also changes what later candidates are judged against. It joins the zoo,
so the Gate refuses its rewrites, and it joins the pool, so the next candidate's partial IC
is what it adds beyond it. Within a round, candidates are admitted in order and each
accepted one joins the pool before the next is judged, so two near-copies of one idea
cannot both pass. The bar rises as the set grows.

With themes turned on, the themes and the choice of directions form a map of what the search has tried:

```{image} ../_static/direction-map.svg
:alt: The direction map: slots by mechanism type, directions written from the theme groups, repeats sent back once, and every evaluated trial joining a theme
:width: 100%
```

## 4. Information: the models never see the future

The data is split into segments in time: training, then validation, then a hold-out. The
models see only statistics of the training segment. No prompt carries a return, a date or
any validation or hold-out number, and papers in seed directions are cited without their
year. The last bars of each segment, whose label would read into the next, are dropped.

The validation segment is used only after the campaign, to judge the accepted factors
together. The hold-out is opened once, behind a lock, after the factors have been chosen
on validation; a second opening is refused unless forced, and the report then says so.

## Guarding against luck

A search that tries hundreds of factors will find some that look good by chance. The agent
guards against that in four layers:

- **A threshold that counts the trials.** The partial-IC t needed for admission is a
  Bonferroni correction over every evaluated trial, each counted twice because the
  direction is read from the data: about 3.2 for the first trial, 3.8 at 10, 4.4 at 100
  and 4.7 at 600.
- **A confirmation segment.** The trial count cannot account for a search that adapts to
  its own feedback. In a test on daily bars, a run that read its trials accepted more
  factors than a run that read none, without a larger sum of validation partial t.
  Optionally, the later part of the training data is set aside as a confirmation segment.
  A candidate that passes everything else must also hold there, and the models are told
  only that it failed, never by how much.
- **Validation of the set.** After the campaign, a quantlab return model is trained with
  and without the accepted factors, and predicts the validation segment. The report can
  also measure each factor alone, by dropping it from the set or adding only it to the
  baseline. There is no threshold here: you choose which factors to keep.
- **A hold-out opened once.** The chosen selection is measured on the hold-out a single
  time.

A control run with feedback turned off, in which every direction is new, shows how much the
learning loop adds over undirected proposals.

## After the campaign

The chosen factors are exported as quantlab factors and whole-market stores, each turned to
its direction so that a higher value always means a higher expected return. See
[acceptance](acceptance.md).

A campaign stops after a set number of rounds or trials, after several rounds in a row
without an acceptance, or when the model provider refuses a call for an exhausted balance
or quota. It can be resumed from its run directory at any point.
