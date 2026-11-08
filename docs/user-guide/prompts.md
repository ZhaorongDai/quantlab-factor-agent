# Prompts

The prompts describe the config's data and research, not one market. They are built from
`market`, the label's horizon (its `span_bars()`), the scoring, the vocabulary and the
mechanism types, through six templates you can replace.

```yaml
market: adjusted daily bars of every US common stock    # required: the data, as the prompts say it
mechanisms:                                              # optional: the default is the five types below
  - {name: behavioural_bias, description: investors' systematic errors that prices correct later}
  - {name: liquidity, description: the cost and the pressure of trading}
prompts:                                                 # optional: templates replaced from files
  review: prompts/review.txt
```

## Mechanism types

`mechanisms` are the types the planner allocates slots to and a candidate may belong to,
each with a one-line description the planner reads, in plan order. `round.min_per_type`
times their number must fit in `round.candidates`. Without `mechanisms` the types are:

| Type | Description |
|---|---|
| `behavioural_bias` | investors' systematic errors (over- and underreaction, anchoring, limited attention) that prices correct later |
| `risk_compensation` | a return paid for bearing a risk investors demand to be compensated for |
| `liquidity` | the cost and the pressure of trading: illiquidity premia, liquidity demand and its reversal |
| `microstructure` | the mechanics of trading within and between bars: order flow, the bid-ask bounce, the open and the close |
| `information_diffusion` | news that reaches prices gradually, across investors, related firms or time |

## Templates

| Template | Role | Must use | May also use |
|---|---|---|---|
| `data` | the data section, the same every round | `market`, `inputs`, `scoring`, `horizon` | |
| `plan` | planner | `data`, `context`, `quota` | `round`, `mechanisms`, `actions` (the actions a direction may take and how it names its parent), `picked` (the picked slots, empty without `round.picked_parents`; required with it) |
| `propose` | proposer | `data`, `context`, `code_rules`, `vocabulary`, `fields`, `n`, `directions` | `round`, `mechanisms` |
| `retry` | proposer's fix request | `code_rules`, `vocabulary`, `fields`, `n`, `failures` | `round`, `data`, `mechanisms` |
| `review` | reviewer | `candidates`, `data` | `round`, `mechanisms` |
| `distill` | distiller | `memory`, `history` | `round`, `data`, `mechanisms`, `accepted` |

Every template may use `market` and `horizon`. A template is a Python format string; each
`{placeholder}` is filled when the prompt is made. When the config is read, each replaced
template is checked for the placeholders its role needs and for any it may not use, and
every problem is reported at once (`PromptError`). A replaced template is copied into the
run's snapshot like the field and operator files, so changing it needs a new run
directory.

## What the model is never shown

The prompts carry training-segment statistics only: no return, no date, no validation or
hold-out metric. Papers in seed directions are cited without their year, so no prompt
names a date after the training segment.
