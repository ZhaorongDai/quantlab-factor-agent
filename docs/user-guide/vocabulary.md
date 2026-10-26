# The vocabulary

The **vocabulary** is everything a campaign's candidates may write with: the fields of
its input families, the operators, and the values the operators' parameters may take. The
framework has none of its own; the config names the files:

```yaml
fields: [fields/sharadar_price_volume.yaml, fields/sharadar_fundamentals.yaml]
operators: operators/kunquant_example.yaml
operator_modules: []        # modules besides KunQuant and quantlab.factor operators may come from
```

At start every field and operator is checked and every problem reported at once, before
any LLM call: a column missing from the price dataset, a malformed unit, a name clash, a
class that is not a KunQuant operator, a class or callable from a module not listed, a
formula outside the syntax rules.

## Field files

One field file per **input family**, a group of data opened or closed as a whole: price
and volume, or fundamentals. Each field names the price dataset's `column` it reads, or a
`formula` over the family's earlier fields, its `unit` and a one-line description, which
the prompts and the reviewer read:

```yaml
family: price_volume
description: adjusted daily bars
bar: true                     # a value missing in any column is a missing bar

fields:
  - name: close
    column: adjClose
    unit: dollars/shares
    description: closing price
  - name: vwap
    formula: (high + low + close) / 3.0
    description: typical price of the day, (high + low + close) / 3
  - name: amount
    formula: vwap * volume
    description: dollars traded in the day, vwap * volume
```

- A unit is a product of powers of `dollars` and `shares`: a price is `dollars/shares`,
  volume `shares`, a ratio `1`. A derived field's unit follows from its formula.
- `bar: true` makes a value missing in any of the family's columns a missing bar; the
  NaN-preserving operator stand-ins keep it NaN.
- A family of periodic data may declare a **period lag**, which lets code write
  `PeriodLag(x, k)`: the field's value `k` reporting periods earlier, counted in reports,
  not trading days.

```yaml
period_lag:
  column: '{column}__q{k}'      # the name a lag is read under
  k: [1, 2, 3, 4, 8]
  period:
    description: one quarterly report
    key: reportperiod           # the column holding the period end of the report shown
    days: 91.3
    tolerance_days: 45
    bars: 63                    # the most bars one period spans, for the warm-up
```

The repository ships `sharadar_price_volume.yaml` (Sharadar SEP's adjusted bars) and
`sharadar_fundamentals.yaml` (SF1 trailing-twelve-month items and DAILY valuations).

## High-frequency families

A **high-frequency family** is an input family whose fields are daily statistics of a
minute dataset: one value per stock and session, such as the realized volatility of the
session's minute returns or its order imbalance. The campaign computes nothing from minute
bars; it reads a store of the statistics built beforehand, beside the daily prices, and
candidate code applies its own windows to the daily values like to any other field.

A new source takes two steps, and no change to factor-agent:

1. **Build the daily statistics store.** Compute the statistics from the minute dataset with
   quantlab's session-statistic factors (`IntradayStats`, `PriceVolumeIntradayStats`),
   resampled to one value per day, and save them as a factor store.
2. **Write a field file for it.** Add a quantlab `FactorStoreDataset` over the store to the
   campaign's price dataset, merged with the daily bars, and write a field file whose
   columns are the statistics' names in the merged panel, with `bar: false`:

```yaml
data:
  prices:
    name: quantlab.dataset.merged.MergedDataset
    datasets:
    - ...                                   # the daily bars, as before
    - name: quantlab.dataset.factor_store.FactorStoreDataset
      zarr_file_path: factors/session_stats/session_stats.zarr
      prefix: 'trades_'                     # rvol is read as trades_rvol
```

```yaml
family: session_statistics
description: daily statistics of each session's minute trade bars
bar: false                    # a missing statistic is a missing value, not a missing bar

fields:
  - name: rvol
    column: trades_rvol
    unit: 1
    description: realized volatility of the session's minute returns
```

Two rules hold for every such store:

- **A store's value for day t is known at day t's close.** The dataset shifts nothing: the
  value stored at day t is read beside day t's prices, so the store must hold at day t only
  what that day's session revealed by its close. Aligning the values is the store
  builder's job.
- **A missing statistic is a missing value.** A session with too few trades leaves its
  statistics missing, and with `bar: false` that blanks only the candidates that read them;
  the same day's prices keep their values and a price-only candidate is unaffected.

The **prefix** keeps two sources apart. Two stores may hold an output of the same name
(trade bars and quote bars both give an `rvol`); read each under its own prefix
(`trades_rvol`, `quotes_rvol`) and give each field file its own field names. Two merged
inputs that both hold a value of the same column for a stock on a day are refused when the
panel is read, so a field never silently reads the wrong source. A field file naming a
column the merged panel lacks (a stored name without its prefix, a typo) is refused at
start, before any LLM call, with every other problem of the files.

## The operator file

The operator file lists each operator candidate code may call: its KunQuant `class`, the
`call` code writes when it is not the class (such as quantlab's NaN-preserving
`quantlab.factor.kunquant_nan.Max`), its signature, the kind of each input, the parameter
set each attribute must be in, its dimension rule and a description.

```yaml
parameters:
  window: [5, 10, 20, 40, 60, 120, 250]
  constant: [0, 1.0e-12, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 60, 120, 250]

operators:
  - name: WindowedAvg
    class: KunQuant.ops.WindowedAvg
    signature: "WindowedAvg(x, window)"
    inputs: [series]
    params: {window: window}
    dimension: same
    description: "mean over the last window bars"
```

The dimension rule tells the Gate the unit of the operator's output (`same`, `match`,
`product`, `quotient`, `pure`, `compare`, ...), so it can infer a candidate's output unit;
the reviewer reads that unit, and a factor whose value carries one (a price, a dollar
amount, a share count) needs a hypothesis about that level.

The repository's example file also lists every other KunQuant and quantlab operator
class under `excluded`, with the reason; a guard test keeps it complete. Your own file
needs no such section.

## Fixes

An operator file may list **fixes**: a `match` (a regular expression a refusal's text
matches when KunQuant turns code down) and a `hint` (how to write it instead). The Gate
gives a writing error every hint it matches, in file order, and the hints go into the
proposer's fix request only.

```yaml
fixes:
  - match: 'Bad inputs, given'
    hint: "write a number as ConstantOp(1), as in Sub(x, ConstantOp(1)), or use x - 1"
```

## What code may write

Candidate code is plain Python that must assign the factor to `alpha`: assignments,
arithmetic and comparisons, calls of the vocabulary's operators, `def`/`return`,
`for`/`while`/`if` and `with` blocks over plain names and constants. It may write
`+ - * /`, unary minus (`MulConst(x, -1)`) and `x ** c` (`Pow(x, ConstantOp(c))`, when the
file registers `Pow`). Imports, class definitions, attribute access, names with double
underscores and any other syntax are refused before the code runs; the code then builds its
graph in a sandboxed subprocess with a time and memory limit.

A missing value stays missing: the scoring skips it. A candidate fills one
(`SetInfOrNanToValue`) only when the mechanism says what a missing value means, stated in
its description; coverage is measured on the graph with every fill removed, so filling a
sparse field does not raise its coverage.
