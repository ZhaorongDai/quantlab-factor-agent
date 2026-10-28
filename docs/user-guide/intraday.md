# Intraday families

An **intraday family** is an input family whose fields are values of each intraday bar
(seconds, minutes or hours wide; one bar width per campaign) in the regular session. A
candidate computes on the bars itself and then reduces each session to one value. A
campaign opens at most one intraday family, beside daily families and, if it likes,
[high-frequency families](vocabulary.md#high-frequency-families), whose statistics are built
beforehand.

A **session** is one trading day's regular hours. A candidate's intraday values for day t
depend only on day t's session, and are known at its close.

## The two stages

A candidate of an intraday family is still one piece of code ending in `alpha`, but it
computes in two stages:

- the **intraday stage** computes on each session's bars, each session on its own, and
  ends in session aggregations;
- a **session aggregation** reduces one session's intraday values to one value for that
  day, per stock. It is the only way from the intraday stage to the daily stage;
- the **daily stage** is everything after the aggregations. It works on one value per stock
  and day, like any daily field, with the daily fields, the daily operators and the daily
  windows.

The proposer is shown two examples, and both pass the Gate on the
[example campaign](#the-example-campaign):

```python
# The share of the session's volume traded in its last 60 bars, averaged over 20 days.
late = SessionSum(Select(session_bars_left < 60, bar_volume, ConstantOp(0)))
alpha = WindowedAvg(late / SessionSum(bar_volume), 20)
```

```python
# The session's volatility of 5-bar returns, averaged over 5 days, against the 20-day
# volatility of daily returns.
intraday_volatility = SessionStd(last_price / BackRef(last_price, 5) - 1)
daily_volatility = WindowedStddev(close / BackRef(close, 1) - 1, 20)
alpha = WindowedAvg(intraday_volatility, 5) / daily_volatility
```

In the first, `Select`, `<` and the two `SessionSum` belong to the intraday stage. The
division and `WindowedAvg(..., 20)` belong to the daily stage, so 20 counts days. In the
second, `BackRef(last_price, 5)` lags 5 bars within the session, and
`BackRef(close, 1)` lags one day.

The Gate splits the candidate's graph at its aggregations. Each aggregation, with what it
reads, becomes an intraday graph, computed once per session. Its value is a **session
column**, named after what the graph computes, so two aggregations that compute the same
values are one column, whether they come from one candidate or two, and the operand order of a
sum or a product does not matter. The daily graph reads each session column as it reads a
field. A graph without an aggregation or an intraday field has no intraday stage: it is
evaluated on the daily bars alone, as in a campaign without an intraday family.

Inside the intraday stage:

- **A window stays within its session.** A window or lag of the intraday stage counts bars
  and is missing on a session's first bars until it fits: `BackRef(x, 5)` is missing on the
  first 5 bars, `WindowedAvg(x, 10)` on the first 9, and `WindowedAvg(WindowedAvg(x, 5), 10)`
  on the first 13. It never reads the session before. The Gate wraps each windowed operator
  of the intraday stage for this; the code is written without it.
- **A cross-sectional operator ranks across the market at each bar.** `Rank(bar_volume)`
  ranks a bar's volume against the same minute's volume of every other stock.
- **A price is missing on a bar without a trade**, and a volume or a trade count is 0 there.
  Nothing assumes a number of bars per session: a half day has fewer, and a stock that did
  not trade at the open still has its bars, without prices.

## The session aggregations

Each aggregation uses the bars of one session where its input has a value. A value is
present when it is a finite number.

| Aggregation | Value | Empty session | Unit |
|---|---|---|---|
| `SessionMean(x)` | mean of x | missing | x's |
| `SessionSum(x)` | sum of x | 0 | x's |
| `SessionCount(x)` | number of bars where x has a value | 0 | pure number |
| `SessionStd(x)` | standard deviation of x, unbiased as `WindowedStddev`; missing below 2 values | missing | x's |
| `SessionSkew(x)` | skewness of x, unbiased as `WindowedSkew`; missing below 3 values or without variance | missing | pure number |
| `SessionMin(x)`, `SessionMax(x)` | least and greatest value of x | missing | x's |
| `SessionFirst(x)`, `SessionLast(x)` | first and last value of x present | missing | x's |
| `SessionWeightedMean(x, w)` | mean of x weighted by w, over the bars where both have a value; missing when the weights sum to 0 | missing | x's |
| `SessionCorr(x, y)` | Pearson correlation of x and y over the bars where both have a value; missing when either is constant there | missing | pure number |

A volume is present on every bar, 0 without a trade, so `SessionCount(bar_volume)` counts
the session's bars and `SessionCount(bar_close)` its bars with a trade. `SessionCorr(x, y)`
and `SessionCorr(y, x)` are one factor to the originality check; `SessionWeightedMean` is
not symmetric.

Moments are computed from values less the session's first value, so a price's deviations are
not lost to its level in float32. Aggregations of one input in a round share their running
sums: `SessionMean(x)` and `SessionStd(x)` add one sum to the other's work, not three.

## The rules the Gate enforces

The stage rules are checked after the code builds its graph and before the parameter
checks. A candidate that breaks one goes back to the proposer with the reason and how to
write it instead, as any Gate refusal does.

| Rule | Refused | Written instead |
|---|---|---|
| An intraday field reaches the daily stage only through a session aggregation. | `WindowedAvg(bar_volume, 20)` | `WindowedAvg(SessionSum(bar_volume), 20)` |
| The intraday stage reads no daily field. | `SessionMean(bar_close / close)` | `SessionMean(bar_close) / close` |
| An aggregation reduces intraday values: no aggregation of a daily value. | `SessionMean(close)` | `close`, used as it is in the daily stage |
| No aggregation is nested in another. | `SessionMean(SessionMax(bar_high))` | a daily window over the sessions: `WindowedAvg(SessionMax(bar_high), 20)` |
| Each operator is used only in the stages its operator-file entry names. | `SessionMean(ExpMovingAvg(bar_close, 10))` | the operator in the daily stage, on a session's value |
| A window in the intraday stage is one of the intraday windows. | `SessionMean(WindowedAvg(bar_close, 20))` with windows `[5, 10, 15, 30, 60, 120]` | `SessionMean(WindowedAvg(bar_close, 15))` |
| A lag in the intraday stage is one of the intraday lags, chained lags included. | `SessionMean(BackRef(BackRef(bar_close, 30), 60))`, a lag of 90 bars | one lag of the set |
| A window or a lag in the daily stage is one of the operator file's values, in days. | `WindowedAvg(SessionSum(bar_volume), 15)` | `WindowedAvg(SessionSum(bar_volume), 10)` |

**Daily-only operators.** An operator's `stages` in the operator file say where it may be
used: `[intraday, daily]`, `[daily]` (the default when it names none), or `[aggregation]`
for a session aggregation. An operator whose memory has no end may not name `intraday`,
because its value on a session's bars would carry the sessions before. In the example file
the exponential average (`ExpMovingAvg`), the windowed linear regressions (KunQuant keeps
their sums as a state running from bar to bar) and `CMRA` (months of returns) are daily
only. Every other windowed operator, loop and cross-sectional operator may be used
in both stages.

The other Gate checks hold for the whole graph as before: the node count, the hand-set
constants, the number of distinct input fields, the output's unit and originality.

## The field file

An intraday family's field file has the keys of any field file (see
[the vocabulary](vocabulary.md#field-files)), plus `intraday`:

```yaml
family: minute_bars
description: one-minute trade bars of each regular session
intraday:
  windows: [5, 10, 15, 30, 60, 120]      # window lengths, in bars, the intraday stage may use
  lags: [1, 5, 10, 15, 30, 60]           # lags, in bars, BackRef may take there

fields:
  - name: bar_close
    column: close
    unit: dollars/shares
    description: last trade price of the bar; missing on a bar without a trade
  - name: session_bars_left
    column: session_bars_left
    unit: 1
    description: bars from this bar to the session's last, 0 for the last bar
```

- `intraday` marks the family. `windows` is required. `lags` is optional, and without it
  the intraday lags are the operator file's lag set. A campaign may open one intraday
  family only, and an intraday family cannot be `bar: true`.
- A field reads a column of the intraday dataset, or one of the three **session inputs**
  the campaign adds to it from the bar timestamps and prices: `session_position` (the bar's
  position in its session, 0 for the first bar), `session_bars_left` (the bars to the
  session's last, 0 for the last bar) and `last_price` (the last trade price at or before
  the bar within the session, read from the bars' `close`; missing before the session's
  first trade).
- A column name may belong to one family only, across both datasets. `Input("close")` is
  intraday exactly when `close` is a column the intraday family reads, so a daily family
  reads `adjClose` and the intraday family reads `close`.

The repository ships `configs/fields/massive_minute_bars.yaml`, the one-minute trade bars
of Massive's trades on permatickers:

| Field | Column | Unit | What it is |
|---|---|---|---|
| `bar_open`, `bar_high`, `bar_low`, `bar_close` | `open`, `high`, `low`, `close` | dollars/shares | the bar's first, highest, lowest and last trade price; missing without a trade |
| `bar_volume` | `volume` | shares | shares traded in the bar |
| `bar_dollar_volume` | `dollar_volume` | dollars | dollars traded in the bar; `bar_dollar_volume / bar_volume` is its VWAP |
| `bar_trades` | `n_trades` | 1 | the bar's trade count |
| `bar_buy_volume`, `bar_sell_volume` | `buy_volume`, `sell_volume` | shares | volume signed by the tick rule |
| `bar_offexchange_volume` | `offexchange_volume` | shares | volume reported off exchange |
| `bar_oddlot_volume` | `oddlot_volume` | shares | volume of trades under 100 shares |
| `session_position`, `session_bars_left` | the session inputs | 1 | the bar's place in its session |
| `last_price` | the session input | dollars/shares | the last trade price within the session |

Every volume, dollar volume and trade count is 0 on a bar without a trade. The store's
auction columns are left out: each is one print on one bar, and the closing cross reaches
the tape after the close, on the session's last bar. A bar is stamped at its end.

## Adding an intraday dataset

1. **Name the dataset in the config** as `data.intraday`, a quantlab component like the
   prices (see [the campaign config](campaign.md#data)). Give the whole-market bars, read on
   the same symbols as the daily prices. A campaign that opens an intraday family without
   `data.intraday` is refused at start.
2. **Write its field file.** Add the `intraday` key with the window and lag sets, one field
   per column, and add the file to `fields`. At start, every intraday field's column is
   checked against the intraday dataset's columns and the session inputs, and a missing
   column is reported with the other problems of the files.
3. **Choose the segments inside the bars.** A day before the intraday data begins has no
   session values. It is not refused: a candidate is missing there, and the days count
   against its coverage. Start the training segment no earlier than the first session.

A session's value is matched to the daily panel by date (a session's label is its day) and
by symbol label as text, so the integer permatickers of the minute bars find the same
permatickers of the daily bars. A symbol the intraday dataset does not hold, and a day
without a session, are missing. The intraday dataset must read a store; a panel held in
memory cannot be computed in chunks.

The run keeps the intraday dataset's config with its snapshot, and the field and operator
files as copies. A resume with another intraday dataset is refused, as a resume with other
field files is. A daily campaign's snapshot is unchanged.

## The example campaign

`configs/us3000_intraday_ret5.yaml` is `us3000_ret5.yaml` with the one-minute trade bars
open beside daily price and volume:

```yaml
segments:
  training:   {start: 2016-10-10, end: 2021-12-31}     # the minute bars start on 2016-10-10
  validation: {start: 2022-01-03, end: 2023-12-29}
  holdout:    {start: 2024-01-02, end: 2026-10-15}
data:
  intraday:
    name: quantlab.dataset.roster.RosterDataset
    dataset: {...}        # MappedTradeBarDataset: massive_permaticker_bars_1m
    roster: {...}         # SEP's domestic common stock, the store the daily prices read
fields: [fields/sharadar_price_volume.yaml, fields/massive_minute_bars.yaml]
```

It keeps that config's prices, label, membership, scoring, reference set, roles and
acceptance. Its known factors are the high-frequency example's, every stored factor of the
minute bars among them, and its seed directions name mechanisms only the minute bars show.
The paths are stores under one data root: point them at your own.

## Evaluation and its cost

Each round, the Evaluator splits every candidate that passed the review into its stages.
All of the round's intraday graphs run as one KunQuant function on the intraday dataset,
over the training segment and the daily warm-up before it. An operator that two candidates
share is computed once when both compute the same values. The session columns are then put on the
daily panel, and the daily graphs run as in a daily round. Coverage is measured on the daily
graph, so a missing session value is a missing value.

The minute bars do not fit in memory at once, so the sessions are computed in chunks. When
the whole span fits in half the memory available it is computed whole. Otherwise it is
computed in the coarsest calendar period (a year, a quarter, a month, a week or a day) whose
largest chunk fits. The values do not depend on the chunks. A chunk holds about 8 bytes per
bar, symbol and column read or computed: a year of one-minute bars is about 98,000 bars, so
on a symbol axis of 17,000 stocks one column of one year is about 13 GB. A round of intraday
candidates reads the minute bars of the whole training segment, so it takes much longer than
a daily round. There is no time limit.

## Export and recompute

An accepted intraday factor exports as a daily one does (see
[acceptance](acceptance.md#export)). Its module's `GraphFactor` subclass holds the factor in
its stages:

| Attribute | Holds |
|---|---|
| `GRAPHS` | the daily graph, reading each session column as an input |
| `SESSIONS` | each session column's intraday graph, by its name |
| `INTRADAY` | the intraday dataset's `get_config()`, with the absolute paths of its stores |
| `CLASSES` | the class of every operator the graphs record |

The factor recomputes its session columns from the intraday bars wherever the module is
imported, to the values the Evaluator scored. Recomputing needs the intraday dataset's
stores at the paths `INTRADAY` records, and the daily dataset's, as for a daily factor. The
export's README says the factor is intraday and how many session columns it reads.

```python
from quantlab.factor.config import FactorConfig
from factor_agent.accepted.late_share import LateShare

factor = LateShare(FactorConfig(
    warmup_bars=260, dataset=prices, mode="batch",
    data_columns=LateShare.data_columns(LateShare.GRAPHS, sessions=LateShare.SESSIONS),
))
values = factor.compute("2024-01-02", "2024-03-29")
```

A graph that reads only session columns reads the first column of the daily dataset for
the panel's dates and symbols. In the [web frontend](web.md), the trial panel and the
factor card tag an intraday factor `intraday`.
