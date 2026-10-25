# quantlab-factor-agent

LLM-driven, mechanism-first formulaic factor research on [quantlab](https://github.com/ZhaorongDai/quantlab).
An LLM proposes economic-mechanism hypotheses and writes each factor as KunQuant code; candidates pass
deterministic gates (sandboxed graph build, originality, complexity) and an LLM reviewer's consistency check
(whether the code computes what the hypothesis says the factor is built from), are evaluated on
quantlab's whole-market panels for incremental value over the existing factor set, and only the training
segment is ever shown to the model.

Status: the multi-round research loop (#5) and acceptance (#7, see below). `Campaign(CampaignConfig.from_file(path)).run()` runs
rounds until a stop condition holds. Each round, every mechanism type gets `min_per_type` slots and
the rest go by Thompson sampling on each type's admission rate; the planner plans one direction per
slot (a new mechanism, an orthogonal mutation of an accepted factor, or a refinement of one); the
proposer writes one candidate per direction; the `Gate` builds each in a sandboxed subprocess and refuses
rewrites of the reference set, the known factors and the accepted factors (compared by their KunQuant
graphs), parameter-only variants of them and large subtrees shared with them, sending the refused ones
back to be fixed; the reviewer, in its own call (possibly another backend) and with the data section's
field descriptions, judges each candidate's hypothesis, description and code for consistency, the inputs
and transformations the hypothesis says the factor is built from included; it is told each candidate's
output unit as the Gate infers it, and a factor whose value carries a unit (a price, a dollar amount, a
share count) needs a hypothesis about that level. An inconsistent candidate is sent back once to be
rewritten, with the reviewer's reason and its output unit; the rewrite passes the Gate once (a refusal
fails it) and is reviewed once more, and one still inconsistent is rejected unevaluated. The `Evaluator`
compiles the rest as one KunQuant graph and scores them on the training segment, as the config's
`scoring` says (ranked over the whole market; neutralised against `scoring.neutralize`'s exposures, size
and industry columns, or not when it is null; masked to the day-t members of `data.membership`, or on
every symbol without one), against the label (rank IC, partial IC residualised on the reference and
accepted factors, correlation with each, deciles), and records the nearest known factor and its
correlation on a fixed sample of the dates. A candidate states no direction: its direction is the sign of
its training IC, and the statistics are read in it. Admission requires a direction, correlation at most
0.7 (with every factor of the reference set and every accepted factor, and sampled with every known
factor), a partial-IC Newey-West t above a threshold that rises with the evaluated trials (each counted
in both directions), monotone deciles, and coverage (the mean daily share of the members with a value) of
at least `admission.min_coverage` (default 0.5). Coverage is measured before any fill: on the candidate's
graph with every `SetInfOrNanToValue` replaced by its input, so filling a sparse field with 0 leaves the
coverage of the field; the scored values still come from the graph as written. The proposer is told to
leave a missing value missing (the scoring skips it) and to fill only when the mechanism says what a
missing value means, stated in the description; the reviewer finds a fill without that reason
inconsistent. An exported factor is turned to its direction: a higher value, a higher return.

The planner and the proposer see the data, the seed `directions`, the reference factor families (then
the known ones, marked as known), the memory, the last round's training statistics and rejection
reasons, and the accepted factors with their marginal weight and residual IC in the accepted pool; the
planner also sees the trials of the last `context.recent_rounds` rounds (default 5) one line each and the
older ones counted by mechanism type and outcome; never a return, a date or a held-out metric. Every `memory_every` rounds the
distiller folds the trials since the last memory into `memory.md` (each version is kept in `memory/`). The run
stops at `max_rounds`, `max_trials`, `patience` rounds without an acceptance, or `budget_usd` (checked before every call), and
records why in `state.json`; `ledger.jsonl` holds every candidate, and `run()` on the same run
directory resumes from the last written round. A first start copies the field, operator and replaced template files into
`<run_dir>/snapshot/` with their hashes (`snapshot.json`); the run, its Acceptance and its replays read
those copies. A resume whose files differ from the snapshot fails (a new search space needs a new run
directory); a resume on other commits of factor_agent or quantlab only warns, and `state.json` lists the
commits of every start and resume. See [docs/design.md](docs/design.md).

## Vocabulary

What candidates may write with comes from files the config names, not from the framework:

```yaml
fields: [fields/sharadar_price_volume.yaml]   # one field file per input family, relative to the config
operators: operators/kunquant_example.yaml    # the operator file
operator_modules: []                          # modules besides KunQuant and quantlab.factor operators may come from
```

- A field file lists its family's fields: each with the price dataset's `column` it reads and its `unit`
  (a product of powers of `dollars` and `shares`: a price is `dollars/shares`, volume `shares`), or a
  derived field's `formula` over the family's earlier fields (under candidate code's syntax rules), and a
  description, which the prompts and the reviewer read. `bar: true` makes a missing value in the family's
  columns a missing bar, which quantlab's NaN-preserving stand-ins keep NaN.
- The operator file lists each operator's KunQuant `class`, the `call` candidate code uses (such as
  `quantlab.factor.kunquant_nan.Max`), its signature, argument and parameter kinds, dimension rule
  and description, plus the parameter value sets. The repository's example file also
  lists every other KunQuant and quantlab operator class under `excluded` with a reason, which a guard
  test checks; a user's own file needs no such section.
- An operator file may list `fixes`: each a `match` (a regular expression a refusal's text matches when
  KunQuant turns code down) and a `hint` (how to write it instead); the Gate gives a writing error the
  hints it matches, in file order, and they go with the reasons into the proposer's fix request only. Candidate code and derived formulas may write `-x`
  (`MulConst(x, -1)`) and `x ** c` (`Pow(x, ConstantOp(c))`, when the file registers `Pow`).
- At start every field and operator is checked and every problem reported at once, before any LLM call:
  a column missing from the price dataset, a malformed unit, a name clash, a class that is not a KunQuant
  operator, a class or callable from a module not listed, a formula outside the syntax rules.

## Prompts

The prompts describe the config's data and research, not one market:

```yaml
market: adjusted daily bars of every US common stock   # required: the data, as the prompts say it
mechanisms:                                             # optional: the default is the five types below
  - {name: behavioural_bias, description: investors' systematic errors that prices correct later}
prompts:                                                # optional: templates replaced from files
  review: prompts/review.txt
```

- The prediction horizon is the label's `span_bars()`; the data section is the market, each field of
  the opened families with its unit and description, and the scoring's sentence.
- `mechanisms` are the types the planner allocates slots to and a candidate may have, each with a
  one-line description the planner reads; `round.min_per_type` times their number must fit in
  `round.candidates`. The default is `behavioural_bias`, `risk_compensation`, `liquidity`,
  `microstructure` and `information_diffusion`.
- `prompts` may replace the templates `plan`, `propose`, `retry` (the proposer's fix request), `review`,
  `distill` and `data` (the data section), each a Python format string. At start each replaced template
  is checked for the placeholders its role needs (and for any it may not use), before anything is
  written; a replaced template is snapshotted like the field and operator files, so changing it needs a
  new run directory. The placeholders are listed in `factor_agent/prompts.py` (`PLACEHOLDERS`).

## LLM backends

Each role under `llm` picks its backend and model; the other keys are the backend's settings.

```yaml
llm:
  planner:  {backend: claude_code, model: claude-opus-5-5, effort: high}
  proposer: {backend: anthropic, model: claude-opus-5-5, max_tokens: 16000,
             input_usd_per_mtok: 4.0, output_usd_per_mtok: 20.0}
  reviewer: {backend: openai_compatible, model: deepseek-chat, base_url: https://api.deepseek.com/v1,
             api_key_env: DEEPSEEK_API_KEY, input_usd_per_mtok: 0.27, output_usd_per_mtok: 1.10}
  distiller: {backend: claude_code, model: claude-opus-5-5}
```

- `claude_code` runs `claude -p --output-format json` with every tool, MCP server, skill and settings
  file off, from an empty directory, through the local Claude Code login; it reports its own cost
  (prices in the config are only used for the estimate).
- `anthropic` reads `ANTHROPIC_API_KEY`; `openai_compatible` (OpenAI, DeepSeek, DashScope, Gemini's
  compatible endpoint, vLLM, Ollama) reads the variable `api_key_env` names, or none for a local server;
  for a reasoning model give `max_completion_tokens` (in place of `max_tokens`) and `reasoning_effort`.
  Their cost is the reply's token usage at the configured prices (check them against the provider's
  price list). A key in the config is refused.
- Every call is written in full to `<run_dir>/llm_log/` (prompt, model, parameters, reply, usage, cost,
  the raw reply). `replay` (every role `{backend: replay, model: ..., log: <run_dir>/llm_log}`) serves such a log back, matched on role and prompt, so the campaign
  reproduces byte for byte without calling a provider; a request the log has no reply for fails.
  A replay searches with the snapshot of the run whose log it replays, not the config's files.
- Before the first round `run()` prints the estimated cost per role against `budget_usd`
  (`Campaign(config).estimate()`); the run stops once the calls have cost the budget.

## Acceptance

After the campaign, `Acceptance(config)` judges the accepted factors by what they add to the acceptance
model, which the config must name:

```yaml
segments:
  training:   {start: 2000-02-02, end: 2014-12-31}
  validation: {start: 2015-01-02, end: 2019-12-31}
  holdout:    {start: 2020-01-02, end: 2026-10-23}
  purge_bars: 5          # at least; a label that reads further purges more (ret_5: 6)
acceptance:
  model:                 # required: the get_config() of any quantlab BaseModel or ModelEnsemble
    name: quantlab.model.predefined.xgb.XGBoostRegressor
    factors: []          # the baseline, as reference entries; empty: the reference set
    hyperparameters: {...}   # us3000_ret5.yaml names SEL0's xgb_deep
  seeds: [0, 1, 2]       # each replaces every model's (every member's) random_seed
  contribution: none     # per-factor contribution in validate(): none, drop_one, add_one or both
  top_k: 100
  factor_root: /data/quantlab/factors/market   # where export() writes the stores; an export needs it
```

- The model is a template. A `ModelEnsemble` is `{name: ...ModelEnsemble, members: [<model>, ...]}`, each
  member with its own `factors` (its baseline; empty: the reference set) and hyperparameters. Acceptance
  fills in the label, the windows, `val_size`, the read strategies and the seed, and appends the accepted
  factors to every member; what a template says of those is replaced, so a trained run's `config.json`
  serves as one. A `SeedEnsemble` is refused (the seeds are `acceptance.seeds`); the removed
  `acceptance.hyperparameters` and `acceptance.features` fail to load.
- `validate(selection=None)` trains the model per seed on its baseline and on its baseline plus a
  selection of the accepted factors, named as `export` names them (description, or
  `description (accepted in round n)`), every accepted factor by default (through the same `scoring` the Evaluator admitted with), up to the
  purged end of training, and predicts the purged validation segment. It reports, per seed and on
  average, rank IC, IC, decile monotonicity, D10−D1, and the fee-free plain TopK return and excess
  (`<run_dir>/acceptance/validation.{json,md}`; a selection of fewer goes to
  `validation_<digest>.{json,md}`, the digest over its members' labels, and each report lists its members).
  An unknown or ambiguous name raises `KeyError`.
- With `contribution`, `validate()` also measures each factor of the selection, with the same seeds:
  drop-one trains the model without it (the whole selection against the selection without it), add-one on
  the baseline plus only it (against the baseline); a feature set equal to another is trained once. The
  reports gain a contribution table, one row per factor and mode, sorted by the rank IC difference: the
  seed-averaged difference of every statistic and, per statistic, how many seeds have the mean's sign (the
  Markdown shows rank IC difference, seeds agreeing and TopK excess difference). No threshold. The
  hold-out never measures contribution.
- Every trained model's result is cached under `<run_dir>/acceptance/<segment>/cache/`, keyed by its
  filled-in config (template, features, label, windows, seed; paths under the run directory relative to it, so a
  moved run directory keeps its cache), the code version (the factor_agent and quantlab commits) and the TopK
  size and scoring; a re-run,
  after an interruption say, reads what is cached and trains the rest. Every selection reads its factors
  from one store of all accepted factors (one per predicted segment), so equal feature sets, a narrowed
  selection's add-one and the whole set's say, share their models. Before training, `validate()` and
  `open_holdout()` print how many models they will train, how many are cached and, once some models of the same template have
  finished, an estimated duration from their times. The print never waits for an answer.
- `open_holdout(selection, force=False)` needs `validate`'s report of exactly that selection, then writes
  `<run_dir>/holdout.lock` (when, with which commits) and makes the same comparison on the hold-out for the
  selection only, trained through validation (`holdout[_<digest>].{json,md}`). A second open, of any
  selection, raises `HoldoutLocked`; `force=True` runs it, appends to the lock, and the report says the
  hold-out was opened again.
- `export({"<description>": "<name>"})`, the keys being the selection, writes `factor_agent/accepted/<name>.py` (a `GraphFactor`, i.e.
  `FactorKunQuant`, subclass fixing the scored graph) and the whole-market store
  `<factor_root>/<name>/` (`NeutralizedFactor` of the ranked graph) with `README.md` and `component.json`;
  the README quotes `validate`'s report of the exported selection, when there is one.

## Setup

```
uv sync          # quantlab is an editable dependency at ../quantlab
uv run pytest
```

Credentials come from environment variables only (`ANTHROPIC_API_KEY`, or the variable named in the
config for an OpenAI-compatible provider); the `claude_code` backend uses the local Claude Code login.

## First campaign

[`configs/us3000_ret5.yaml`](configs/us3000_ret5.yaml): whole-market Sharadar SEP with bad prints masked, IC on
the members of day t of us3000 `membership_estu`, the label `labels/market/ret_5`, and SEL0's 191 factors as
the reference set. A reference entry is a component (`get_config()` dict or store folder) or
`{store: <folder>, factor_names: [...]}`, the store's factor pinned to those outputs. The Gate compares a
candidate with the graph of the factor a `*_neutral` store neutralises; every stored reference or baseline
factor is read from its store's first bar (the intraday stores start in 2016, so they are empty on training).

A reference or known entry given as a component (a `get_config()` dict without a store) is scored as a
candidate is, through the campaign's `scoring` (neutralised when it neutralises, so it should output
ranked values, as candidates do), and then masked to the day's members; an entry given as a store (a
store folder, a store entry, or a component whose store exists, such as a `*_neutral` store) is read as
stored and masked. Both lists are read by this one rule.

`known` (optional, default none) lists the known factors besides the reference set: every other existing
factor a candidate is checked against for novelty. Its entries are written as `reference`'s; an output a
reference factor of the same class already shows is dropped from `known`, so whole classes can be listed
without subtracting the reference outputs, and a malformed entry fails when the config is read. The Gate
refuses a rewrite, a parameter-only variant or a large shared subtree of a known factor, naming it as
`STD20 (Alpha158Stock, a known factor)`; `zoo.json` lists the known factors it covers and those without a
KunQuant graph it cannot; the families section lists the known classes after the reference families, marked
"known, not in the reference set, refused when rewritten". Known factors never enter the partial IC. Like
`reference`, `known` is read from the config on every start, a resume included, and is not snapshotted.

On each start or resume the `Evaluator` reads the known factors as it reads the reference set and keeps
only their normal scores on one scored training date in `KNOWN_SAMPLE_EVERY` (5), counting from the
first, in a panel of its own (float32, one column per output). Each
evaluated candidate's correlation with every known output is averaged over those dates; the nearest one and
its correlation are in the ledger's metrics (`nearest_known`, `nearest_known_correlation`; empty and NaN
without `known`) and in the next round's feedback, and admission's correlation step refuses an absolute
sampled correlation above `max_correlation`, naming the factor and saying the correlation is sampled.
The panel costs 4 bytes per known output per sampled member-day: measured on the synthetic test market
(Alpha158's 169 outputs, 103 sampled dates of 100 members) it is 6.6 MiB, exactly that. Estimated, not
measured, for us3000 (training 2000-02 to 2014-12, about 750 sampled dates of about 3,000 members) it is
about 9 MB per known output: about 1.8 GB for 200 outputs beyond the reference set, 2.7 GB for 300. While it
is built, one known factor's panel at a time is read whole, as a reference factor's is.

```yaml
known:
  - /data/quantlab/factors/market/alpha158_neutral   # a store folder: every output
  - {store: /data/quantlab/factors/market/alpha101_neutral, factor_names: [alpha001, alpha002]}
```

On the server (`~/projects/quantlab-factor-agent`, next to `~/projects/quantlab`), on NUMA Node 1 (Node 0's
memory bandwidth is degraded); KunQuant runs `evaluator.njobs` = 51 threads; set no thread variables:

```
uv sync
numactl --cpunodebind=1 --membind=1 uv run pytest -n 16 -q              # includes the offline replay smoke test
nohup numactl --cpunodebind=1 --preferred=1 uv run python -c \
  "from factor_agent.campaign import Campaign; from factor_agent.config import CampaignConfig; \
   Campaign(CampaignConfig.from_file('configs/us3000_ret5.yaml')).run()" > campaign.log 2>&1 &
```

After the campaign, the same way: `Acceptance(CampaignConfig.from_file('configs/us3000_ret5.yaml')).validate()`.
The campaign uses no GPU. `Acceptance` trains SEL0's xgb on the GPU (quantlab picks CUDA); start it only when it
ends before 06:00 US Eastern on a weekday, when the GPU must be free.
