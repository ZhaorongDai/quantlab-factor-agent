# quantlab-factor-agent

LLM-driven, mechanism-first formulaic factor research on [quantlab](https://github.com/ZhaorongDai/quantlab).
An LLM proposes economic-mechanism hypotheses and writes each factor as KunQuant code; candidates pass
deterministic gates (sandboxed graph build, originality, complexity, mechanism binding), are evaluated on
quantlab's whole-market panels for incremental value over the existing factor set, and only the training
segment is ever shown to the model.

Status: the multi-round research loop (#5) and acceptance (#7, see below). `Campaign(CampaignConfig.from_file(path)).run()` runs
rounds until a stop condition holds. Each round, every mechanism type gets `min_per_type` slots and
the rest go by Thompson sampling on each type's admission rate; the planner plans one direction per
slot (a new mechanism, an orthogonal mutation of an accepted factor, or a refinement of one); the
proposer writes one candidate per direction; the `Gate` builds each in a sandboxed subprocess and
refuses rewrites of the zoo (the KunQuant graphs of the `reference` factors plus the accepted ones),
parameter-only variants, large shared subtrees and hypotheses that name an input or transformation
the graph never uses, sending the refused ones back to be fixed; the reviewer, in its own call
(possibly another backend), judges each candidate's hypothesis, description and code for
consistency and an inconsistent one is rejected unevaluated. The `Evaluator` compiles the rest as
one KunQuant graph and scores them on the training segment, as the config's `scoring` says (ranked
over the whole market; neutralised against `scoring.neutralize`'s exposures, size and industry columns, or
not when it is null; masked to the day-t members of `data.membership`, or on every symbol without one),
against the label (rank IC, partial IC residualised on the reference and accepted factors, correlation with each,
deciles). Admission requires the stated sign, correlation at most 0.7, a partial-IC Newey-West t above
a threshold that rises with the evaluated trials, and monotone deciles.

The planner and the proposer see the data, the seed `directions`, the reference factor families, the
memory, every earlier trial, the last round's training statistics and rejection reasons, and the
accepted factors with their marginal weight and residual IC in the accepted pool; never a return, a
date or a held-out metric. Every `memory_every` rounds the distiller writes `memory.md`. The run
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
  derived field's `formula` over the family's earlier fields (under candidate code's syntax rules), a
  description, and the `synonyms` (phrases) that bind a hypothesis to it; `rules` name phrases that need
  several fields (overnight needs open and close). `bar: true` makes a missing value in the family's
  columns a missing bar, which quantlab's NaN-preserving stand-ins keep NaN.
- The operator file lists each operator's KunQuant `class`, the `call` candidate code uses (such as
  `quantlab.factor.kunquant_nan.Max`), its signature, argument and parameter kinds, dimension rule,
  description and binding phrases, plus the parameter value sets. The repository's example file also
  lists every other KunQuant and quantlab operator class under `excluded` with a reason, which a guard
  test checks; a user's own file needs no such section.
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
  new run directory. The placeholders are listed in `factor_agent/_prompts.py` (`PLACEHOLDERS`).

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

After the campaign, `Acceptance(config)` judges the accepted factors by what they add to SEL0 xgb:

```yaml
segments:
  training:   {start: 2000-02-02, end: 2014-12-31}
  validation: {start: 2015-01-02, end: 2019-12-31}
  holdout:    {start: 2020-01-02, end: 2026-10-23}
  purge_bars: 5          # at least; a label that reads further purges more (ret_5: 6)
acceptance:
  features: [...]        # SEL0's features as components; empty: the reference set
  seeds: [0, 1, 2]
  top_k: 100
  hyperparameters: {...}  # the xgb's; unset: quantlab's XGBoost defaults (us3000_ret5.yaml names SEL0's xgb_deep)
  factor_root: /data/quantlab/factors/market   # where export() writes the stores; an export needs it
```

- `validate()` trains the xgb per seed on the baseline features and on those plus every accepted
  factor (through the same `scoring` the Evaluator admitted with), up to the
  purged end of training, and predicts the purged validation segment. It reports, per seed and on
  average, rank IC, IC, decile monotonicity, D10−D1, and the fee-free plain TopK return and excess
  (`<run_dir>/acceptance/validation.{json,md}`).
- `open_holdout(force=False)` writes `<run_dir>/holdout.lock` (when, with which commits), then makes the
  same comparison on the hold-out, trained through validation. A second open raises `HoldoutLocked`;
  `force=True` runs it, appends to the lock, and the report says the hold-out was opened again.
- `export({"<description>": "<name>"})` writes `factor_agent/accepted/<name>.py` (a `GraphFactor`, i.e.
  `FactorKunQuant`, subclass fixing the scored graph) and the whole-market store
  `<factor_root>/<name>/` (`NeutralizedFactor` of the ranked graph) with `README.md` and `component.json`.

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
The campaign uses no GPU. `Acceptance` trains xgb on the GPU (quantlab picks CUDA); start it only when it
ends before 06:00 US Eastern on a weekday, when the GPU must be free.
