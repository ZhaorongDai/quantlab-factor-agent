# quantlab-factor-agent

LLM-driven, mechanism-first formulaic factor research on [quantlab](https://github.com/ZhaorongDai/quantlab).
An LLM proposes economic-mechanism hypotheses and writes each factor as KunQuant code; candidates pass
deterministic gates (sandboxed graph build, originality, complexity, mechanism binding), are evaluated on
quantlab's whole-market panels for incremental value over the existing factor set, and only the training
segment is ever shown to the model.

Status: the multi-round research loop (#5). `Campaign(CampaignConfig.from_file(path)).run()` runs
rounds until a stop condition holds. Each round, every mechanism type gets `min_per_type` slots and
the rest go by Thompson sampling on each type's admission rate; the planner plans one direction per
slot (a new mechanism, an orthogonal mutation of an accepted factor, or a refinement of one); the
proposer writes one candidate per direction; the `Gate` builds each in a sandboxed subprocess and
refuses rewrites of the zoo (the KunQuant graphs of the `reference` factors plus the accepted ones),
parameter-only variants, large shared subtrees and hypotheses that name an input or transformation
the graph never uses, sending the refused ones back to be fixed; the reviewer, in its own call
(possibly another backend), judges each candidate's hypothesis, description and code for
consistency and an inconsistent one is rejected unevaluated. The `Evaluator` compiles the rest as
one KunQuant graph and scores them on the training segment: ranked over the whole market,
neutralised against industry and log market cap, masked to the day-t members, against the 5-day
label (rank IC, partial IC residualised on the reference and accepted factors, correlation with each,
deciles). Admission requires the stated sign, correlation at most 0.7, a partial-IC Newey-West t above
a threshold that rises with the evaluated trials, and monotone deciles.

The planner and the proposer see the data, the seed `directions`, the reference factor families, the
memory, every earlier trial, the last round's training statistics and rejection reasons, and the
accepted factors with their marginal weight and residual IC in the accepted pool; never a return, a
date or a held-out metric. Every `memory_every` rounds the distiller writes `memory.md`. The run
stops at `max_rounds`, `max_trials`, `patience` rounds without an acceptance, or `budget_usd` (checked before every call), and
records why in `state.json`; `ledger.jsonl` holds every candidate, and `run()` on the same run
directory resumes from the last written round. See [docs/design.md](docs/design.md).

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
- Before the first round `run()` prints the estimated cost per role against `budget_usd`
  (`Campaign(config).estimate()`); the run stops once the calls have cost the budget.

## Setup

```
uv sync          # quantlab is an editable dependency at ../quantlab
uv run pytest
```

Credentials come from environment variables only (`ANTHROPIC_API_KEY`, or the variable named in the
config for an OpenAI-compatible provider); the `claude_code` backend uses the local Claude Code login.
