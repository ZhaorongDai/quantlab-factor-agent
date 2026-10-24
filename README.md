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
directory resumes from the last written round. Only the `scripted` LLM backend exists so far.
See [docs/design.md](docs/design.md).

## Setup

```
uv sync          # quantlab is an editable dependency at ../quantlab
uv run pytest
```

Credentials come from environment variables only (`ANTHROPIC_API_KEY`, or the variable named in the
config for an OpenAI-compatible provider); the `claude_code` backend uses the local Claude Code login.
