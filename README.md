# quantlab-factor-agent

LLM-driven, mechanism-first formulaic factor research on [quantlab](https://github.com/ZhaorongDai/quantlab).
An LLM proposes economic-mechanism hypotheses and writes each factor as KunQuant code; candidates pass
deterministic gates (sandboxed graph build, originality, complexity, mechanism binding), are evaluated on
quantlab's whole-market panels for incremental value over the existing factor set, and only the training
segment is ever shown to the model.

Status: tracer bullet (#9). `Campaign(CampaignConfig.from_file(path)).run()` runs one round: the
proposer's candidates are parsed, built in a sandboxed subprocess by the `Gate`, compiled as one
KunQuant graph (a batch that fails is bisected to the failing graphs) and scored by the `Evaluator` on
the training segment: each candidate is ranked over the whole market, neutralised against industry and
log market cap like the shared `*_neutral` stores and masked to the day-t members, then measured
against the 5-day label (rank IC, partial IC residualised on the reference set and the accepted
factors, correlation with each of them, decile returns). Admission requires, in order, the stated
sign, correlation at most 0.7, a partial-IC Newey-West t above a threshold that starts at 3 and rises
with the number of evaluated trials, and monotone deciles; every candidate is written to
`ledger.jsonl` in the run directory. The Gate also rejects a candidate that rewrites a factor of the zoo (the KunQuant
graphs of the `reference` factors in the config plus the factors accepted in earlier rounds), changes
only its windows or parameters, or shares a subtree of more than 5 nodes with one, and a hypothesis
that names an input or transformation the graph never uses; `zoo.json` lists the reference factors
without a KunQuant graph, which the graph comparison cannot cover. Only the `scripted` LLM backend exists so far. See [docs/design.md](docs/design.md).

## Setup

```
uv sync          # quantlab is an editable dependency at ../quantlab
uv run pytest
```

Credentials come from environment variables only (`ANTHROPIC_API_KEY`, or the variable named in the
config for an OpenAI-compatible provider); the `claude_code` backend uses the local Claude Code login.
