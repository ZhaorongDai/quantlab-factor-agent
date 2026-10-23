# quantlab-factor-agent

LLM-driven, mechanism-first formulaic factor research on [quantlab](https://github.com/ZhaorongDai/quantlab).
An LLM proposes economic-mechanism hypotheses and writes each factor as KunQuant code; candidates pass
deterministic gates (sandboxed graph build, originality, complexity, mechanism binding), are evaluated on
quantlab's whole-market panels for incremental value over the existing factor set, and only the training
segment is ever shown to the model.

Status: tracer bullet (#9). `Campaign(CampaignConfig.from_file(path)).run()` runs one round: the
proposer's candidates are parsed, built in a sandboxed subprocess by the `Gate`, compiled as one
KunQuant graph and scored by the `Evaluator` on the training segment (daily rank IC and Newey-West t
against the 5-day label on the day-t members), admitted at t >= 3 and written to `ledger.jsonl` in
the run directory. Only the `scripted` LLM backend exists so far. See [docs/design.md](docs/design.md).

## Setup

```
uv sync          # quantlab is an editable dependency at ../quantlab
uv run pytest
```

Credentials come from environment variables only (`ANTHROPIC_API_KEY`, or the variable named in the
config for an OpenAI-compatible provider); the `claude_code` backend uses the local Claude Code login.
