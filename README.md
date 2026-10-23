# quantlab-factor-agent

LLM-driven, mechanism-first formulaic factor research on [quantlab](https://github.com/ZhaorongDai/quantlab).
An LLM proposes economic-mechanism hypotheses and writes each factor as KunQuant code; candidates pass
deterministic gates (sandboxed graph build, originality, complexity, mechanism binding), are evaluated on
quantlab's whole-market panels for incremental value over the existing factor set, and only the training
segment is ever shown to the model.

Status: design only. See [docs/design.md](docs/design.md).

## Setup

```
uv sync          # quantlab is an editable dependency at ../quantlab
uv run pytest
```

Credentials come from environment variables only (`ANTHROPIC_API_KEY`, or the variable named in the
config for an OpenAI-compatible provider); the `claude_code` backend uses the local Claude Code login.
