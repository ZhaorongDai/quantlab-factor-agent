# Installation

quantlab-factor-agent needs Python 3.13 or newer, [uv](https://docs.astral.sh/uv/), a C++
compiler on the `PATH` (KunQuant compiles each round's factor graph to native code) and a
checkout of [quantlab](https://github.com/ZhaorongDai/quantlab) next to its own:

```text
projects/
├── quantlab/                 # editable dependency, ../quantlab
└── quantlab-factor-agent/
```

## Install

```bash
git clone https://github.com/ZhaorongDai/quantlab.git
git clone https://github.com/ZhaorongDai/quantlab-factor-agent.git
cd quantlab-factor-agent
uv sync                       # installs quantlab from ../quantlab in editable mode
uv run pytest -n auto         # the offline test suite: no LLM call, a synthetic market
```

`uv sync --extra web` adds the [web frontend](../user-guide/web.md), and
`uv sync --group docs` what building this site needs. On macOS, xgboost needs Homebrew's
`libomp` (`brew install libomp`).

quantlab-factor-agent follows quantlab's `main`, which is newer than the quantlab release on
PyPI, so it is installed from a checkout rather than with `pip install`.

## Credentials

Keys come from environment variables only; a config that holds one is refused.

| Backend | Reads |
|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` |
| `openai_compatible` | the variable the role's `api_key_env` names (for example `DEEPSEEK_API_KEY`), or none for a local server |
| `claude_code` | the local Claude Code login |
| `replay` | nothing: it serves a recorded log |

## Data

A campaign reads quantlab stores: a whole-market price dataset, a forward-return label, a
membership dataset for the universe, the exposures to neutralise against, and the factor
stores of the reference set; see quantlab's [user guide](https://zhaorongdai.github.io/quantlab/user-guide/index.html)
for building them. The example configs in `configs/` read US equities from Sharadar under
one data root; point their paths at your own stores.

A campaign uses no GPU; KunQuant runs `evaluator.njobs` threads. Acceptance trains quantlab
models, which use CUDA when it is available.
