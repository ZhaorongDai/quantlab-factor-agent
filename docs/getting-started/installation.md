# Installation

quantlab-factor-agent needs Python 3.13 or newer and a C++ compiler on the `PATH` (KunQuant
compiles each round's factor graph to native code).

## Install from PyPI

```bash
pip install "quantlab-factor-agent[web]"
```

The `web` extra adds the [web frontend](../user-guide/web.md); leave it out for the library
alone. The package installs [quantlab](https://github.com/ZhaorongDai/quantlab) from PyPI
as `quantlab-py`. On macOS, xgboost needs Homebrew's `libomp` (`brew install libomp`).

## Install from a checkout

To change the code, run the example configs in `configs/` or the offline test suite, install
with [uv](https://docs.astral.sh/uv/) from a checkout, with quantlab's checkout next to it:

```text
projects/
├── quantlab/                 # editable dependency, ../quantlab
└── quantlab-factor-agent/
```

```bash
git clone https://github.com/ZhaorongDai/quantlab.git
git clone https://github.com/ZhaorongDai/quantlab-factor-agent.git
cd quantlab-factor-agent
uv sync --extra web           # installs quantlab from ../quantlab in editable mode
uv run pytest -n auto         # the offline test suite: no LLM call, a synthetic market
```

`uv sync --group docs` adds what building this site needs. A checkout follows quantlab's
`main`, which may be ahead of its latest release.

## Credentials

Keys come from environment variables only; a config that holds one is refused.

| Backend | Reads |
|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` |
| `openai_compatible` | the variable the role's `api_key_env` names (for example `DEEPSEEK_API_KEY`), or none for a local server |
| `claude_code` | the local Claude Code login |
| `replay` | nothing: it serves a recorded log |

## Check a config

The package installs the `factor-agent` command. Before starting a campaign, check its
config and run directory; the check writes nothing and computes no factor, and `--ping`
also sends each LLM backend one short request to confirm the model name, the base URL and
the key:

```bash
factor-agent check configs/your-campaign.yaml --ping
```

It prints one line per check and exits with status 1 when one found a problem (see
[checking a config](../user-guide/campaign.md#checking-a-config)). From a checkout, run it
as `uv run factor-agent check ...`.

## Data

A campaign reads quantlab stores: a whole-market price dataset, a forward-return label, a
membership dataset for the universe, the exposures to neutralise against, and the factor
stores of the reference set; see quantlab's [user guide](https://zhaorongdai.github.io/quantlab/user-guide/index.html)
for building them. The example configs in `configs/` read US equities from Sharadar under
one data root; point their paths at your own stores.

A campaign uses no GPU; KunQuant runs `evaluator.njobs` threads. Acceptance trains quantlab
models, which use CUDA when it is available.
