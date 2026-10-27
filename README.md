# quantlab-factor-agent

<p>
  <a href="https://www.python.org/downloads/"><img alt="Python 3.13+" src="https://img.shields.io/badge/python-3.13%2B-3776ab?logo=python&logoColor=white"></a>
  <a href="https://github.com/ZhaorongDai/quantlab"><img alt="Built on quantlab" src="https://img.shields.io/badge/built%20on-quantlab-3776ab"></a>
  <a href="https://github.com/Menooker/KunQuant"><img alt="Factors: KunQuant" src="https://img.shields.io/badge/factors-KunQuant-0ea5e9"></a>
</p>

quantlab-factor-agent mines daily stock factors with language models, mechanism first. A
model states an economic mechanism and writes the factor it implies as KunQuant code; a
fixed pipeline decides whether the factor is new and adds to the factors you already have.
The language model only proposes and reviews. Every decision to accept is made by
deterministic checks and by statistics that [quantlab](https://github.com/ZhaorongDai/quantlab)
computes on the training segment, which is all the model is ever shown.

- **Documentation:** built locally from `docs/` (see [docs/README.md](docs/README.md)):
  `uv sync --group docs && uv run sphinx-build -b html docs docs/_build/html`
- **Example configs:** [configs/](configs/)

## The research loop

```text
            +-----------+   directions   +-----------+   candidates   +--------+
 ledger --> |  planner  | -------------> |  proposer | -------------> |  Gate  | -- refused: fix and retry
 memory     +-----------+                +-----------+                +--------+
                                                                          |
                    accepted <-- admission <-- Evaluator <-- reviewer <---+
                       |                       (training     (hypothesis,
                       +--> zoo, pool, ledger    segment)     code agree?)
```

A campaign runs in rounds. Each round:

| Step | Who | What it decides | Main classes |
|------|-----|-----------------|--------------|
| 1. Plan | planner (LLM) | one research direction per slot: a new mechanism, a mutation or a refinement of an accepted factor | `Campaign`, `Researcher` |
| 2. Propose | proposer (LLM) | a hypothesis, a description and KunQuant code per direction, in parallel proposal groups | `Researcher`, `Candidate` |
| 3. Gate | code | the code builds in a sandbox, reads only the vocabulary, and is not a rewrite of an existing factor | `Gate`, `Vocabulary`, `Zoo` |
| 4. Review | reviewer (LLM) | the hypothesis, the description and the code describe the same factor | `Researcher` |
| 5. Evaluate | code | IC, partial IC, correlations, deciles and coverage on the training segment | `Evaluator`, `Scoring` |
| 6. Admit | code | correlation, partial-IC t (Bonferroni over the trials), monotonicity and coverage thresholds | `Campaign` |

After the campaign, `Acceptance` trains a quantlab return model with and without the
accepted factors on the validation segment, opens the hold-out once, and exports the chosen
factors as quantlab factors and whole-market stores.

## Why quantlab-factor-agent

- **Mechanisms, not formula search.** The agent is asked for economic mechanisms by
  mechanism type, and the Gate refuses rewrites, parameter-only variants and large shared
  subtrees of every reference, known and accepted factor, compared by their KunQuant graphs.
- **The model never sees the future.** Prompts carry training-segment statistics only: no
  return, no date, no validation or hold-out metric. The hold-out opens once, behind a lock.
- **Deterministic judges.** The reviewer can send a candidate back, but only code accepts:
  the Evaluator scores every candidate on quantlab's whole-market panels, masked to the
  day's members, against the reference pool, and admission's thresholds tighten with every
  evaluated trial.
- **Nothing hard-coded about the market.** Fields, operators, parameter sets, prompt
  templates, mechanism types and the scoring all come from the config's files, so a new
  input family is a new field file.
- **Any model, any provider.** Each role picks its backend: OpenAI-compatible endpoints
  (DeepSeek, OpenAI, vLLM, ...), Anthropic, or Claude Code. Every call is logged in full, and
  the `replay` backend reproduces a run byte for byte without a provider.
- **Resumable runs.** A run directory holds the ledger, the state, the LLM log and a
  snapshot of the search files; `run()` resumes after a crash or an exhausted balance.

## Install

```bash
# next to a quantlab checkout: projects/quantlab and projects/quantlab-factor-agent
uv sync              # quantlab is an editable dependency at ../quantlab
uv run pytest -n auto
```

Credentials come from environment variables only: `ANTHROPIC_API_KEY`, or the variable a
role's `api_key_env` names for an OpenAI-compatible provider; the `claude_code` backend uses
the local Claude Code login. A config that holds a key is refused.

## Run a campaign

```python
from factor_agent.acceptance import Acceptance
from factor_agent.campaign import Campaign
from factor_agent.config import CampaignConfig

config = CampaignConfig.from_file("configs/us3000_ret5.yaml")
result = Campaign(config).run()          # resumes when run_dir already holds a state
print(result.stop, len(result.accepted), "accepted of", len(result.records))

report = Acceptance(config).validate()   # baseline vs baseline + accepted, on validation
print(report.markdown())
```

A campaign runs for hours, most of it waiting for the language model, and uses no GPU;
`Acceptance` trains quantlab models, which use CUDA when it is available. The configs in
`configs/` are complete examples; point their data paths at your own quantlab stores.

## Documentation

| Page | Covers |
|------|--------|
| [Installation](docs/getting-started/installation.md), [first campaign](docs/getting-started/quickstart.md) | setup, credentials, running and reading a campaign |
| [Concepts](docs/user-guide/concepts.md) | roles, factor sets, segments, direction, what "accepted" means |
| [The campaign config](docs/user-guide/campaign.md) | every section and default, stopping, resuming, the run directory |
| [The vocabulary](docs/user-guide/vocabulary.md) | field files, the operator file, fixes, what code may write |
| [Prompts](docs/user-guide/prompts.md) | mechanism types, templates and their placeholders |
| [LLM backends](docs/user-guide/llm-backends.md) | backends, the LLM log, replay, cost and quota |
| [Gate, review and evaluation](docs/user-guide/evaluation.md) | the Gate's checks, the review, the statistics, admission |
| [Acceptance](docs/user-guide/acceptance.md) | validation, contribution, the hold-out, export |
| [The web frontend](docs/user-guide/web.md) | `factor-agent-web`: your runs in a browser on localhost, also over SSH |
