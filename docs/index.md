# quantlab-factor-agent

**Version {{ release }}**

quantlab-factor-agent mines daily stock factors with language models. A model states an
economic mechanism, writes the factor it implies as KunQuant code, and a fixed pipeline
decides whether the factor is new and adds to the factors you already have. The language
model only proposes and reviews; every decision to accept is made by deterministic checks
and statistics computed by [quantlab](https://zhaorongdai.github.io/quantlab/) on the
training segment, which is all the model is ever shown.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Getting started
:link: getting-started/index
:link-type: doc

Install the package next to quantlab and run a first campaign from a config file.
:::

:::{grid-item-card} User guide
:link: user-guide/index
:link-type: doc

The research loop, the campaign config, the vocabulary, the prompts, the LLM backends,
evaluation and acceptance.
:::

:::{grid-item-card} API reference
:link: reference/index
:link-type: doc

Classes and functions: `Campaign`, `CampaignConfig`, `Gate`, `Evaluator`, `Acceptance`
and the LLM backends.
:::

:::{grid-item-card} Glossary
:link: https://github.com/ZhaorongDai/quantlab-factor-agent/blob/main/GLOSSARY.md

The project's terms: input family, field, vocabulary, proposal group, known factor and
the rest.
:::
::::

## What a campaign does

```text
            +-----------+   directions   +-----------+   candidates   +--------+
 ledger --> |  planner  | -------------> |  proposer | -------------> |  Gate  | -- refused: fix and retry
 memory     +-----------+                +-----------+                +--------+
                                                                          |
                    accepted <-- admission <-- Evaluator <-- reviewer <---+
                       |                       (training     (hypothesis,
                       +--> zoo, pool, ledger    segment)     code agree?)
```

| Step | Who | Decides |
|---|---|---|
| Plan | planner (LLM) | one research direction per slot: a new mechanism, a mutation or a refinement of an accepted factor |
| Propose | proposer (LLM) | a hypothesis, a description and KunQuant code for each direction |
| Gate | code | the code builds in a sandbox, reads only the vocabulary, is not a rewrite of an existing factor |
| Review | reviewer (LLM) | the hypothesis, description and code describe the same factor |
| Evaluate | code | IC, partial IC, correlations, deciles and coverage on the training segment |
| Admit | code | the factor clears the correlation, partial-IC t, monotonicity and coverage thresholds |

After a campaign, [acceptance](user-guide/acceptance.md) trains a quantlab return model
with and without the accepted factors on the validation segment, opens the hold-out once,
and exports the chosen factors as quantlab factors and whole-market stores.

```{toctree}
:hidden:
:maxdepth: 2

getting-started/index
user-guide/index
reference/index
```
