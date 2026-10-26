# First campaign

This page runs a campaign from one of the repository's configs, reads its outcome, and
judges the accepted factors with acceptance. The configs in `configs/` are complete
examples: `us3000_ret5.yaml` opens daily price and volume, `us3000_fund_ret5.yaml` adds
Sharadar fundamentals, and `us3000_hf_ret5.yaml` adds daily statistics of one-minute trade
bars.

## 1. Read the config

A campaign is one YAML file. Its main sections:

```yaml
run_dir: runs/my_campaign                    # everything the run writes
market: adjusted daily bars of every US common stock

segments:                                    # only training is ever shown to the LLM
  training:   {start: 2000-02-02, end: 2014-12-31}
  validation: {start: 2015-01-02, end: 2019-12-31}
  holdout:    {start: 2020-01-02, end: 2026-10-15}
  purge_bars: 5

data:       {prices: ..., label: ..., membership: ...}   # quantlab components or store folders
scoring:    {neutralize: {exposures: ..., size_column: marketcap, industry_column: industry}}
fields:     [fields/sharadar_price_volume.yaml]          # the input families candidates may read
operators:  operators/kunquant_example.yaml              # the operators they may write with
reference:  [...]                                        # the factors candidates must add to
known:      [...]                                        # other existing factors, checked for novelty
directions: [...]                                        # seed research directions for the planner

round:      {candidates: 20, groups: 4, max_rounds: 30, patience: 5}
llm:
  planner:   {backend: openai_compatible, model: deepseek-v4-pro, base_url: https://api.deepseek.com,
              api_key_env: DEEPSEEK_API_KEY, max_tokens: 32000}
  proposer:  {...}
  reviewer:  {...}
  distiller: {...}
acceptance: {model: {...}, seeds: [0, 1, 2]}
```

[The campaign config](../user-guide/campaign.md) describes every section. Copy a config,
change `run_dir`, and check the data paths before the first start.

## 2. Run the campaign

```python
from factor_agent.campaign import Campaign
from factor_agent.config import CampaignConfig

config = CampaignConfig.from_file("configs/us3000_ret5.yaml")
result = Campaign(config).run()
print(result.stop, len(result.accepted), "accepted of", len(result.records))
```

A campaign runs for hours, most of it waiting for the LLM, so start it in the background
and keep the log:

```bash
nohup uv run python -c \
  "from factor_agent.campaign import Campaign; from factor_agent.config import CampaignConfig; \
   Campaign(CampaignConfig.from_file('configs/us3000_ret5.yaml')).run()" > campaign.log 2>&1 &
```

Before any LLM call, `run()` checks the field, operator and template files and reports
every problem at once. The run stops at `round.max_rounds`, `round.max_trials`,
`round.patience` rounds in a row without an acceptance, or when the provider's balance is
exhausted. Calling `run()` again on the same `run_dir` resumes where it stopped.

## 3. Read the outcome

```python
import json

for line in open("runs/my_campaign/ledger.jsonl"):
    record = json.loads(line)
    if record["status"] == "accepted":
        metrics = record["metrics"]
        print(record["round"], record["candidate"]["description"][:80],
              f"IC {metrics['ic_mean']:+.4f}, partial IC t {metrics['partial_ic_t']:+.2f}")
```

Every candidate is one line of `ledger.jsonl`: accepted, rejected (by the reviewer or by
admission) or failed (refused by the parser or the Gate, or its graph did not compile),
with the stage where it ended and the reasons. [The run directory](../user-guide/campaign.md#the-run-directory)
lists the other files.

## 4. Judge the accepted factors

```python
from factor_agent.acceptance import Acceptance

acceptance = Acceptance(config)
report = acceptance.validate()            # baseline vs baseline + accepted, on validation
print(report.markdown())
```

`validate()` trains the acceptance model once per seed with and without the accepted
factors and compares rank IC, deciles and a plain TopK book on the validation segment.
Then, once, `open_holdout(selection)` makes the same comparison on the hold-out, and
`export({...})` writes the chosen factors as quantlab factors and stores. See
[acceptance](../user-guide/acceptance.md).
