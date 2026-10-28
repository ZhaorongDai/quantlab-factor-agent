# Acceptance

Admission judges each factor alone, on the training segment. Acceptance judges the
accepted factors together, by what they add to a return model out of sample:
`Acceptance(config)` trains the acceptance model with and without them on the validation
segment, opens the hold-out once, and exports the chosen factors.

## Configure

```yaml
segments:
  training:   {start: 2000-02-02, end: 2014-12-31}
  validation: {start: 2015-01-02, end: 2019-12-31}
  holdout:    {start: 2020-01-02, end: 2026-10-23}
  purge_bars: 5             # at least; a label that reads further purges more
acceptance:
  model:                    # required: the get_config() of a quantlab BaseModel or ModelEnsemble
    name: quantlab.model.predefined.xgb.XGBoostRegressor
    factors: []             # the baseline, as reference entries; empty: the reference set
    hyperparameters: {...}
  seeds: [0, 1, 2]          # each replaces every model's (every member's) random_seed
  contribution: none        # none, drop_one, add_one or both
  top_k: 100
  factor_root: /path/to/quantlab/factors/market   # where export() writes the stores
```

The model is a template. Acceptance fills in the label, the windows, `val_size`, the read
strategies and the seed, and appends the accepted factors to every member; what the
template says of those is replaced, so a trained run's `config.json` serves as one. A
`ModelEnsemble` is `{name: ...ModelEnsemble, members: [<model>, ...]}`, each member with
its own baseline. A `SeedEnsemble` is refused: the seeds are `acceptance.seeds`.

## Validate

```python
from factor_agent.acceptance import Acceptance
from factor_agent.config import CampaignConfig

acceptance = Acceptance(CampaignConfig.from_file("configs/us3000_ret5.yaml"))
report = acceptance.validate()                                  # every accepted factor
narrow = acceptance.validate(["<description> (accepted in round 5)", "..."])
```

`validate(selection=None)` trains the model per seed on its baseline and on its baseline
plus the selection, up to the purged end of training, and predicts the purged validation
segment. It reports, per seed and on average, rank IC, IC, decile monotonicity, D10−D1 and
the fee-free plain TopK return and excess, in `<run_dir>/acceptance/validation.{json,md}`
(a narrower selection writes `validation_<digest>.{json,md}`). A factor is named by its
description, or `description (accepted in round n)` when two share one.

With `contribution`, validation also measures each factor of the selection with the same
seeds: drop-one trains without it, add-one on the baseline plus only it. The report gains
a table, one row per factor and mode, sorted by the rank IC difference, with how many
seeds agree in sign. There is no threshold: you choose the selection.

Every trained model's result is cached under `<run_dir>/acceptance/<segment>/cache/`,
keyed by its filled-in config, the code versions and the scoring, so a re-run trains only
what is missing and equal feature sets share their models. Before training, `validate()`
prints how many models it will train, how many are cached, and an estimated duration once
some have finished.

## Open the hold-out

```python
holdout = acceptance.open_holdout(selection)
```

`open_holdout(selection, force=False)` needs `validate`'s report of exactly that
selection, writes `<run_dir>/holdout.lock`, and makes the same comparison on the hold-out,
trained through validation. A second open, of any selection, raises `HoldoutLocked`;
`force=True` runs it, appends to the lock, and the report says the hold-out was opened
again. The hold-out never measures contribution.

An intraday factor is validated, opened on the hold-out and exported in its stages: its
session columns are computed from the intraday bars over each segment.

## Export

```python
acceptance.export({"<description>": "my_factor_name"})
```

`export` writes, for each chosen factor, a module `factor_agent/accepted/<name>.py` (a
`GraphFactor` subclass fixing the scored graph, a quantlab `FactorKunQuant`) and the
whole-market store `<factor_root>/<name>/` (a quantlab `NeutralizedFactor` of the ranked
graph) with `README.md` and `component.json`. The README quotes validation's report of the
exported selection, when there is one. Every exported factor is turned to its direction: a
higher value, a higher return. An intraday factor's class also holds its session columns'
graphs (`SESSIONS`) and the intraday dataset (`INTRADAY`), and recomputes from the
intraday bars; see [export and recompute](intraday.md#export-and-recompute).
