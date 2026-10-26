# The campaign config

One YAML file holds every setting of a campaign. `CampaignConfig.from_file(path)` reads it,
resolving the field, operator and template paths relative to the file, and refuses
unknown sections, malformed entries and credentials. `Campaign(config).run()` runs it.

## Sections

| Section | Required | Holds |
|---|---|---|
| `run_dir` | yes | the directory the run writes to |
| `market` | yes | the data as the prompts describe it, one line |
| `segments` | yes | `training`, `validation`, `holdout` (`{start, end}` each) and `purge_bars` |
| `data` | yes | `prices`, `label` and optionally `membership` |
| `scoring` | yes | `neutralize`: the exposures to neutralise against, or `null` |
| `fields`, `operators` | yes | the [vocabulary](vocabulary.md) files; `operator_modules` optionally |
| `reference` | no | the reference set; empty, candidates are only compared with each other |
| `known` | no | the known factors |
| `directions` | no | seed research directions for the planner, one string each |
| `mechanisms`, `prompts` | no | mechanism types and replaced templates; see [prompts](prompts.md) |
| `evaluator`, `gate`, `admission`, `round`, `context` | no | settings with defaults (below) |
| `llm` | yes | one entry per role; see [LLM backends](llm-backends.md) |
| `acceptance` | yes | the acceptance model and seeds; see [acceptance](acceptance.md) |

## Data

`data`, `reference` and `known` hold quantlab components. An entry is one of:

- a component's `get_config()` dict, written inline;
- the path of a store folder, whose `component.json` describes the factor that owns it;
- `{store: <folder>, factor_names: [...]}`: that store's factor, pinned to some of its outputs.

```yaml
data:
  prices: {...}                                   # the whole-market price dataset
  label: /path/to/quantlab/labels/market/ret_5       # a forward label store
  membership: {...}                               # the members of each day, or omitted
scoring:
  neutralize:
    exposures: {...}
    size_column: marketcap                        # its logarithm is regressed on
    industry_column: industry                     # an integer industry code
reference:
  - /path/to/quantlab/factors/market/alpha158_neutral
  - {store: /path/to/quantlab/factors/market/barra_style, factor_names: [size, beta]}
known:
  - /path/to/quantlab/factors/market/alpha101_neutral
```

Factors are computed on the whole market, then masked to the day's members. A candidate
is ranked across the whole market, neutralised when `scoring.neutralize` is given, and
masked; a reference or known entry given as a store is read as stored and masked, one
given as a component without a store is scored as a candidate is. `scoring.neutralize`
must be written, as `null` to score ranked values, so no campaign is neutralised by
default.

An output of a known factor that the reference set already holds is taken from the
reference set only, so whole classes can be listed in `known`.

Every stored reference or baseline factor is read from its store's first bar, so a store
that starts after the training segment (an intraday store from 2016, say) is empty there.

The Evaluator keeps the known factors only on one scored training date in five, as normal
scores in a float32 panel of its own: 4 bytes per known output per sampled member-day. For
us3000 (training 2000 to 2014, about 750 sampled dates of about 3,000 members) that is
about 9 MB per known output, about 2.7 GB for 300 outputs. It is built once per start, one
known factor at a time.

## Settings with defaults

| Key | Default | Meaning |
|---|---|---|
| `evaluator.warmup_bars` | 260 | bars read before the segment so every window is filled |
| `evaluator.njobs` | 51 | KunQuant executor threads |
| `gate.timeout_seconds` | 10 | seconds the sandboxed graph build may take |
| `gate.memory_mb` | 1024 | memory the sandboxed build may use |
| `gate.max_nodes` | 200 | nodes a candidate's graph may have |
| `gate.max_free_parameter_ratio` | 0.5 | hand-set constants (other than 0 and ±1) per operator |
| `gate.max_base_fields` | 5 | distinct input fields a candidate may read |
| `gate.max_common_subtree_nodes` | 5 | the largest subtree a candidate may share with a zoo factor |
| `admission.t_threshold` | 3.0 | partial-IC t the first evaluated trial needs |
| `admission.max_correlation` | 0.7 | largest absolute correlation with a reference, accepted or known factor |
| `admission.min_monotonicity` | 0.6 | smallest Spearman correlation of the deciles with the decile number |
| `admission.min_coverage` | 0.5 | smallest mean daily share of the members with a value |
| `round.candidates` | 20 | candidates per round |
| `round.min_per_type` | 3 | slots of every mechanism type per round |
| `round.retries` | 2 | times a refused candidate goes back to the proposer |
| `round.groups` | 1 | proposal groups per round (the repository's configs use 4) |
| `round.max_rounds` | 30 | rounds at most |
| `round.max_trials` | 600 | trials at most; a round that would pass it is not started |
| `round.patience` | 5 | rounds in a row without an acceptance before the run stops |
| `round.memory_every` | 5 | rounds between two distillations |
| `round.seed` | 0 | seed of the Thompson sampling |
| `context.recent_rounds` | 5 | rounds whose trials the planner reads one line each |

## Stopping and resuming

Before each round the run checks its stop conditions: `max_rounds`, `max_trials` and
`patience`. The first that holds ends the run, and `state.json` records which one. When a
provider refuses a call because the balance or quota is exhausted, the run drops the
round in progress, records the stop `provider_quota` with the provider's message and
returns normally; the library counts no money, so spending limits belong to the provider.

`run()` on a run directory that has a state resumes it: the rounds written are kept, a
round a crash left half written is dropped and run again, and the zoo, the pool and the
graphs the Gate compares with are rebuilt from the ledger. The calls of a dropped round
move to `llm_log/dropped/`, so the log holds exactly the calls the ledger was written from.

The first start copies the field, operator and replaced template files into
`snapshot/` with their hashes, and the run searches with those copies. A resume whose
files differ fails: a new search space needs a new run directory. A resume on other
commits of factor_agent or quantlab only warns, and `state.json` lists the commits of every
start and resume. `round.groups` may change on a resume; `context.recent_rounds` may not.
The reference set and the known factors are read from the config on every start.

## The run directory

| Path | Holds |
|---|---|
| `ledger.jsonl` | one line per candidate: round, candidate, status, stage, reasons, the Gate's statistics, the graph, the metrics, refused attempts |
| `state.json` | rounds written, the stop, the memory's round, the code versions of every start |
| `llm_log/` | every LLM call in full: prompt, parameters, reply, token usage, the raw reply |
| `memory.md`, `memory/round_NN.md` | the distilled memory and every version of it |
| `snapshot/`, `snapshot.json` | the search files of the first start and their hashes |
| `zoo.json` | the reference and known factors the Gate covers, and those without a graph it cannot |
| `acceptance/` | validation and hold-out reports and the model cache |
| `holdout.lock` | written when the hold-out is opened |

A trial's `status` is `accepted`, `rejected` (by the reviewer, or by admission after
evaluation) or `failed` (refused by the parser or the Gate after every retry, or its
graph did not compile). Its `stage` is where it ended: `parse`, `gate`, `review`,
`evaluate` or `admission`.

## Memory and context

The planner and the proposer read the same context: the seed directions, the reference
factor families and then the known ones (marked as known), the memory, the last round's
training statistics and rejection reasons, and the accepted factors with their weight and
residual IC in the accepted pool. The planner also reads the trials of the last
`context.recent_rounds` rounds one line each, and the older trials counted by mechanism
type and outcome. Every `round.memory_every` rounds the distiller folds the trials since
the last memory into `memory.md`; an empty or unparsable reply keeps the previous memory.
