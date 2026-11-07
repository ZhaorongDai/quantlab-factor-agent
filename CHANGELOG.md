# Changelog

quantlab-factor-agent follows [semantic versioning](https://semver.org/). While the version
is 0.x, any minor release may break the API, the config format or the run directory's
layout; there are no compatibility shims or config migrations.

## Unreleased

### Added

- `Campaign.check()` reports what `run()` would refuse before its first LLM call, every
  problem at once, without writing to the run directory or computing a factor: a data
  component that does not load or reads a store that does not exist, the scoring, a role
  whose backend cannot be built (an unset API key variable), the reference and known
  factors, a resume's files or settings that differ from the run snapshot, the vocabulary
  and the templates; a check that needs a part that failed is reported as skipped with the
  reason, and other code versions on a resume as a warning (`factor_agent.preflight`).
- `factor-agent check <config> [--ping]`, a new console script beside `factor-agent-web`,
  prints the report of `Campaign.check()` and exits with status 1 when it holds a problem;
  a config that cannot be read is printed as its first problem instead of a traceback.
  With `--ping` (`Campaign.check(ping=True)`) each distinct backend gets one short request
  outside the LLM log and the report gives its duration or its error; roles on the replay
  or scripted backend are not pinged.
- `context.lineage`: the planner's accepted factors and the proposer's parent trial carry
  each trial's Lineage, one line on the trials built on it: how many, refinements and
  mutations, how many stopped before evaluation, which were accepted, and the best child's
  residual t against the parent's. Off by default and fixed by the run's snapshot; it needs
  `context.feedback`.
- `round.max_children: n`: a trial that n trials have built on, those that stopped before
  evaluation included, is taken off the trials a refinement or a mutation may name, and a
  direction naming it is planned as a new one with a plan issue. Unset by default; set, it
  is fixed by the run's snapshot and needs `context.feedback`.
- `round.picked_parents: k`: each round the code chooses the parents of k slots, the
  Picked parents (`factor_agent._trial_graph`): the evaluated trials ranked by Pareto
  front and crowding distance on their quality (residual t in their direction), fertility
  (their Lineage's) and novelty (distance from the accepted factors), each shared over the
  trials' lineage, structure and correlation edges, a trial built on five times left out.
  Each takes a slot of its own mechanism type while that type has one unclaimed; the
  planner is told the picked slots (the plan template's new `{picked}` placeholder) and
  writes each as a refinement or a mutation of its parent, and one it leaves out is planned
  as a refinement of its parent with a plan issue. The plan event records the picked
  parents with their utilities and fronts (`Plan.picked`, empty when off). 0 by default;
  above 0 it needs `context.theme_correlation`, a replaced plan template without `{picked}`
  is refused at start and by `Campaign.check()`, and it is fixed by the run's snapshot.
- The web panel's trials page shows the run's Lineage: a round stepper with any round's
  plan (its slots with their action, parent and outcome, which parents the code picked, the
  quota and the plan's issues), the round in progress included, and a Lineage graph of every
  trial in a column of its round, its action as the node's shape, its outcome as the colour,
  an arrow from each parent (solid when the code picked it, dashed when the planner chose
  it) and a ring per round a trial was a Picked parent. The side reader gains the trial's
  Lineage and a close button, and a trial selected on the graph or the admission map is
  marked on both. `GET /api/run/round` reads one round's plan, and a run's trials carry
  their Lineage, the rounds they were picked in and whether their parent was picked.

### Changed

- `CampaignConfig.from_file` refuses an LLM role option its backend does not take, naming
  the role and listing the options the backend takes, and a segment entry key other than
  `start` and `end`, instead of dropping them silently. Each backend declares its options
  (`OPTIONS`), and `factor_agent.llm.base.backends()` is the one table of backends that
  `make_backend` builds from and `backend_options` looks options up in. An option that
  looks like a key (`api_key`, say) is refused there too, saying to name an environment
  variable with `api_key_env`.
- `CampaignConfig.load_factors` names every reference or known entry that cannot be
  rebuilt, by its section, position and store, with the cause (an unimportable class,
  say), instead of raising the first entry's bare exception; `Campaign.check()` reports
  each (#80).

## 0.1.0 (2026-11-06)

The first public version.

### Added

- `Campaign`: the research loop in rounds of plan, propose, Gate, review, evaluate and
  admit, with a planner, a proposer, a reviewer and a distiller on any LLM backend, and a run
  directory that holds the ledger, the state, the LLM log and a snapshot of the search
  files, so that `run()` resumes after a crash or an exhausted balance. A resume reads the
  accepted factors' and the evaluated trials' scored values from the run directory's
  `columns/` instead of computing them again.
- `Gate`: candidate code builds its KunQuant graph in a sandboxed subprocess, reads only the
  vocabulary, and is refused when it rewrites, varies only the parameters of, or shares a
  large subtree with a reference, known or accepted factor.
- `Evaluator` and admission: IC, partial IC, correlations, deciles and coverage on the
  training segment, on quantlab's whole-market panels masked to the day's members, with a
  partial-IC t threshold that tightens with the number of trials, and an optional
  confirmation segment.
- `Acceptance`: a quantlab return model with and without the accepted factors on the
  validation segment, a hold-out opened once behind a lock, and export of the chosen factors
  as quantlab factors and whole-market stores.
- Field files for Sharadar price and volume, Sharadar fundamentals, daily session statistics
  and one-minute trade bars, on which intraday candidates compute their own daily values.
- LLM backends: OpenAI-compatible endpoints, Anthropic, Claude Code, a scripted backend for
  tests, and `replay`, which reproduces a run from its log without a provider.
- `factor-agent-web` (extra `web`): a read-only page over the run directories on localhost,
  also over an SSH tunnel.
- Example configs for US equities under `configs/`.
