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
