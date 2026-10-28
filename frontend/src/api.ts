// The JSON of factor-agent-web's routes (factor_agent/web/run_view.py's dataclasses).

export type Stop = { by: string; reason: string }

export type RunSummary = {
  name: string
  code_version: string
  rounds: number
  trials: number
  accepted: number
  stop: Stop | null
  updated: string
  // Which file of the run could not be read and why; null for a run the list read.
  unreadable: string | null
  // 'unknown' when the campaign's process cannot be checked: no events file, or another host.
  state: 'stopped' | 'running' | 'crashed' | 'unknown'
  max_rounds: number | null
}

// A trial's refusal for being too like an existing factor (code: the Gate; ranking: admission).
export type Collision = {
  label: string
  name: string
  kind: 'code' | 'ranking'
  accepted_round: number | null
  known: boolean
}

export type Trial = {
  round: number
  slot: number
  status: 'accepted' | 'rejected' | 'failed'
  stage: string
  reasons: string[]
  candidate: Record<string, unknown>
  direction: Record<string, string> | null
  attempts: Record<string, unknown>[]
  stats: Record<string, number> | null
  metrics: Record<string, unknown> | null
  collisions: Collision[]
  // Whether the graph the Gate built holds a session aggregation; false without a graph.
  intraday: boolean
}

export type LineChange = { change: 'kept' | 'added' | 'removed'; line: string }
export type MemoryLine = LineChange

// The config of one start or resume of a run; changes is null for the earliest kept.
export type ConfigCopy = { opening: number; opened_at: string | null; text: string; changes: LineChange[] | null }
// A file of the run snapshot, named fields/<file>, operators/<file> or prompts/<file>.
export type SnapshotFile = { name: string; text: string }
// The configs, the latest opening first (none for a run started before configs were kept), and the snapshot.
// A trial's daily rank IC and partial IC, raw, on each day of the label; null on an unscored day.
export type TrialIC = { days: string[]; ic: (number | null)[]; partial_ic: (number | null)[] }

export type RunFiles = { configs: ConfigCopy[]; snapshot: SnapshotFile[] }

export type MemoryVersion = { round: number; text: string; changes: MemoryLine[] | null }

export type Neighbour = { label: string; name: string; accepted_round: number | null; correlation: number | null }

export type Factor = {
  label: string
  round: number
  slot: number
  mechanism_type: string | null
  hypothesis: string
  description: string
  code: string
  factor_direction: 1 | -1
  ic_mean: number | null
  ic_t: number | null
  partial_ic_mean: number | null
  partial_ic_t: number | null
  threshold: number | null
  // admission: its round's admission events; refusals: what its round's refusals quote; carried_over: an earlier round's.
  threshold_source: 'admission' | 'refusals' | 'carried_over' | null
  monotonicity: number | null
  nearest: Neighbour | null
  nearest_known: Neighbour | null
  decile_returns: (number | null)[]
  coverage: number | null
  metrics: Record<string, unknown>
  // Whether its graph holds a session aggregation.
  intraday: boolean
}

export type Tokens = { input: number | null; output: number | null; cache_hits: number | null; reasoning: number | null }

export type Health = {
  phase: 'planning' | 'asking' | 'evaluating' | null
  round: number | null
  max_rounds: number | null
  process: 'running' | 'gone' | 'unknown'
  crashed: boolean
  last_call: string | null
  stalled: boolean
  calls: Record<string, number>
  time_left: number | null
  tokens: Tokens | null
  round_tokens: { round: number; tokens: Tokens }[]
}

// One check of a pending trial's candidate: the parser and the Gate on one version, or the reviewer's verdict.
export type Check = {
  by: 'gate' | 'reviewer'
  step: 'proposed' | 'fixed' | 'rewritten' | null
  passed: boolean
  stage: 'parse' | 'gate' | null
  reasons: string[]
  hints: string[]
  time: string
}

export type Stage = 'planned' | 'proposed' | 'fixed' | 'rewritten' | 'consistent' | 'inconsistent'

// What is known so far of one slot of the round in progress.
export type PendingTrial = {
  slot: number
  stage: Stage
  direction: Record<string, string>
  latest: unknown
  checks: Check[]
}

// An LLM call of the round in progress: `number` (its call file's) is null while it is out, or when it failed.
export type RoundCall = {
  number: number | null
  role: string
  model: string
  group: number | null
  slots: number[]
  sent: string
  returned: string | null
  error: string | null
}

export type RunningRound = {
  round: number
  started: string
  calls: RoundCall[]
  evaluation: string | null
  pending: PendingTrial[]
}

// One LLM call in full, read by its call file's number.
export type LLMCall = {
  number: number
  role: string
  model: string
  round: number | null
  group: number | null
  slots: number[]
  time: string
  seconds: number | null
  tokens: Tokens
  truncated: boolean | null
  empty: boolean
  prompt: string
  reply: string
  reasoning: string | null
}

// One finding of a run's diagnosis: what was seen and what to do.
export type Finding = { rule: string; trials: number; seen: string; do: string }

export type Run = {
  summary: RunSummary
  trials: Trial[]
  factors: Factor[]
  correlation_limit: number | null
  // The least coverage and monotonicity admission required, from the run's last start; null without events.
  admission: { min_coverage: number; min_monotonicity: number } | null
  evaluated: Placed[]
  threshold: number | null
  never_evaluated: Refusal[]
  memory: MemoryVersion[]
  acceptance: AcceptanceReports
  diagnosis: Finding[]
  health: Health
  running: RunningRound | null
  now: string
}

// Acceptance's reports: the model with and without the accepted factors, on validation and the hold-out.
export type Stat = 'rank_ic' | 'ic' | 'monotonicity' | 'long_short' | 'top_k_return' | 'top_k_excess' | 'top_k_annualised'

export type Contribution = {
  factor: string
  mode: 'drop_one' | 'add_one'
  difference: Partial<Record<Stat, number | null>>
  agreeing: Partial<Record<Stat, number>>
}

export type Comparison = {
  name: string
  segment: string
  holdout: boolean
  factors: string[]
  selection: boolean | null
  seeds: number[]
  top_k: number | null
  train_window: string[]
  test_window: string[]
  baseline: Partial<Record<Stat, number | null>>
  with_accepted: Partial<Record<Stat, number | null>>
  difference: Partial<Record<Stat, number | null>>
  verdict: 'improves' | 'mixed' | 'does_not_improve' | null
  forced: boolean
  contribution: Contribution[] | null
  code_version: string
  written: string
}

export type HoldoutOpen = { opened_at: string; code_version: string; forced: boolean }

export type AcceptanceReports = {
  validation: Comparison | null
  holdout: Comparison | null
  reports: Comparison[]
  holdout_opens: HoldoutOpen[]
}
// An evaluated trial as the trials page's admission map places it, its statistics in its direction.
export type Placed = {
  round: number
  slot: number
  status: 'accepted' | 'rejected'
  factor_direction: 1 | -1
  ic_mean: number | null
  ic_t: number | null
  partial_ic_mean: number | null
  partial_ic_t: number | null
  largest_correlation: number | null
  threshold: number | null
  monotonicity: number | null
}

export type TrialRef = { round: number; slot: number }

// Trials one stage refused for the same reason before the Evaluator scored them.
export type Refusal = { stage: string; reason: string; rounds: number[]; trials: TrialRef[] }

async function get<T>(url: string): Promise<T> {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`${url}: ${response.status} ${await response.text()}`)
  return response.json()
}

export const runs = () => get<RunSummary[]>('/api/runs')
export const run = (name: string) => get<Run>(`/api/run?name=${encodeURIComponent(name)}`)
export const call = (name: string, number: number) =>
  get<LLMCall>(`/api/run/call?name=${encodeURIComponent(name)}&number=${number}`)
export const ic = (name: string, round: number, slot: number) =>
  get<TrialIC>(`/api/run/ic?name=${encodeURIComponent(name)}&round=${round}&slot=${slot}`)
export const files = (name: string) => get<RunFiles>(`/api/run/files?name=${encodeURIComponent(name)}`)
export const version = (name: string) => get<string>(`/api/run/version?name=${encodeURIComponent(name)}`)
