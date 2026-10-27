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
}

export type MemoryLine = { change: 'kept' | 'added' | 'removed'; line: string }

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
  monotonicity: number | null
  nearest: Neighbour | null
  nearest_known: Neighbour | null
  decile_returns: (number | null)[]
  coverage: number | null
  metrics: Record<string, unknown>
}

export type Run = {
  summary: RunSummary
  trials: Trial[]
  factors: Factor[]
  correlation_limit: number | null
  evaluated: Placed[]
  threshold: number | null
  never_evaluated: Refusal[]
  memory: MemoryVersion[]
  acceptance: AcceptanceReports
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
