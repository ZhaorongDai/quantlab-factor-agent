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
  now: string
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
