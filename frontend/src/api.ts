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

export type Run = {
  summary: RunSummary
  trials: Trial[]
  factors: Factor[]
  correlation_limit: number | null
  memory: MemoryVersion[]
  health: Health
  now: string
}

async function get<T>(url: string): Promise<T> {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`${url}: ${response.status} ${await response.text()}`)
  return response.json()
}

export const runs = () => get<RunSummary[]>('/api/runs')
export const run = (name: string) => get<Run>(`/api/run?name=${encodeURIComponent(name)}`)
export const version = (name: string) => get<string>(`/api/run/version?name=${encodeURIComponent(name)}`)
