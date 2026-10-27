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

// One finding of a run's diagnosis: what was seen and what to do.
export type Finding = { rule: string; trials: number; seen: string; do: string }

export type Run = {
  summary: RunSummary
  trials: Trial[]
  factors: Factor[]
  correlation_limit: number | null
  memory: MemoryVersion[]
  diagnosis: Finding[]
  now: string
}

async function get<T>(url: string): Promise<T> {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`${url}: ${response.status} ${await response.text()}`)
  return response.json()
}

export const runs = () => get<RunSummary[]>('/api/runs')
export const run = (name: string) => get<Run>(`/api/run?name=${encodeURIComponent(name)}`)
