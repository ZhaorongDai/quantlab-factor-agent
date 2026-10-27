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

export type Run = { summary: RunSummary; trials: Trial[]; now: string }

async function get<T>(url: string): Promise<T> {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`${url}: ${response.status} ${await response.text()}`)
  return response.json()
}

export const runs = () => get<RunSummary[]>('/api/runs')
export const run = (name: string) => get<Run>(`/api/run?name=${encodeURIComponent(name)}`)
