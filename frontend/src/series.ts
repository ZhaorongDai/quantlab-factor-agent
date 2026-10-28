// How the run list groups runs into series, run with `npm test`.
import type { RunSummary } from './api'

export type Series = { name: string; runs: { run: RunSummary; version: string }[] }

// A run named <series>_v<N> is version N of <series>; a run without the suffix is that series' first version, v1.
// Series come latest active first, and their runs from the latest version to the first.
export function series(runs: RunSummary[]): Series[] {
  const groups = new Map<string, Series['runs']>()
  for (const run of runs) {
    const m = run.name.match(/^(.*)_v(\d+)$/)
    const name = m ? m[1] : run.name
    groups.set(name, [...(groups.get(name) ?? []), { run, version: `v${m ? m[2] : 1}` }])
  }
  const latest = (rs: Series['runs']) => Math.max(...rs.map(r => Date.parse(r.run.updated)))
  return [...groups]
    .map(([name, rs]) => ({ name, runs: rs.sort((a, b) => Number(b.version.slice(1)) - Number(a.version.slice(1))) }))
    .sort((a, b) => latest(b.runs) - latest(a.runs))
}
