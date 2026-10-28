// How the run list groups runs into series, run with `npm test`.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import type { RunSummary } from './api.ts'
import { series } from './series.ts'

const run = (name: string, updated: string): RunSummary => ({
  name, updated, code_version: 'abc', rounds: 1, trials: 1, accepted: 0, stop: null, unreadable: null,
  state: 'unknown', max_rounds: null,
})

test('runs named with a version suffix are the versions of one series, the run without it its first', () => {
  const found = series([run('fund_v3', '2026-10-02'), run('fund', '2026-10-01'), run('fund_v4', '2026-10-03')])
  assert.deepEqual(found.map(s => s.name), ['fund'])
  assert.deepEqual(found[0].runs.map(r => [r.run.name, r.version]), [['fund_v4', 'v4'], ['fund_v3', 'v3'], ['fund', 'v1']])
})

test('versions sort by their number, not as text', () => {
  const found = series([run('hf_v2', '2026-10-01'), run('hf_v10', '2026-10-01')])
  assert.deepEqual(found[0].runs.map(r => r.version), ['v10', 'v2'])
})

test('series come latest active first, and a run without siblings is a series of its own', () => {
  const found = series([run('old', '2026-09-01'), run('hf_v2', '2026-10-05'), run('fund_v4', '2026-10-03')])
  assert.deepEqual(found.map(s => s.name), ['hf', 'fund', 'old'])
})

test('a run in a subfolder keeps its folder in its series', () => {
  const found = series([run('team/a_v2', '2026-10-01'), run('b_v2', '2026-10-01')])
  assert.deepEqual(found.map(s => s.name).sort(), ['b', 'team/a'])
})
