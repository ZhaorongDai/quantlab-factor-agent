// The propagation view's axes and rank ladder, run with `npm test`.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { axisRow, ladder, pickLine } from './propagation.ts'

const node = (trial: string, own: number, neighbourhood: number, propagated: number, rank: number | null = null) => {
  const u = (v: number) => ({ quality: v, fertility: 0, novelty: 0 })
  return { trial, own: u(own), neighbourhood: u(neighbourhood), propagated: u(propagated), rank_own: rank, rank_propagated: rank }
}

const close = (found: number[], expected: number[]) =>
  assert.ok(found.length === expected.length && found.every((v, i) => Math.abs(v - expected[i]) < 1e-9), `${found} ≠ ${expected}`)

test('the result of a trial propagated by gamma sits the gamma fraction of the way from its own mark to its neighbours mark', () => {
  for (const gamma of [0.75, 0.4]) {
    const row = axisRow(node('T1-01', -1, 3, -1 + gamma * 4), [], 'quality', 200)
    assert.ok(Math.abs((row.result - row.own) / (row.neighbours - row.own) - gamma) < 1e-12)
  }
})

test('the result sits at the value after propagation', () => {
  const row = axisRow(node('T1-01', 0, 4, 1), [], 'quality', 104, 2)
  close([row.own, row.result, row.neighbours], [2, 27, 102])
})

test('an axis spans every mark and tick, from the first pixel to the last', () => {
  const others = [node('T1-02', 0, 0, 5), node('T1-03', 0, 0, -2)]
  const row = axisRow(node('T1-01', 1, 2, 1.5), others, 'quality', 110, 5)
  close([row.own, row.neighbours, row.result, ...row.ticks], [5 + 300 / 7, 5 + 400 / 7, 5 + 350 / 7, 105, 5])
})

test('an axis whose marks all share one value puts them in its middle', () => {
  const row = axisRow(node('T1-01', 2, 2, 2), [node('T1-02', 0, 0, 2)], 'quality', 100, 0)
  assert.deepEqual([row.own, row.neighbours, row.result, ...row.ticks], [50, 50, 50, 50])
})

test('the pick line falls under the last Picked parent after propagation', () => {
  const nodes = [node('T1-01', 0, 0, 0, 1), node('T1-02', 0, 0, 0, 4), node('T1-03', 0, 0, 0, 2)]
  assert.equal(pickLine(nodes, ['T1-01', 'T1-02']), 4)
  assert.equal(pickLine(nodes, []), 0)
})

test('the ladder shows every rank to the pick line and a few past it', () => {
  assert.equal(ladder(3, 1, 2, 15).rows, 15)
  assert.equal(ladder(3, 1, 20, 15).rows, 21)
})

test('a rank past the ladder shows as its number below it, a rank on it as its row', () => {
  const found = ladder(23, 4, 2, 15)
  assert.deepEqual(found.before, { row: null, below: 23 })
  assert.deepEqual(found.after, { row: 4, below: null })
})
