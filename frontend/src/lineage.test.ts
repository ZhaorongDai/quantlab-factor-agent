// The Lineage graph's layout and its nodes' kin, run with `npm test`.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { family, idOf, keyOf, layout, rings, steppedRounds } from './lineage.ts'

const node = (id: string, parent = '') => ({ id, round: Number(id.slice(1, id.indexOf('-'))), parent })

test('a trial id and the page key name the same trial', () => {
  assert.equal(idOf(3, 0), 'T3-01')
  assert.equal(idOf(12, 9), 'T12-10')
  assert.equal(keyOf('T12-10'), '12.9')
  assert.equal(keyOf('not a trial'), null)
})

test('each round is a column and its trials sit in slot order when none has a parent', () => {
  const at = layout([node('T1-01'), node('T1-02'), node('T2-01')], [1, 2])
  assert.deepEqual(Object.fromEntries(at), { 'T1-01': [0, 0], 'T1-02': [0, 1], 'T2-01': [1, 0] })
})

test('a child sits in the order of its parent, a new direction after the children', () => {
  const at = layout([
    node('T1-01'), node('T1-02'), node('T1-03'),
    node('T2-01'), node('T2-02', 'T1-03'), node('T2-03', 'T1-01'),
  ], [1, 2])
  assert.deepEqual([at.get('T2-03'), at.get('T2-02'), at.get('T2-01')], [[1, 0], [1, 1], [1, 2]])
})

test('a round with no trials keeps its column, and a trial of a round not shown is left out', () => {
  const at = layout([node('T1-01'), node('T3-01'), node('T4-01')], [1, 2, 3])
  assert.deepEqual(Object.fromEntries(at), { 'T1-01': [0, 0], 'T3-01': [2, 0] })
})

test('a trial\'s family is its ancestors and every trial built on it, down the generations', () => {
  const nodes = [node('T1-01'), node('T2-01', 'T1-01'), node('T3-01', 'T2-01'), node('T3-02', 'T1-01'), node('T3-03')]
  assert.deepEqual([...family(nodes, 'T2-01')].sort(), ['T1-01', 'T2-01', 'T3-01'])
  assert.deepEqual([...family(nodes, 'T1-01')].sort(), ['T1-01', 'T2-01', 'T3-01', 'T3-02'])
  assert.deepEqual([...family(nodes, null)], [])
})

test('the stepper steps through the written rounds and the round in progress once it is planned', () => {
  assert.deepEqual(steppedRounds(3, null), [1, 2, 3])
  assert.deepEqual(steppedRounds(3, { round: 4, planned: false }), [1, 2, 3])
  assert.deepEqual(steppedRounds(3, { round: 4, planned: true }), [1, 2, 3, 4])
  assert.deepEqual(steppedRounds(0, null), [])
})

test('a Picked parent has one ring per round it was picked in, the stepped round marked', () => {
  const found = rings([2, 4, 5, 7, 9], 5)
  assert.deepEqual(found.map(r => r.round), [2, 4, 5, 7, 9])
  assert.deepEqual(found.map(r => r.now), [false, false, true, false, false])
})

test('the rings are concentric from the inner radius, as far apart as fits inside the outer one', () => {
  assert.deepEqual(rings([1, 2, 3], 1).map(r => r.r), [8, 10.5, 13])
  assert.deepEqual(rings([1, 2, 3, 4, 5], 1).map(r => r.r), [8, 9.5, 11, 12.5, 14])
  assert.deepEqual(rings([3], 1).map(r => r.r), [8])
})
