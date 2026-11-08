// The round plan's groups by mechanism type, run with `npm test`.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { groupOf, groupSlots } from './slots.ts'

const slot = (n: number, mechanism_type: string) => ({ slot: n, mechanism_type, trial: `T4-0${n + 1}` })

test('the slots group by mechanism type, the types in plan order and each group in slot order', () => {
  const groups = groupSlots([slot(0, 'momentum'), slot(1, 'liquidity'), slot(2, 'momentum'), slot(3, 'value'), slot(4, 'liquidity')])
  assert.deepEqual(groups.map(g => g.type), ['momentum', 'liquidity', 'value'])
  assert.deepEqual(groups.map(g => g.slots.map(s => s.slot)), [[0, 2], [1, 4], [3]])
  assert.deepEqual(groupSlots([]), [])
})

test('the group holding a trial is its mechanism type, and none for a trial not in the plan', () => {
  const slots = [slot(0, 'momentum'), slot(1, 'liquidity'), slot(2, 'momentum')]
  assert.equal(groupOf(slots, s => s.trial === 'T4-02'), 'liquidity')
  assert.equal(groupOf(slots, s => s.trial === 'T4-03'), 'momentum')
  assert.equal(groupOf(slots, s => s.trial === 'T5-01'), null)
})
