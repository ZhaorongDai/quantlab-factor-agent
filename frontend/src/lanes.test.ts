// How the calls of one role are laid on the rows of its lane, run with `npm test`.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { rows } from './lanes.ts'

test('calls one after another share a row', () => {
  assert.deepEqual(rows([{ from: 0, to: 10, item: 'a' }, { from: 20, to: 30, item: 'b' }], 2), [['a', 'b']])
})

test('calls asked side by side go on rows of their own', () => {
  const spans = [{ from: 0, to: 10, item: 'a' }, { from: 1, to: 9, item: 'b' }, { from: 2, to: 4, item: 'c' }]
  assert.deepEqual(rows(spans, 2), [['a'], ['b'], ['c']])
})

test('a short call takes the least width, so the next one does not land on it', () => {
  assert.deepEqual(rows([{ from: 0, to: 0.1, item: 'a' }, { from: 1, to: 2, item: 'b' }], 3), [['a'], ['b']])
  assert.deepEqual(rows([{ from: 0, to: 0.1, item: 'a' }, { from: 3, to: 4, item: 'b' }], 3), [['a', 'b']])
})

test('a freed row is used again before a new one is opened', () => {
  const spans = [{ from: 0, to: 10, item: 'a' }, { from: 1, to: 3, item: 'b' }, { from: 5, to: 8, item: 'c' }]
  assert.deepEqual(rows(spans, 1), [['a'], ['b', 'c']])
})
