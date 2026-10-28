// How a run is followed live and paused: when the page reads the run again and where its clock stands, run with `npm test`.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { clock, follow, pausable, pause, polled, read, resume } from './poll.ts'

const SERVER = Date.parse('2026-10-27T12:00:00Z')

test('between two reads the clock is the server time of the last read moved on by the browser time since', () => {
  const live = read(follow(), 'v1', SERVER, 1_000)
  assert.equal(clock(live, 1_000), SERVER)
  assert.equal(clock(live, 8_000), SERVER + 7_000)
})

test('paused, the clock stands at the moment of the pause', () => {
  const live = pause(read(follow(), 'v1', SERVER, 1_000), 4_000)
  assert.equal(clock(live, 4_000), SERVER + 3_000)
  assert.equal(clock(live, 600_000), SERVER + 3_000)
})

test('live, a poll reads the run again when its version changed, or when a minute passed and the run has not stopped', () => {
  const live = read(follow(), 'v1', SERVER, 1_000)
  assert.equal(polled(live, 'v2', 6_000, false), 'read')
  assert.equal(polled(live, 'v1', 6_000, false), 'same')
  assert.equal(polled(live, 'v1', 62_000, false), 'read')
  assert.equal(polled(live, 'v1', 62_000, true), 'same')
})

test('paused, a poll never reads the run again: it only says whether the run changed since', () => {
  const paused = pause(read(follow(), 'v1', SERVER, 1_000), 4_000)
  assert.equal(polled(paused, 'v1', 9_000, false), 'same')
  assert.equal(polled(paused, 'v1', 300_000, false), 'same')
  assert.equal(polled(paused, 'v2', 9_000, false), 'changed')
})

test('resumed, the run is read again at once, even unchanged, and the clock moves on from that read', () => {
  const resumed = resume(pause(read(follow(), 'v1', SERVER, 1_000), 4_000))
  assert.equal(polled(resumed, 'v1', 9_000, true), 'read')
  const again = read(resumed, 'v1', SERVER + 8_000, 9_000)
  assert.equal(clock(again, 11_000), SERVER + 10_000)
})

test('pausing is offered only while the run can still change', () => {
  assert.equal(pausable(false, false), true)
  assert.equal(pausable(true, false), false)
  assert.equal(pausable(false, true), false)
})
