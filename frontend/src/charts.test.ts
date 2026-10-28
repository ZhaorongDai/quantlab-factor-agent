// What a factor's charts and checks draw, run with `npm test`.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { checks, cumulative, excess, yearTicks } from './charts.ts'

test('the cumulative IC adds each day in the factor direction and skips an unscored day', () => {
  assert.deepEqual(cumulative([0.1, null, -0.05, 0.2], 1), [0.1, 0.1, 0.05, 0.25].map(v => v))
  assert.deepEqual(cumulative([0.1, null, -0.05], -1).map(v => +v.toFixed(10)), [-0.1, -0.1, -0.05])
})

test('a tick marks the first day of each year', () => {
  assert.deepEqual(yearTicks(['2016-12-30', '2017-01-03', '2017-06-01', '2018-01-02']),
    [{ index: 0, year: '2016' }, { index: 1, year: '2017' }, { index: 3, year: '2018' }])
})

test('decile bars are the returns over their mean, a missing decile left out of the mean', () => {
  assert.deepEqual(excess([1, 2, null, 3]), [-1, 0, null, 1])
})

test('each statistic passes or fails against the bar admission set, and is undecided when the bar is unknown', () => {
  const found = checks({
    partial_ic_t: 5.1, threshold: 3.97, largest_correlation: 0.75, limit: 0.7,
    coverage: 0.92, min_coverage: null, monotonicity: 0.4, min_monotonicity: 0.6,
  })
  assert.deepEqual(found.map(c => [c.label, c.passed]),
    [['partial-IC t', true], ['largest |ρ|', false], ['coverage', null], ['monotonicity', false]])
})
