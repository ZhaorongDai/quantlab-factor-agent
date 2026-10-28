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
    [{ index: 0, year: '2016', label: true }, { index: 1, year: '2017', label: true }, { index: 3, year: '2018', label: true }])
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

test('a year label too near the next one is left out, the short first year first, its line kept', () => {
  const days = ['2016-10-11', '2016-12-30', ...Array.from({ length: 5 }, (_, k) => ['01-03', '04-01', '07-01', '10-01'].map(d => `${2017 + k}-${d}`)).flat()]
  const ticks = yearTicks(days, 0.15)
  assert.deepEqual(ticks.map(t => [t.year, t.label]),
    [['2016', false], ['2017', true], ['2018', true], ['2019', true], ['2020', true], ['2021', true]])
})

test('when years crowd the axis every other label is kept', () => {
  const days = Array.from({ length: 10 }, (_, k) => `${2000 + k}-01-03`)
  assert.deepEqual(yearTicks(days, 0.15).filter(t => t.label).map(t => t.year), ['2000', '2002', '2004', '2006', '2008'])
})
