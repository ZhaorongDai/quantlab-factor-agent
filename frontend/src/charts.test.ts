// What a factor's charts and checks draw, run with `npm test`.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { admissionAxis, checks, cumulative, excess, niceTicks, yearTicks } from './charts.ts'

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

test('a check the run switched off shows as off and neither passes nor fails', () => {
  const found = checks({
    partial_ic_t: 1.2, threshold: 'off', largest_correlation: 0.95, limit: 'off',
    coverage: 0.2, min_coverage: 'off', monotonicity: -0.3, min_monotonicity: 'off',
  })
  assert.deepEqual(found.map(c => [c.label, c.bar, c.passed]), [
    ['partial-IC t', 'off', null], ['largest |ρ|', 'off', null], ['coverage', 'off', null], ['monotonicity', 'off', null],
  ])
})

test('the admission map plots the partial-IC t against its threshold, or the IC against min_ic with the t check off', () => {
  const p = { partial_ic_t: 5.2, ic_mean: 0.012 }
  const t = admissionAxis(false, 3.8, 0.004)
  assert.deepEqual([t.label, t.value(p), t.threshold, t.thresholdLabel], ['partial-IC t', 5.2, 3.8, 't threshold 3.80'])
  const ic = admissionAxis(true, null, 0.004)
  assert.deepEqual([ic.label, ic.value(p), ic.threshold, ic.thresholdLabel], ['IC in its direction', 0.012, 0.004, 'min IC 0.004'])
  assert.equal(admissionAxis(true, null, null).threshold, null)
})

test('ticks fall on round steps at most eight apart, small ones as well as large', () => {
  assert.deepEqual(niceTicks(-2.5, 20), [0, 5, 10, 15, 20])
  assert.deepEqual(niceTicks(-0.002, 0.031), [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03])
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
