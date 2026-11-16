// What a factor's charts and checks draw, run with `npm test`.

// The running sum of a daily IC in the factor's direction (``sign``); an unscored day (null) adds nothing.
export function cumulative(values: (number | null)[], sign: number): number[] {
  let sum = 0
  return values.map(v => (sum += v == null ? 0 : sign * v))
}

// The first day of each year among ``days`` (ISO dates, in order), as the chart's ticks. A tick's year is labelled
// only when it stands at least ``gap`` (a share of the axis) from the last labelled one; a short first year leaves
// its label to the next. Every tick keeps its line.
export function yearTicks(days: string[], gap = 0): { index: number; year: string; label: boolean }[] {
  const ticks: { index: number; year: string; label: boolean }[] = []
  days.forEach((day, index) => {
    const year = day.slice(0, 4)
    if (!ticks.length || ticks[ticks.length - 1].year !== year) ticks.push({ index, year, label: false })
  })
  const span = Math.max(1, days.length)
  let last = -Infinity
  ticks.forEach((t, k) => {
    const next = ticks[k + 1]
    const after = ticks[k + 2]
    // A first year shorter than the one after it, and too short for its label, leaves the label to the next year.
    if (k === 0 && next && (next.index - t.index) / span < gap && (!after || next.index - t.index < after.index - next.index)) return
    if ((t.index - last) / span >= gap) { t.label = true; last = t.index }
  })
  return ticks
}

// Each decile's mean return over the deciles' mean, so the bars show the spread rather than the market's level.
export function excess(deciles: (number | null)[]): (number | null)[] {
  const known = deciles.filter((v): v is number => v != null)
  const mean = known.length ? known.reduce((a, b) => a + b, 0) / known.length : 0
  return deciles.map(v => (v == null ? null : v - mean))
}

export type Check = { label: string; value: string; bar: string; passed: boolean | null }

const num = (v: number, digits = 2) => v.toFixed(digits)

// The statistics admission judges a factor by, each against its bar: passed, failed, or null when the bar is unknown.
// A bar is a number, null when the run did not record it, or 'off' when the run switched the check off.
export type Bar = number | null | 'off'

// The bar of an admission check: 'off' when the run's opening records the check with no threshold (a run
// without events records none), else the threshold the run recorded, if any.
export function bar(opened: Record<string, number | null> | null, key: string, recorded: number | null = opened?.[key] ?? null): Bar {
  return opened != null && opened[key] == null ? 'off' : recorded
}

function check(label: string, value: string, bar: Bar, shown: (b: number) => string, passes: (b: number) => boolean): Check {
  if (bar === 'off') return { label, value, bar: 'off', passed: null }
  return { label, value, bar: bar == null ? 'bar unknown' : shown(bar), passed: bar == null ? null : passes(bar) }
}

export function checks(s: {
  partial_ic_t: number | null; threshold: Bar; largest_correlation: number | null; limit: Bar
  coverage: number | null; min_coverage: Bar; monotonicity: number | null; min_monotonicity: Bar
}): Check[] {
  const out: Check[] = []
  const t = s.partial_ic_t, rho = s.largest_correlation, coverage = s.coverage, rising = s.monotonicity
  if (t != null) out.push(check('partial-IC t', num(t), s.threshold, b => `≥ ${num(b)}`, b => t >= b))
  if (rho != null) out.push(check('largest |ρ|', num(rho), s.limit, b => `≤ ${num(b)}`, b => rho <= b))
  if (coverage != null) {
    out.push(check('coverage', `${Math.round(coverage * 100)}%`, s.min_coverage, b => `≥ ${Math.round(b * 100)}%`, b => coverage >= b))
  }
  if (rising != null) out.push(check('monotonicity', num(rising), s.min_monotonicity, b => `≥ ${num(b)}`, b => rising >= b))
  return out
}

// What the admission map's vertical axis plots: the partial-IC t against its threshold, or, with the t check
// switched off, the IC in the factor's direction against min_ic (none when the run set no IC floor).
export type Axis = {
  label: string
  value: (p: { partial_ic_t: number | null; ic_mean: number | null }) => number | null
  threshold: number | null
  thresholdLabel: string
  floor: number
}

export function admissionAxis(tOff: boolean, threshold: number | null, minIc: number | null): Axis {
  if (tOff) {
    return { label: 'IC', value: p => p.ic_mean, threshold: minIc, thresholdLabel: `min IC ${minIc}`, floor: 0 }
  }
  return {
    label: 'partial-IC t', value: p => p.partial_ic_t, threshold,
    thresholdLabel: threshold == null ? '' : `t threshold ${num(threshold)}`, floor: -2,
  }
}

// Ticks from lo to hi on a step of 1, 2 or 5 times a power of ten, the smallest that leaves at most `most` steps.
export function niceTicks(lo: number, hi: number, most = 8): number[] {
  const rough = (hi - lo) / most
  const power = 10 ** Math.floor(Math.log10(rough))
  const step = [1, 2, 5, 10].map(m => m * power).find(s => s >= rough) ?? 10 * power
  const first = Math.ceil(lo / step), last = Math.floor(hi / step)
  return Array.from({ length: last - first + 1 }, (_, i) => Number(((first + i) * step).toPrecision(12)))
}
