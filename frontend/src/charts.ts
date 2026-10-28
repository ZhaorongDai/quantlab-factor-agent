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
export function checks(s: {
  partial_ic_t: number | null; threshold: number | null; largest_correlation: number | null; limit: number | null
  coverage: number | null; min_coverage: number | null; monotonicity: number | null; min_monotonicity: number | null
}): Check[] {
  const out: Check[] = []
  if (s.partial_ic_t != null) {
    out.push({ label: 'partial-IC t', value: num(s.partial_ic_t), bar: s.threshold == null ? 'bar unknown' : `≥ ${num(s.threshold)}`,
      passed: s.threshold == null ? null : s.partial_ic_t >= s.threshold })
  }
  if (s.largest_correlation != null) {
    out.push({ label: 'largest |ρ|', value: num(s.largest_correlation), bar: s.limit == null ? 'bar unknown' : `≤ ${num(s.limit)}`,
      passed: s.limit == null ? null : s.largest_correlation <= s.limit })
  }
  if (s.coverage != null) {
    out.push({ label: 'coverage', value: `${Math.round(s.coverage * 100)}%`,
      bar: s.min_coverage == null ? 'bar unknown' : `≥ ${Math.round(s.min_coverage * 100)}%`,
      passed: s.min_coverage == null ? null : s.coverage >= s.min_coverage })
  }
  if (s.monotonicity != null) {
    out.push({ label: 'monotonicity', value: num(s.monotonicity), bar: s.min_monotonicity == null ? 'bar unknown' : `≥ ${num(s.min_monotonicity)}`,
      passed: s.min_monotonicity == null ? null : s.monotonicity >= s.min_monotonicity })
  }
  return out
}
