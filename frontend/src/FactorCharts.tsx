// A factor's statistics against admission's bars, its decile returns and its cumulative IC, for a trial's side panel
// and an accepted factor's write-up alike.
import { useEffect, useState } from 'react'
import * as api from './api'
import type { TrialIC } from './api'
import { cumulative, excess, yearTicks } from './charts'
import type { Check } from './charts'

export function Checks({ items }: { items: Check[] }) {
  if (!items.length) return null
  return (
    <div className="checks">
      {items.map(c => (
        <span key={c.label} className={`check ${c.passed == null ? 'unknown' : c.passed ? 'pass' : 'fail'}`} title={c.bar}>
          {c.passed == null ? '?' : c.passed ? '✓' : '✗'} {c.label} <b>{c.value}</b> <span className="bar">{c.bar}</span>
        </span>
      ))}
    </div>
  )
}

// The mean return of each tenth of the daily cross section, lowest values first, over the deciles' mean; in the
// factor's direction the bars should rise from left to right.
export function DecileChart({ deciles, direction, monotonicity }: { deciles: (number | null)[]; direction: number; monotonicity: number | null }) {
  const bars = excess(deciles)
  const known = bars.filter((v): v is number => v != null)
  if (!known.length) return null
  const top = Math.max(...known.map(Math.abs)) || 1
  const w = 300, h = 120, mid = h / 2 - 8, gap = 4, bw = (w - gap * (bars.length - 1)) / bars.length
  return (
    <figure className="chart">
      <figcaption>
        Deciles{monotonicity != null && <> · monotonicity <b>{monotonicity.toFixed(2)}</b></>}
        <span className="muted small"> · direction {direction > 0 ? '+1' : '−1'}</span>
      </figcaption>
      <svg viewBox={`0 0 ${w} ${h}`} width="100%" role="img" aria-label="decile returns over their mean">
        <line x1={0} x2={w} y1={mid} y2={mid} className="axis" />
        {bars.map((v, i) => {
          if (v == null) return null
          const len = (Math.abs(v) / top) * (mid - 4)
          return (
            <g key={i}>
              <rect x={i * (bw + gap)} y={v >= 0 ? mid - len : mid} width={bw} height={Math.max(len, 1)} className={v >= 0 ? 'up' : 'down'}>
                <title>{`decile ${i + 1}: ${(deciles[i] as number).toExponential(2)} mean return, ${v.toExponential(2)} over the deciles' mean`}</title>
              </rect>
              <text x={i * (bw + gap) + bw / 2} y={h - 2} className="tick">{i + 1}</text>
            </g>
          )
        })}
      </svg>
    </figure>
  )
}

function Line({ values, x, y, className }: { values: number[]; x: (i: number) => number; y: (v: number) => number; className: string }) {
  const step = Math.max(1, Math.floor(values.length / 600))
  const points = values.filter((_, i) => i % step === 0 || i === values.length - 1)
    .map((v, k) => `${x(Math.min(k * step, values.length - 1)).toFixed(1)},${y(v).toFixed(1)}`)
  return <polyline points={points.join(' ')} className={className} />
}

// The running sum of the daily rank IC and partial IC in the factor's direction over the training days, read from
// the run when the chart is shown; a run that kept no daily IC says so.
export function CumulativeIC({ name, round, slot, direction }: { name: string; round: number; slot: number; direction: number }) {
  const [series, setSeries] = useState<TrialIC | null | undefined>(undefined)
  useEffect(() => {
    let live = true
    setSeries(undefined)
    api.ic(name, round, slot).then(s => live && setSeries(s), () => live && setSeries(null))
    return () => { live = false }
  }, [name, round, slot])
  if (series === undefined) return <p className="muted small">Loading cumulative IC…</p>
  if (series === null) {
    return <p className="muted small">No cumulative IC (not kept by this run).</p>
  }
  const ic = cumulative(series.ic, direction), partial = cumulative(series.partial_ic, direction)
  const all = [...ic, ...partial, 0]
  const lo = Math.min(...all), hi = Math.max(...all), span = hi - lo || 1
  const w = 300, h = 140, pad = 14
  const x = (i: number) => (i / Math.max(1, ic.length - 1)) * w
  const y = (v: number) => pad + (1 - (v - lo) / span) * (h - 2 * pad)
  // A year's label is some 30 of the axis's 300 units wide.
  const ticks = yearTicks(series.days, 0.1)
  return (
    <figure className="chart">
      <figcaption>
        Cumulative IC · <span className="key ic">rank IC</span> <b>{ic[ic.length - 1].toFixed(2)}</b>{' '}
        <span className="key partial">partial IC</span> <b>{partial[partial.length - 1].toFixed(2)}</b>
      </figcaption>
      <svg viewBox={`0 0 ${w} ${h + 12}`} width="100%" role="img" aria-label="cumulative rank IC and partial IC">
        <line x1={0} x2={w} y1={y(0)} y2={y(0)} className="axis" />
        {ticks.map(t => (
          <g key={t.year}>
            <line x1={x(t.index)} x2={x(t.index)} y1={pad / 2} y2={h - pad / 2} className="grid" />
            {t.label && <text x={x(t.index) + 2} y={h + 10} className="tick start">{t.year}</text>}
          </g>
        ))}
        <Line values={partial} x={x} y={y} className="partial" />
        <Line values={ic} x={x} y={y} className="ic" />
      </svg>
    </figure>
  )
}
