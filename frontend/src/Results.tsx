// The results page: a health line and one card per accepted factor with what decides how far to trust it;
// a card opens to the factor written up. Every IC and t is in the factor's direction, the raw value on hover.
import { useState } from 'react'
import type { Factor, Neighbour, Run, RunSummary } from './api'
import Verdict, { ContributionLines, dropOne } from './Verdict'
import Diagnosis from './Diagnosis'

const SORTS = {
  contribution: 'drop-one contribution',
  round: 'round',
  t: 'partial-IC t',
  limit: 'distance from the correlation limit',
  coverage: 'coverage',
} as const
type Sort = keyof typeof SORTS

const fmt = (v: number | null | undefined, digits = 2) => (v == null ? '–' : v.toFixed(digits))
const signed = (v: number) => (v < 0 ? `−${Math.abs(v)}` : `+${v}`)
const percent = (v: number | null) => (v == null ? '–' : `${Math.round(v * 100)}%`)
const raw = (f: Factor, key: string) => {
  const v = f.metrics[key]
  return typeof v === 'number' ? `raw (as the ledger records it): ${v.toFixed(4)}` : 'no raw value'
}

// The factor's largest absolute correlation with an existing factor, reference, accepted or known.
function closest(f: Factor): number | null {
  const values = [f.nearest, f.nearest_known].map(n => n?.correlation).filter((v): v is number => v != null)
  return values.length ? Math.max(...values.map(Math.abs)) : null
}

function sortKey(run: Run, f: Factor, sort: Sort): number {
  // Larger first, except the round (earliest first).
  if (sort === 'contribution') return dropOne(run, f.label) ?? -Infinity
  if (sort === 'round') return -(f.round * 1000 + f.slot)
  if (sort === 't') return f.partial_ic_t ?? -Infinity
  if (sort === 'limit') return -(closest(f) ?? 0)
  return f.coverage ?? -Infinity
}

function Health({ summary }: { summary: RunSummary }) {
  const s = summary
  return (
    <p className="facts">
      {s.stop ? <>Stopped: {s.stop.reason}.</> : <>Not stopped.</>}{' '}
      {s.rounds} round{s.rounds === 1 ? '' : 's'} written, {s.trials} trial{s.trials === 1 ? '' : 's'},{' '}
      {s.accepted} accepted.
    </p>
  )
}

function Deciles({ values }: { values: (number | null)[] }) {
  const known = values.filter((v): v is number => v != null)
  if (!known.length) return null
  // Every decile's mean return is near the market's: each bar is its excess over the deciles' mean.
  const mean = known.reduce((a, b) => a + b, 0) / known.length
  const w = 240, h = 48, bw = w / values.length
  const m = Math.max(...known.map(v => Math.abs(v - mean))) || 1
  return (
    <svg className="deciles" width={w} height={h} role="img"
         aria-label="decile returns over their mean, decile 1 (lowest value) to decile 10 (highest)">
      <line x1={0} x2={w} y1={h / 2} y2={h / 2} className="axis" />
      {values.map((v, i) => {
        const x = v == null ? 0 : v - mean
        const bh = (Math.abs(x) / m) * (h / 2 - 2)
        return (
          <rect key={i} x={i * bw + 1} width={bw - 2} y={x >= 0 ? h / 2 - bh : h / 2} height={bh}
                className={x >= 0 ? 'pos' : 'neg'}>
            <title>{`decile ${i + 1}: ${v == null ? 'no value' : `${v.toExponential(2)} mean return, ${x.toExponential(2)} over the deciles' mean`}`}</title>
          </rect>
        )
      })}
    </svg>
  )
}

function NeighbourName({ n }: { n: Neighbour }) {
  return <span title={n.label}>{n.name}</span>
}

function Margin({ f }: { f: Factor }) {
  const t = f.partial_ic_t
  if (t == null) return null
  const scale = Math.max(t, f.threshold ?? 0) * 1.1 || 1
  return (
    <div className="margin" title={raw(f, 'partial_ic_t')}>
      <span>partial-IC t <b>{fmt(t)}</b></span>
      <div className="bar"><i style={{ width: `${(Math.max(t, 0) / scale) * 100}%` }} />
        {f.threshold != null && <em style={{ left: `${(f.threshold / scale) * 100}%` }} />}</div>
      <small>{f.threshold != null ? `${fmt(t - f.threshold)} over ${fmt(f.threshold)}` : 'threshold not quoted'}</small>
    </div>
  )
}

function Card({ run, f, limit, open, toggle }: { run: Run; f: Factor; limit: number | null; open: boolean; toggle: () => void }) {
  const rho = closest(f)
  return (
    <article className={`factor${open ? ' open' : ''}`}>
      <header>
        <span className="muted">round {f.round} · {f.mechanism_type ?? 'no mechanism type'}</span>
        <span className="direction" title={f.factor_direction > 0 ? 'a higher value went with a higher return' : 'a lower value went with a higher return'}>
          {signed(f.factor_direction)}
        </span>
      </header>
      <h3><button className="link" onClick={toggle} aria-expanded={open}>{f.description}</button></h3>
      <Deciles values={f.decile_returns} />
      <Margin f={f} />
      <dl className="trust">
        <dt>monotonicity</dt><dd>{fmt(f.monotonicity)}</dd>
        <dt>closest</dt>
        <dd>{f.nearest ? <><NeighbourName n={f.nearest} /> ρ {fmt(f.nearest.correlation)}</> : <span className="muted">none in the pool</span>}</dd>
        <dt>closest known</dt>
        <dd>{f.nearest_known ? <><NeighbourName n={f.nearest_known} /> ρ {fmt(f.nearest_known.correlation)}</> : <span className="muted">no known factors</span>}</dd>
        {limit != null && rho != null && <><dt>ρ limit</dt><dd>{fmt(limit - rho)} below {limit}</dd></>}
        <dt>coverage</dt><dd>{percent(f.coverage)}</dd>
      </dl>
      <ContributionLines run={run} label={f.label} />
      {open && <WriteUp f={f} limit={limit} />}
    </article>
  )
}

function Correlation({ n, known }: { n: Neighbour | null; known: boolean }) {
  if (!n) return <>{known ? 'The run has no known factors to compare it with.' : 'There was no reference or accepted factor to compare it with.'} </>
  const kind = known ? 'known factor' : 'reference or accepted factor'
  const accepted = n.accepted_round != null ? 'the factor accepted in ' : ''
  return (
    <>Its closest {kind} is {accepted}<span title={n.label}>{n.name}</span>,
      with a cross-sectional correlation of {fmt(n.correlation)}{known ? ' (sampled on some training dates)' : ''}. </>
  )
}

function WriteUp({ f, limit }: { f: Factor; limit: number | null }) {
  const rho = closest(f)
  return (
    <div className="write-up">
      <h4>Hypothesis</h4><p>{f.hypothesis}</p>
      <h4>Description</h4><p>{f.description}</p>
      <h4>Code</h4><pre className="code">{f.code}</pre>
      <h4>How far to trust it</h4>
      <p>
        It was accepted in round {f.round} as {f.mechanism_type ?? 'a factor of no recorded mechanism type'}. Its factor direction is{' '}
        {signed(f.factor_direction)}: {f.factor_direction > 0 ? 'a higher value went with a higher' : 'a lower value went with a higher'} training
        return, and every IC and t here is read in that direction.{' '}
        Its training IC is <span title={raw(f, 'ic_mean')}>{fmt(f.ic_mean, 4)}</span> with a t of <span title={raw(f, 'ic_t')}>{fmt(f.ic_t)}</span>;
        its partial IC, after the reference set and the factors accepted before it, is <span title={raw(f, 'partial_ic_mean')}>{fmt(f.partial_ic_mean, 4)}</span>{' '}
        with a Newey-West t of <span title={raw(f, 'partial_ic_t')}>{fmt(f.partial_ic_t)}</span>
        {f.threshold != null && f.partial_ic_t != null
          ? <>, {fmt(f.partial_ic_t - f.threshold)} over the threshold of {fmt(f.threshold)} it had to clear (as the round's refusals quote it). </>
          : <>; no refusal of its round or an earlier one quotes the threshold it had to clear. </>}
        Its decile returns rise with the decile, in its direction, with a monotonicity of {fmt(f.monotonicity)} (1 is a steady rise).{' '}
        <Correlation n={f.nearest} known={false} />
        <Correlation n={f.nearest_known} known />
        {rho != null && limit != null && <>Its largest correlation is {fmt(limit - rho)} below the limit of {limit}. </>}
        On an average training day it has a value for {percent(f.coverage)} of the members.
      </p>
    </div>
  )
}

export default function Results({ run }: { run: Run }) {
  // The cards rank by their drop-one contribution when Acceptance measured it.
  const ranked = run.factors.some(f => dropOne(run, f.label) != null)
  const [chosen, setSort] = useState<Sort | null>(null)
  const sort = chosen ?? (ranked ? 'contribution' : 'round')
  const sorts = (Object.keys(SORTS) as Sort[]).filter(s => s !== 'contribution' || ranked)
  const [open, setOpen] = useState<string | null>(null)
  const factors = [...run.factors].sort((a, b) => sortKey(run, b, sort) - sortKey(run, a, sort))
  return (
    <>
      <Health summary={run.summary} />
      <Verdict run={run} />
      <Diagnosis run={run} />
      {!factors.length ? (
        <p className="muted">
          {run.summary.stop ? 'The run accepted no factor.' : 'No factor accepted yet.'}
        </p>
      ) : (
        <>
          <div className="sorts">
            <h2>{factors.length} accepted factor{factors.length === 1 ? '' : 's'}</h2>
            <span className="muted">sort by</span>
            {sorts.map(s => (
              <button key={s} className={sort === s ? 'on' : ''} onClick={() => setSort(s)}>{SORTS[s]}</button>
            ))}
          </div>
          <p className="muted small">
            Each threshold is the one admission quoted in the round's refusals, carried over from the round before when
            a round quotes none. The correlation limit {run.correlation_limit != null
              ? <>({run.correlation_limit}) is the one a refusal quoted.</>
              : <>is unknown: no refusal of this run quoted it.</>}{' '}
            Hover an IC or t for its raw value.
          </p>
          <div className="factors">
            {factors.map(f => (
              <Card key={f.label} run={run} f={f} limit={run.correlation_limit} open={open === f.label}
                    toggle={() => setOpen(open === f.label ? null : f.label)} />
            ))}
          </div>
        </>
      )}
    </>
  )
}
