// The selection view: the accepted factors in a table with every fact that decides how far to trust them, each with a
// tick box, and the Acceptance calls for the ticked ones to paste into Python. The page runs none of them.
import { useState } from 'react'
import type { Contribution, Factor, Neighbour, Run } from './api'
import { contributionOf } from './Verdict'
import { acceptanceCalls } from './calls'

const fmt = (v: number | null | undefined, digits = 2) => (v == null ? '–' : v.toFixed(digits))
const signed = (v: number | null | undefined) => (v == null ? '–' : v < 0 ? `−${(-v).toFixed(4)}` : `+${v.toFixed(4)}`)
const percent = (v: number | null) => (v == null ? '–' : `${Math.round(v * 100)}%`)
const sameSet = (a: string[], b: string[]) => a.length === b.length && a.every(x => b.includes(x))

function Closest({ n }: { n: Neighbour | null }) {
  if (!n) return <span className="muted">–</span>
  return <span title={n.label}>{n.name} <span className="muted">ρ {fmt(n.correlation)}</span></span>
}

function Added({ c, seeds }: { c: Contribution | undefined; seeds: number }) {
  if (!c) return <span className="muted">–</span>
  return (
    <span className={(c.difference.rank_ic ?? 0) >= 0 ? 'up' : 'down'} title="rank IC difference; seeds agreeing on its sign">
      {signed(c.difference.rank_ic)} <span className="muted">{c.agreeing.rank_ic ?? '–'}/{seeds}</span>
    </span>
  )
}

function Row({ run, f, ticked, tick, open, toggle }:
  { run: Run; f: Factor; ticked: boolean; tick: () => void; open: boolean; toggle: () => void }) {
  const seeds = run.acceptance.validation?.seeds.length ?? 0
  const t = f.partial_ic_t
  return (
    <>
      <tr className={ticked ? 'ticked' : ''}>
        <td><input type="checkbox" checked={ticked} onChange={tick} aria-label={`select ${f.description}`} /></td>
        <td>{f.round}</td>
        <td className="name"><button className="link" onClick={toggle} aria-expanded={open}>{f.description}</button></td>
        <td title={f.factor_direction > 0 ? 'a higher value went with a higher return' : 'a lower value went with a higher return'}>
          {f.factor_direction > 0 ? '+1' : '−1'}
        </td>
        <td>{fmt(t)}{f.threshold != null && t != null && <span className="muted"> ({fmt(t - f.threshold)} over {fmt(f.threshold)})</span>}</td>
        <td>{fmt(f.monotonicity)}</td>
        <td><Closest n={f.nearest} /></td>
        <td><Closest n={f.nearest_known} /></td>
        <td>{percent(f.coverage)}</td>
        <td><Added c={contributionOf(run, f.label, 'drop_one')} seeds={seeds} /></td>
        <td><Added c={contributionOf(run, f.label, 'add_one')} seeds={seeds} /></td>
      </tr>
      {open && (
        <tr className="opened">
          <td />
          <td colSpan={10}>
            <h4>Hypothesis</h4><p>{f.hypothesis}</p>
            <h4>Code</h4><pre className="code">{f.code}</pre>
          </td>
        </tr>
      )}
    </>
  )
}

function Holdout({ run }: { run: Run }) {
  const opens = run.acceptance.holdout_opens
  if (!opens.length) {
    return (
      <p className="holdout-state closed">Hold-out not opened. Open it once, on the selection you keep.</p>
    )
  }
  const last = opens[opens.length - 1]
  return (
    <p className="holdout-state opened">
      Hold-out opened {opens.length === 1 ? 'once' : `${opens.length} times`}
      {` (last ${last.opened_at.replace('T', ' ').slice(0, 16)}${opens.some(o => o.forced) ? ', with force' : ''})`}: no longer
      untouched for a new selection.
    </p>
  )
}

export default function Selection({ run }: { run: Run }) {
  // Factors are unticked by label, so a factor accepted while the page is open arrives ticked.
  const [unticked, setUnticked] = useState<Set<string>>(new Set())
  const [open, setOpen] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const factors = run.factors
  const selection = factors.filter(f => !unticked.has(f.label))
  const labels = selection.map(f => f.label)
  const calls = acceptanceCalls(selection, run.acceptance.holdout_opens.length > 0)
  const validated = run.acceptance.reports.find(r => !r.holdout && sameSet(r.factors, labels))
  const tick = (label: string) => {
    const next = new Set(unticked)
    if (!next.delete(label)) next.add(label)
    setUnticked(next)
    setCopied(false)
  }
  if (!factors.length) {
    return <p className="muted">{run.summary.stop ? 'No factor accepted.' : 'No factor accepted yet.'}</p>
  }
  const all = selection.length === factors.length
  return (
    <div className="selection">
      <div className="sorts">
        <h2>{selection.length} of {factors.length} accepted factor{factors.length === 1 ? '' : 's'} selected</h2>
        <button onClick={() => { setUnticked(all ? new Set(factors.map(f => f.label)) : new Set()); setCopied(false) }}>
          {all ? 'untick all' : 'tick all'}
        </button>
      </div>
      <p className="muted small">In the factor's direction · contribution: rank IC difference, seeds agreeing</p>
      <div className="table-scroll">
        <table className="pick">
          <thead>
            <tr>
              <th /><th>round</th><th>factor</th><th>dir.</th><th>partial-IC t</th><th>monotonicity</th>
              <th>closest</th><th>closest known</th><th>coverage</th><th>drop-one</th><th>add-one</th>
            </tr>
          </thead>
          <tbody>
            {factors.map(f => (
              <Row key={f.label} run={run} f={f} ticked={!unticked.has(f.label)} tick={() => tick(f.label)}
                   open={open === f.label} toggle={() => setOpen(open === f.label ? null : f.label)} />
            ))}
          </tbody>
        </table>
      </div>
      <h2>Acceptance calls</h2>
      <Holdout run={run} />
      {!selection.length ? <p className="muted">Tick at least one factor.</p> : (
        <>
          {validated && (
            <p className="small">Already validated ({validated.name}, {validated.written.replace('T', ' ').slice(0, 16)}).</p>
          )}
          <div className="calls">
            <button onClick={() => navigator.clipboard?.writeText(calls).then(() => setCopied(true))}>
              {copied ? 'copied' : 'copy'}
            </button>
            <pre className="code">{calls}</pre>
          </div>
          <p className="muted small">Not run by the page: paste into Python where the run is; rename the exports.</p>
        </>
      )}
    </div>
  )
}
