// The trials page: a numbers bar, the admission map of every evaluated trial (largest correlation with an existing
// factor against the partial-IC t in the trial's direction), the trials that never reached the Evaluator grouped by
// stage and reason, and one side panel reading the picked trial, LLM call or pending trial. While the campaign runs,
// the round in progress sits between the bar and the map. Trials that arrive while the page is open stand out for a
// while.
import { useEffect, useRef, useState } from 'react'
import type { Placed, Refusal, Run, Trial } from './api'
import { checks } from './charts'
import { Checks, CumulativeIC, DecileChart } from './FactorCharts'
import { CallReader, PendingReader, RoundStrip, type Picked } from './Round'

const W = 720, H = 440, P = 44
// How long a trial that arrived while the page was open stands out.
const FRESH_MS = 15000
const STAGES: Record<string, string> = {
  parse: 'Parse: the reply could not be read as candidates',
  gate: 'Gate: the code was refused before evaluation',
  review: 'Review: the reviewer still found it inconsistent after a rewrite',
  evaluate: 'Evaluation: the Evaluator could not build the code',
}

const key = (t: { round: number; slot: number }) => `${t.round}.${t.slot}`
const fmt = (v: number | null | undefined, digits = 2) => (v == null ? '–' : v.toFixed(digits))
const describe = (t: Trial) => String(t.candidate.description ?? t.candidate.raw ?? `slot ${t.slot + 1}`)

// The keys of the trials that arrived since the page was opened, for FRESH_MS after each arrives.
function useFresh(trials: Trial[]): Set<string> {
  const seen = useRef<Set<string> | null>(null)
  const [fresh, setFresh] = useState<Set<string>>(new Set())
  useEffect(() => {
    if (!seen.current) { seen.current = new Set(trials.map(key)); return }
    const added = trials.map(key).filter(k => !seen.current!.has(k))
    if (!added.length) return
    added.forEach(k => seen.current!.add(k))
    setFresh(before => new Set([...before, ...added]))
    const timer = setTimeout(() => setFresh(before => new Set([...before].filter(k => !added.includes(k)))), FRESH_MS)
    return () => clearTimeout(timer)
  }, [trials])
  return fresh
}

function Numbers({ run }: { run: Run }) {
  const s = run.summary
  const failed = run.trials.filter(t => t.status === 'failed').length
  return (
    <section className="numbers">
      <span><b>{s.rounds}</b> round{s.rounds === 1 ? '' : 's'}</span>
      <span><b>{s.trials}</b> trials</span>
      <span><b>{run.evaluated.length}</b> evaluated</span>
      <span className="good"><b>{s.accepted}</b> accepted</span>
      <span className="muted"><b>{failed}</b> failed</span>    </section>
  )
}

function AdmissionMap({ run, picked, fresh, pick }: { run: Run; picked: string | null; fresh: Set<string>; pick: (k: string) => void }) {
  const points = run.evaluated.filter(p => p.partial_ic_t != null)
  const limit = run.correlation_limit, threshold = run.threshold
  const ts = points.map(p => p.partial_ic_t!)
  const low = Math.min(-2, ...ts), high = Math.max((threshold ?? 0) + 1, ...ts)
  const y0 = low - (high - low) * 0.04, y1 = high + (high - low) * 0.04
  const step = [1, 2, 5, 10, 20, 50, 100].find(s => (y1 - y0) / s <= 8) ?? 200
  const ticks = Array.from({ length: Math.floor(y1 / step) - Math.ceil(y0 / step) + 1 }, (_, i) => (Math.ceil(y0 / step) + i) * step)
  const x1 = Math.max(1, ...points.map(p => p.largest_correlation ?? 0))
  const X = (v: number) => P + (v / x1) * (W - 2 * P)
  const Y = (v: number) => H - P - ((v - y0) / (y1 - y0)) * (H - 2 * P)
  const failed = run.trials.filter(t => t.status === 'failed').length
  const size = (p: Placed) => (picked === key(p) || fresh.has(key(p)) ? 7 : p.status === 'accepted' ? 5 : 3.5)
  return (
    <figure className="admission">
      <svg viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label="every evaluated trial by its largest correlation with an existing factor and its partial-IC t in its direction">
        {threshold != null && (
          <rect x={X(0)} y={Y(y1)} width={X(limit ?? x1) - X(0)} height={Y(threshold) - Y(y1)} className="zone">
            <title>Admission accepts a trial in this region, when its coverage and decile monotonicity also pass.</title>
          </rect>
        )}
        <line x1={P} x2={W - P} y1={Y(0)} y2={Y(0)} className="axis" />
        <line x1={P} x2={P} y1={P} y2={H - P} className="axis" />
        {limit != null && <>
          <line x1={X(limit)} x2={X(limit)} y1={P} y2={H - P} className="limit" />
          <text x={X(limit) + 4} y={P + 12} className="label">correlation limit {limit}</text>
        </>}
        {threshold != null && <>
          <line x1={P} x2={W - P} y1={Y(threshold)} y2={Y(threshold)} className="limit" />
          <text x={W - P - 4} y={Y(threshold) - 4} textAnchor="end" className="label">final t threshold {fmt(threshold)}</text>
        </>}
        {[0, 0.25, 0.5, 0.75, 1].filter(v => v <= x1).map(v => (
          <text key={v} x={X(v)} y={H - P + 14} textAnchor="middle" className="label">{v}</text>
        ))}
        {ticks.map(v => (
          <text key={v} x={P - 4} y={Y(v) + 4} textAnchor="end" className="label">{v}</text>
        ))}
        <text x={W / 2} y={H - 8} textAnchor="middle" className="label">largest |correlation| with a reference, accepted or known factor</text>
        <text x={12} y={H / 2} transform={`rotate(-90 12 ${H / 2})`} textAnchor="middle" className="label">partial-IC t, in the trial's direction</text>
        {points.map(p => (
          <circle key={key(p)} cx={X(p.largest_correlation ?? 0)} cy={Y(p.partial_ic_t!)} r={size(p)}
                  className={`point ${p.status}${fresh.has(key(p)) ? ' fresh' : ''}${picked === key(p) ? ' picked' : ''}`}
                  onClick={() => pick(key(p))}>
            <title>{`round ${p.round}, slot ${p.slot + 1}: ${p.status}, t ${fmt(p.partial_ic_t)}, |ρ| ${fmt(p.largest_correlation)}`}</title>
          </circle>
        ))}
      </svg>
      <figcaption className="muted small">
        <span className="key accepted">●</span> accepted <span className="key rejected">●</span> rejected by admission{' '}
        <span className="key failed">●</span> failed: {failed} trial{failed === 1 ? '' : 's'}, never scored, listed below
        with the trials the reviewer rejected.{' '}
        {threshold == null
          ? 'The t threshold is unknown yet, so it is not drawn. '
          : 'Each round had its own threshold, rising with the trials evaluated: an early accepted trial can sit below the final line. '}
        {limit == null && 'The correlation limit is unknown, so it is not drawn. '}
        A trial with nothing to compare with sits at 0.
      </figcaption>
    </figure>
  )
}

function NeverEvaluated({ groups, trials, picked, pick }: { groups: Refusal[]; trials: Map<string, Trial>; picked: string | null; pick: (k: string) => void }) {
  const [open, setOpen] = useState<Set<string>>(new Set())
  if (!groups.length) return null
  const total = groups.reduce((n, g) => n + g.trials.length, 0)
  const stages = [...new Set(groups.map(g => g.stage))]
  const toggle = (k: string) => setOpen(before => {
    const after = new Set(before)
    if (after.has(k)) after.delete(k); else after.add(k)
    return after
  })
  return (
    <section className="never">
      <h3>Never evaluated <span className="muted">· {total} trial{total === 1 ? '' : 's'}</span></h3>
      {stages.map(stage => (
        <div key={stage}>
          <h4>{STAGES[stage] ?? stage}</h4>
          {groups.filter(g => g.stage === stage).map(g => {
            const id = key(g.trials[0])
            const many = g.trials.length > 1
            const shown = !many || open.has(id)
            return (
              <div key={id} className="refusal">
                <button className={`link row${g.trials.some(t => key(t) === picked) ? ' picked' : ''}`}
                        onClick={() => (many ? toggle(id) : pick(id))} aria-expanded={many ? shown : undefined}>
                  <span className={`key ${stage === 'review' ? 'rejected' : 'failed'}`}>●</span>{' '}
                  {many && <b>×{g.trials.length} </b>}
                  <span className="muted">round{g.rounds.length > 1 ? 's' : ''} {g.rounds.join(', ')}</span>{' '}
                  {g.reason.length > 160 ? `${g.reason.slice(0, 160)}…` : g.reason}
                </button>
                {many && shown && g.trials.map(t => (
                  <button key={key(t)} className={`link row sub${key(t) === picked ? ' picked' : ''}`} onClick={() => pick(key(t))}>
                    <span className="muted">round {t.round}</span> {trials.has(key(t)) ? describe(trials.get(key(t))!) : ''}
                  </button>
                ))}
              </div>
            )
          })}
        </div>
      ))}
    </section>
  )
}

type Attempt = { item?: unknown; stage?: string; reasons?: string[]; hints?: string[] }

// The versions refused before the trial's own: a trial that failed at parse or the Gate ends on its own refusal.
function refusedBefore(t: Trial): Attempt[] {
  const attempts = t.attempts as Attempt[]
  return t.status === 'failed' && (t.stage === 'parse' || t.stage === 'gate') ? attempts.slice(0, -1) : attempts
}

function Version({ a, n }: { a: Attempt; n: number }) {
  const item = a.item as Record<string, unknown> | string | undefined
  const code = item && typeof item === 'object' ? String(item.code ?? '') : ''
  return (
    <li>
      <div className="muted small">version {n}, refused at {a.stage}</div>
      {(a.reasons ?? []).map((r, i) => <p key={i} className="reason">{r}</p>)}
      {(a.hints ?? []).map((h, i) => <p key={i} className="hint">Hint: {h}</p>)}
      {code ? <pre className="code">{code}</pre> : item != null && <pre className="code">{typeof item === 'string' ? item : JSON.stringify(item, null, 2)}</pre>}
    </li>
  )
}

function Statistics({ p, t, run, name }: { p: Placed; t: Trial; run: Run; name: string }) {
  const m = t.metrics ?? {}
  const items = checks({
    partial_ic_t: p.partial_ic_t, threshold: p.threshold, largest_correlation: p.largest_correlation, limit: run.correlation_limit,
    coverage: typeof m.coverage === 'number' ? m.coverage : null, min_coverage: run.admission?.min_coverage ?? null,
    monotonicity: p.monotonicity, min_monotonicity: run.admission?.min_monotonicity ?? null,
  })
  const raw = (k: string) => (typeof m[k] === 'number' ? `raw (as the ledger records it): ${(m[k] as number).toFixed(4)}` : 'no raw value')
  const nearest = (label: unknown, rho: unknown) => (label ? <>{String(label)} ρ {fmt(rho as number | null)}</> : <span className="muted">none</span>)
  return (
    <>
    <Checks items={items} />
    <DecileChart deciles={(m.decile_returns as (number | null)[]) ?? []} direction={p.factor_direction} monotonicity={p.monotonicity} />
    <CumulativeIC name={name} round={t.round} slot={t.slot} direction={p.factor_direction} />
    <dl className="trust">
      <dt>direction</dt>
      <dd>{p.factor_direction > 0 ? '+1: a higher value went with a higher return' : '−1: a lower value went with a higher return'}</dd>
      <dt>partial-IC t</dt>
      <dd title={raw('partial_ic_t')}>{fmt(p.partial_ic_t)}{p.threshold != null && <> against its round's threshold {fmt(p.threshold)}</>}</dd>
      <dt>partial IC</dt><dd title={raw('partial_ic_mean')}>{fmt(p.partial_ic_mean, 4)}</dd>
      <dt>IC, t</dt><dd><span title={raw('ic_mean')}>{fmt(p.ic_mean, 4)}</span>, <span title={raw('ic_t')}>{fmt(p.ic_t)}</span></dd>
      <dt>closest</dt><dd>{nearest(m.nearest, m.nearest_correlation)}</dd>
      <dt>closest known</dt><dd>{nearest(m.nearest_known, m.nearest_known_correlation)}</dd>
      <dt>coverage</dt><dd>{typeof m.coverage === 'number' ? `${Math.round(m.coverage * 100)}%` : '–'}</dd>
    </dl>
    </>
  )
}

function Reader({ t, placed, run, name }: { t: Trial; placed: Placed | undefined; run: Run; name: string }) {
  const d = t.direction ?? {}
  const before = refusedBefore(t)
  return (
    <>
      <div className="muted small">
        round {t.round}, slot {t.slot + 1} · {d.mechanism_type ?? 'no mechanism type'} · {d.action ?? ''}{d.parent ? ` of ${d.parent}` : ''}
      </div>
      <h3>{describe(t)}</h3>
      <span className={`badge ${t.status}`}>{t.status} at {t.stage}</span>
      {t.reasons.map((r, i) => <p key={i} className="reason">{r}</p>)}
      {placed && <><h4>Statistics, in its direction</h4><Statistics p={placed} t={t} run={run} name={name} /></>}
      <h4>Research direction</h4><p>{d.direction ?? <span className="muted">none recorded</span>}</p>
      <h4>Hypothesis</h4><p>{String(t.candidate.hypothesis ?? '–')}</p>
      <h4>Description</h4><p>{String(t.candidate.description ?? '–')}</p>
      <h4>Code</h4>
      <pre className="code">{t.candidate.code != null ? String(t.candidate.code) : String(t.candidate.raw ?? '–')}</pre>
      {before.length > 0 && <>
        <h4>Refused versions before this one</h4>
        <ol className="versions">{before.map((a, i) => <Version key={i} a={a} n={i + 1} />)}</ol>
      </>}
    </>
  )
}

export default function Trials({ run, name, clock }: { run: Run; name: string; clock: number }) {
  const [picked, setPicked] = useState<Picked>(null)
  const fresh = useFresh(run.trials)
  const trials = new Map(run.trials.map(t => [key(t), t]))
  const placed = new Map(run.evaluated.map(p => [key(p), p]))
  const pickedTrial = picked && 'trial' in picked ? picked.trial : null
  const pick = (k: string) => setPicked({ trial: k })
  const open = pickedTrial ? trials.get(pickedTrial) : undefined
  let reader
  if (picked && 'call' in picked) reader = <CallReader name={name} number={picked.call} pick={setPicked} />
  else if (picked && 'slot' in picked) {
    const from = picked.from
    reader = <PendingReader p={run.running?.pending.find(p => p.slot === picked.slot)}
                            back={from != null ? { call: from, go: () => setPicked({ call: from }) } : undefined} />
  }
  else if (open) reader = <Reader t={open} placed={placed.get(key(open))} run={run} name={name} />
  else reader = (
    <p className="muted">
      Click a point on the map, or a trial under “Never evaluated”, to read it: its research direction,
      hypothesis, description, code, the versions refused before it and its statistics.
      {run.running && ' Click a call of the round in progress, or a pending trial, to read it here too.'}
    </p>
  )
  return (
    <>
      <Numbers run={run} />
      {run.running && <RoundStrip running={run.running} clock={clock} crashed={run.health.crashed}
                                   picked={picked} pick={setPicked} />}
      <div className="trials-page">
        <div>
          {run.evaluated.length
            ? <AdmissionMap run={run} picked={pickedTrial} fresh={fresh} pick={pick} />
            : <p className="muted">No trial has been evaluated yet.</p>}
          <NeverEvaluated groups={run.never_evaluated} trials={trials} picked={pickedTrial} pick={pick} />
        </div>
        <aside className="reader">{reader}</aside>
      </div>
    </>
  )
}
