// The run's Lineage on the trials page: a round stepper with the stepped round's plan, the Lineage graph (every trial
// in a column of its round, its action as the node's shape, its outcome as the colour, an arrow from each parent, a
// ring per round a trial was a Picked parent) and the side reader's Lineage of a trial.
import { useEffect, useMemo, useState } from 'react'
import * as api from './api'
import type { PendingTrial, Placed, RoundPlan, Run, Trial } from './api'
import { family, idOf, keyOf, layout, steppedRounds } from './lineage'
import type { LineageNode } from './lineage'
import type { Picked } from './Round'

const fmt = (v: number | null | undefined, digits = 2) => (v == null ? '–' : (v > 0 ? '+' : '') + v.toFixed(digits))

// The stepped round, following the latest while the reader has not stepped back.
export function useStepper(run: Run) {
  const rounds = steppedRounds(run.summary.rounds, run.running && { round: run.running.round, planned: run.running.pending.length > 0 })
  const [chosen, setChosen] = useState<number | null>(null)
  const last = rounds[rounds.length - 1] ?? null
  const round = chosen != null && rounds.includes(chosen) ? chosen : last
  const at = round == null ? -1 : rounds.indexOf(round)
  const go = (next: number) => setChosen(next === last ? null : next)
  return { rounds, round, go, previous: at > 0 ? rounds[at - 1] : null, next: at >= 0 && at < rounds.length - 1 ? rounds[at + 1] : null }
}
export type Stepper = ReturnType<typeof useStepper>

// What the graph shows of a trial, written or pending.
type GraphNode = LineageNode & {
  action: string; outcome: string; parentPicked: boolean; pickedIn: number[]; open: Picked; title: string
}

function nodes(run: Run): GraphNode[] {
  const written: GraphNode[] = run.trials.map(t => {
    const action = t.direction?.action || 'new'
    return {
      id: idOf(t.round, t.slot), round: t.round, parent: t.direction?.parent ?? '', action, outcome: t.status,
      parentPicked: t.parent_picked, pickedIn: t.picked_in, open: { trial: `${t.round}.${t.slot}` },
      title: `${idOf(t.round, t.slot)} · ${action} · ${t.status}${t.picked_in.length ? ` · picked parent in round${t.picked_in.length > 1 ? 's' : ''} ${t.picked_in.join(', ')}` : ''}`,
    }
  })
  const pending = (run.running?.pending ?? []).map((p: PendingTrial) => {
    const id = idOf(run.running!.round, p.slot), action = p.direction.action || 'new'
    return {
      id, round: run.running!.round, parent: p.direction.parent ?? '', action, outcome: 'pending', parentPicked: p.parent_picked,
      pickedIn: [], open: { slot: p.slot }, title: `${id} · ${action} · pending`,
    }
  })
  return [...written, ...pending]
}

export function StepperBar({ stepper, children }: { stepper: Stepper; children?: React.ReactNode }) {
  return (
    <div className="stepper">
      <button disabled={stepper.previous == null} onClick={() => stepper.go(stepper.previous!)}>previous</button>
      <b>Round {stepper.round}</b>
      <button disabled={stepper.next == null} onClick={() => stepper.go(stepper.next!)}>next</button>
      {children}
    </div>
  )
}

// The stepped round's plan, read when the stepper reaches it and again when a round is written; null until it is
// read, and while the plan read is of another round.
export function useRoundPlan(name: string, run: Run, round: number | null) {
  const [plan, setPlan] = useState<RoundPlan | null>(null)
  const [error, setError] = useState('')
  const written = run.summary.rounds
  useEffect(() => {
    if (round == null) return
    let live = true
    setError('')
    api.plan(name, round).then(p => live && setPlan(p), e => live && setError(String(e)))
    return () => { live = false }
  }, [name, round, written])
  return { plan: plan && plan.round === round ? plan : null, error }
}

// The stepped round's plan.
export function RoundPlanView({ run, round, plan, error, picked, pick }: {
  run: Run; round: number; plan: RoundPlan | null; error: string; picked: Picked; pick: (p: Picked) => void
}) {
  const written = run.summary.rounds
  if (error) return <p className="error small">{error}</p>
  if (!plan) return <p className="muted small">Reading round {round}'s plan…</p>
  const pending = round > written
  const quota = Object.entries(plan.quota).filter(([, n]) => n > 0)
  const on = (s: { slot: number }) => picked != null && (pending
    ? 'slot' in picked && picked.slot === s.slot : 'trial' in picked && picked.trial === `${round}.${s.slot}`)
  return (
    <section className="plan">
      <div className="slots">
        {plan.slots.map(s => (
          <button key={s.slot} className={`slot o-${s.outcome ?? 'pending'}${s.picked ? ' picked' : ''}${on(s) ? ' on' : ''}`}
                  title={s.direction} onClick={() => pick(pending ? { slot: s.slot } : { trial: `${round}.${s.slot}` })}>
            <span className="muted">{s.mechanism_type}</span>
            <span>{s.action}{s.parent && <> of <span className="mono">{s.parent}</span></>}</span>
            {s.picked && <span className="tag">picked parent</span>}
          </button>
        ))}
      </div>
      {quota.length > 0 && <p className="muted small">Quota: {quota.map(([kind, n]) => `${kind} ${n}`).join(' · ')}</p>}
      {plan.issues.length > 0 && <ul className="issues">{plan.issues.map((x, i) => <li key={i}>{x}</li>)}</ul>}
    </section>
  )
}

function Arrow({ from, to, className }: { from: [number, number]; to: [number, number]; className: string }) {
  const [x1, y1] = from, [x2, y2] = to
  const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1, ux = dx / len, uy = dy / len, nx = -uy, ny = ux
  const sx = x1 + ux * 5, sy = y1 + uy * 5, ex = x2 - ux * 7, ey = y2 - uy * 7
  return (
    <g className={`arrow ${className}`}>
      <line x1={sx} y1={sy} x2={ex - ux * 5} y2={ey - uy * 5} />
      <polygon points={`${ex},${ey} ${ex - ux * 6 + nx * 3},${ey - uy * 6 + ny * 3} ${ex - ux * 6 - nx * 3},${ey - uy * 6 - ny * 3}`} />
    </g>
  )
}

function Shape({ x, y, action, className, onClick, title }: {
  x: number; y: number; action: string; className: string; onClick: () => void; title: string
}) {
  const props = { className, onClick }
  if (action === 'refinement') return <rect x={x - 4} y={y - 4} width={8} height={8} {...props}><title>{title}</title></rect>
  if (action === 'mutation') {
    return <polygon points={`${x},${y - 5.5} ${x + 5.5},${y} ${x},${y + 5.5} ${x - 5.5},${y}`} {...props}><title>{title}</title></polygon>
  }
  return <circle cx={x} cy={y} r={4.5} {...props}><title>{title}</title></circle>
}

function Key({ picks }: { picks: boolean }) {
  const glyph = (children: React.ReactNode) => <svg viewBox="-7 -7 14 14" width={14} height={14}>{children}</svg>
  return (
    <div className="lineage-key muted small">
      <span>{glyph(<circle r={4.5} className="glyph" />)} new</span>
      <span>{glyph(<rect x={-4} y={-4} width={8} height={8} className="glyph" />)} refinement</span>
      <span>{glyph(<polygon points="0,-5.5 5.5,0 0,5.5 -5.5,0" className="glyph" />)} mutation</span>
      {picks && <span><svg viewBox="0 -3 22 6" width={22} height={6}><line x1={0} x2={22} className="edge" /></svg> parent picked by the code</span>}
      <span><svg viewBox="0 -3 22 6" width={22} height={6}><line x1={0} x2={22} className="edge planned" /></svg> parent chosen by the planner</span>
      {picks && <span>{glyph(<circle r={5.5} className="pick-ring" />)} Picked parent</span>}
    </div>
  )
}

// Every trial of the run in a column of its round. With a trial selected its family is lit; otherwise the stepped
// round's trials and the parents they built on are. Clicking a ring of the stepped round's picks selects its trial and
// calls `onPicks`, which opens the propagation view.
export function LineageGraph({ run, stepper, selected, pick, onPicks }: {
  run: Run; stepper: Stepper; selected: Picked; pick: (p: Picked) => void; onPicks?: () => void
}) {
  const all = useMemo(() => nodes(run), [run])
  const at = useMemo(() => layout(all, stepper.rounds), [all, stepper.rounds.join()])
  const round = stepper.round
  if (!stepper.rounds.length || round == null) return null
  const CW = 92, RH = 15, PX = 22, PY = 28
  const xy = (id: string): [number, number] | null => {
    const p = at.get(id)
    return p ? [PX + p[0] * CW, PY + p[1] * RH] : null
  }
  const rows = Math.max(1, ...[...at.values()].map(p => p[1] + 1))
  const W = PX * 2 + (stepper.rounds.length - 1) * CW, H = PY + (rows - 1) * RH + 14
  const selectedId = selected && 'trial' in selected ? (() => { const [r, s] = selected.trial.split('.'); return idOf(Number(r), Number(s)) })()
    : selected && 'slot' in selected && run.running ? idOf(run.running.round, selected.slot) : null
  const kin = family(all, selectedId)
  const inRound = new Set(all.filter(n => n.round === round).map(n => n.id))
  const parents = new Set(all.filter(n => inRound.has(n.id)).map(n => n.parent))
  const lit = (id: string) => (selectedId ? kin.has(id) : inRound.has(id) || parents.has(id))
  const pickedNow = (n: GraphNode) => n.pickedIn.includes(round)
  const byId = new Map(all.map(n => [n.id, n]))
  return (
    <figure className="lineage">
      <Key picks={all.some(n => n.pickedIn.length > 0)} />
      <svg viewBox={`0 0 ${W} ${H}`} style={{ maxWidth: W * 1.4 }} role="img"
           aria-label="every trial in a column of its round, with an arrow from each parent to each child">
        {stepper.rounds.map((r, c) => (
          <g key={r} className={`column${r === round ? ' on' : ''}`}>
            <rect x={PX + c * CW - 13} y={0} width={26} height={H} />
            <text x={PX + c * CW} y={13} textAnchor="middle" onClick={() => stepper.go(r)}>{r}<title>{`round ${r}`}</title></text>
          </g>
        ))}
        {all.map(n => {
          const from = xy(n.parent), to = xy(n.id)
          if (!from || !to) return null
          const parent = byId.get(n.parent)!
          const now = n.round === round && n.parentPicked && pickedNow(parent)
          const on = selectedId ? kin.has(n.id) && kin.has(n.parent) : n.round === round
          return <Arrow key={n.id} from={from} to={to}
                        className={`${n.parentPicked ? 'picked' : 'planned'}${now ? ' now' : ''}${on ? '' : ' dim'}`} />
        })}
        {all.filter(n => n.pickedIn.length).map(n => {
          const [x, y] = xy(n.id) ?? [0, 0]
          if (!at.has(n.id)) return null
          return (
            <g key={`rings ${n.id}`} className={`picks${lit(n.id) || pickedNow(n) ? '' : ' dim'}`}
               onClick={() => { pick(n.open); if (pickedNow(n)) onPicks?.() }}>
              {n.pickedIn.slice(0, 3).map((r, i) => <circle key={r} cx={x} cy={y} r={8 + i * 2.6} className="pick-ring" />)}
              {pickedNow(n) && <circle cx={x} cy={y} r={8} className="pick-now" />}
              <title>{`${n.id}: picked parent in round${n.pickedIn.length > 1 ? 's' : ''} ${n.pickedIn.join(', ')}`}</title>
            </g>
          )
        })}
        {all.map(n => {
          const p = xy(n.id)
          if (!p) return null
          return <Shape key={n.id} x={p[0]} y={p[1]} action={n.action} title={n.title}
                        className={`node o-${n.outcome}${n.id === selectedId ? ' selected' : ''}${lit(n.id) || pickedNow(n) ? '' : ' faint'}`}
                        onClick={() => pick(n.open)} />
        })}
      </svg>
    </figure>
  )
}

function TrialLink({ id, pick }: { id: string; pick: (p: Picked) => void }) {
  const key = keyOf(id)
  return key ? <button className="link mono" onClick={() => pick({ trial: key })}>{id}</button> : <span className="mono">{id}</span>
}

// The reader's Lineage of a trial: its parent, the counts of its children, its own t against its best child's, its
// fertility, and each child with its action and outcome.
export function LineageSection({ t, trials, placed, pick }: {
  t: Trial; trials: Trial[]; placed: Map<string, Placed>; pick: (p: Picked) => void
}) {
  const l = t.lineage
  if (!l) return null
  const id = idOf(t.round, t.slot)
  const children = trials.filter(c => c.direction?.parent === id)
  const parent = t.direction?.parent
  const top = Math.max(1e-9, ...[l.own_t, l.best_child_t].filter((v): v is number => v != null).map(Math.abs))
  const bar = (v: number | null) => ({ width: `${(Math.abs(v ?? 0) / top) * 100}%` })
  return (
    <section className="lineage-reader">
      <h4>Lineage</h4>
      {parent
        ? <p className="small">{t.direction?.action} of <TrialLink id={parent} pick={pick} />, {t.parent_picked ? 'a Picked parent' : 'chosen by the planner'}</p>
        : <p className="small muted">a new direction</p>}
      {t.picked_in.length > 0 && <p className="small">Picked parent in round{t.picked_in.length > 1 ? 's' : ''} {t.picked_in.join(', ')}</p>}
      {l.attempts === 0 ? <p className="muted small">Nothing built on it yet.</p> : <>
        <div className="counts">
          <span><b>{l.attempts}</b> built on it</span>
          <span><b>{l.refinements}</b> refinement{l.refinements === 1 ? '' : 's'}</span>
          <span><b>{l.mutations}</b> mutation{l.mutations === 1 ? '' : 's'}</span>
          <span><b>{l.unevaluated}</b> stopped</span>
          <span className="good"><b>{l.accepted.length}</b> accepted</span>
          <span><b>{l.fertility.toFixed(2)}</b> fertility</span>
        </div>
        {l.best_child && (
          <div className="tbars small">
            <span className="mono">{id}</span><i style={bar(l.own_t)} /><span>{fmt(l.own_t)}</span>
            <TrialLink id={l.best_child} pick={pick} /><i className="best" style={bar(l.best_child_t)} /><span>{fmt(l.best_child_t)}</span>
          </div>
        )}
        <ul className="children small">
          {children.map(c => {
            const cid = idOf(c.round, c.slot)
            const tv = placed.get(`${c.round}.${c.slot}`)?.partial_ic_t
            return (
              <li key={cid}>
                <span className={`key ${c.status}`}>●</span> <TrialLink id={cid} pick={pick} /> {c.direction?.action}{' '}
                <span className="muted">{c.status}{tv != null ? `, t ${fmt(tv)}` : ''}</span>
              </li>
            )
          })}
        </ul>
      </>}
    </section>
  )
}
