// The round in progress on the trials page: a timeline of its LLM calls by role and its evaluation, from the end of
// the round before to the page's clock, with its pending trials one expand away; and the side panel's readers of a
// call (fetched by its call file's number) and of a pending trial.
import { useEffect, useState } from 'react'
import * as api from './api'
import type { Check, LLMCall, PendingTrial, RoundCall, RunningRound, Stage } from './api'
import { duration } from './Health'
import { rows } from './lanes'
import { formatted } from './reply'

// What the side panel reads: a trial (by "round.slot"), a call (by its call file's number) or a pending trial (slot),
// with the call it was opened from, so that the panel can go back to that call.
export type Picked = { trial: string } | { call: number } | { slot: number; from?: number } | null

export const STAGES: Record<Stage, string> = {
  planned: 'planned',
  proposed: 'proposed',
  fixed: 'fixed after the Gate',
  rewritten: 'rewritten after review',
  consistent: 'found consistent',
  inconsistent: 'found inconsistent',
}

const ROLES = ['planner', 'proposer', 'reviewer']
const time = (iso: string) => new Date(iso).toLocaleTimeString()
const count = (n: number | null) => (n == null ? '–' : n.toLocaleString())

// `end` is the page's clock, or the time the round was last heard of when its campaign crashed: a call still out
// then will never come back.
function CallMark({ c, x, end, picked, pick }: {
  c: RoundCall; x: (ms: number) => number; end: number; picked: Picked; pick: (p: Picked) => void
}) {
  const from = x(Date.parse(c.sent)), to = x(c.returned ? Date.parse(c.returned) : end)
  const state = c.error ? 'failed' : c.number == null ? 'out' : 'back'
  const on = picked != null && 'call' in picked && picked.call === c.number
  const slots = c.slots.length ? `, slots ${c.slots.map(s => s + 1).join(', ')}` : ''
  const title = c.error ? `${c.role}, sent ${time(c.sent)}: failed (${c.error})`
    : c.number == null ? `${c.role}, sent ${time(c.sent)}: no reply yet${slots}`
      : `call ${c.number}, ${c.role}, back ${time(c.returned!)}${slots}: click to read it`
  return (
    <button className={`mark ${state}${on ? ' picked' : ''}`} style={{ left: `${from}%`, width: `max(${to - from}%, ${LEAST}%)` }}
            disabled={c.number == null} title={title} aria-label={title} onClick={() => pick({ call: c.number! })} />
  )
}

// The least width of a call on the timeline, in percent of it, so that a short call can still be clicked.
const LEAST = 1.5

// A crashed round's timeline ends when the round was last heard of, not at the page's clock, so that its calls are
// not squeezed into the start of a timeline that keeps growing after the crash.
function Timeline({ running, clock, crashed, picked, pick }: {
  running: RunningRound; clock: number; crashed: boolean; picked: Picked; pick: (p: Picked) => void
}) {
  const t0 = Date.parse(running.started)
  const heard = Math.max(t0, ...running.calls.flatMap(c => [Date.parse(c.sent), c.returned ? Date.parse(c.returned) : 0]),
    running.evaluation ? Date.parse(running.evaluation) : 0)
  const end = crashed ? heard : clock
  const t1 = Math.max(end, t0 + 1000)
  const x = (ms: number) => Math.min(100, Math.max(0, ((ms - t0) / (t1 - t0)) * 100))
  const roles = [...ROLES, ...new Set(running.calls.map(c => c.role).filter(r => !ROLES.includes(r)))]
  return (
    <div className="timeline">
      {roles.map(role => (
        <div key={role} className="lane">
          <span className="role">{role}</span>
          <div className="rows">
            {rows(running.calls.filter(c => c.role === role).map((c, i) => ({
              from: x(Date.parse(c.sent)), to: x(c.returned ? Date.parse(c.returned) : end), item: { c, i },
            })), LEAST).map((row, r) => (
              <div key={r} className="track">
                {row.map(({ c, i }) => <CallMark key={`${c.sent}-${i}`} c={c} x={x} end={end} picked={picked} pick={pick} />)}
              </div>
            ))}
            {!running.calls.some(c => c.role === role) && <div className="track" />}
          </div>
        </div>
      ))}
      <div className="lane">
        <span className="role">evaluation</span>
        <div className="rows">
          <div className="track">
            {running.evaluation && (
              <span className="mark evaluating" style={{ left: `${x(Date.parse(running.evaluation))}%`, right: 0 }}
                    title={`the Evaluator started at ${time(running.evaluation)}`} />
            )}
          </div>
        </div>
      </div>
      <div className="axis muted small">
        <span>{time(running.started)}</span>
        <span>{duration((t1 - t0) / 1000)} {crashed ? `until it was last heard of, at ${time(new Date(heard).toISOString())}` : 'so far'}</span>
      </div>
    </div>
  )
}

function PendingCards({ pending, picked, pick }: { pending: PendingTrial[]; picked: Picked; pick: (p: Picked) => void }) {
  if (!pending.length) return <p className="muted small">No pending trials yet.</p>
  const by = (stage: Stage) => pending.filter(p => p.stage === stage).length
  const summary = (Object.keys(STAGES) as Stage[]).filter(by).map(s => `${by(s)} ${STAGES[s]}`).join(', ')
  return (
    <details className="pending">
      <summary>Pending trials <span className="muted">· {pending.length} slots: {summary}</span></summary>
      <div className="cards">
        {pending.map(p => (
          <button key={p.slot} className={`card st-${p.stage}${picked && 'slot' in picked && picked.slot === p.slot ? ' picked' : ''}`}
                  onClick={() => pick({ slot: p.slot })} title={p.direction.direction}>
            <b>slot {p.slot + 1}</b> <span className="muted">{p.direction.mechanism_type}</span>
            <span className="stage">{STAGES[p.stage]}</span>
          </button>
        ))}
      </div>
    </details>
  )
}

// `crashed`: the campaign's process exited mid-round; resuming it drops the round and runs it again.
export function RoundStrip({ running, clock, crashed, picked, pick }: {
  running: RunningRound; clock: number; crashed: boolean; picked: Picked; pick: (p: Picked) => void
}) {
  const back = running.calls.filter(c => c.number != null).length
  return (
    <section className="round">
      <h3>
        Round {running.round} {crashed ? '(crashed)' : 'in progress'}{' '}
        <span className="muted">
          · {back} call{back === 1 ? '' : 's'} back{running.evaluation ? ' · evaluating' : ''}
          {crashed && ' · rerun on resume'}
        </span>
      </h3>
      <Timeline running={running} clock={clock} crashed={crashed} picked={picked} pick={pick} />
      <PendingCards pending={running.pending} picked={picked} pick={pick} />
    </section>
  )
}

function Slots({ slots, from, pick }: { slots: number[]; from: number; pick: (p: Picked) => void }) {
  if (!slots.length) return <p className="small muted">Whole round.</p>
  return (
    <p className="small">
      Slots:{' '}
      {slots.map(s => <button key={s} className="link" onClick={() => pick({ slot: s, from })}>{s + 1}</button>)
        .reduce<React.ReactNode[]>((all, b, i) => (i ? [...all, ', ', b] : [b]), [])}
    </p>
  )
}

export function CallReader({ name, number, pick }: { name: string; number: number; pick: (p: Picked) => void }) {
  const [call, setCall] = useState<LLMCall | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    let live = true
    setCall(null)
    setError('')
    api.call(name, number).then(c => live && setCall(c), e => live && setError(String(e)))
    return () => { live = false }
  }, [name, number])
  if (error) return <p className="error small">{error}</p>
  if (!call) return <p className="muted">Reading call {number}…</p>
  const t = call.tokens
  return (
    <>
      <div className="muted small">call {call.number}{call.round != null && <>, round {call.round}</>}</div>
      <h3>{call.role} <span className="muted">· {call.model}</span></h3>
      <p className="small">
        back at {time(call.time)}{call.seconds != null && <> after {duration(call.seconds)}</>} ·{' '}
        {count(t.input)} in{t.cache_hits != null && <> ({count(t.cache_hits)} from the cache)</>}, {count(t.output)} out
        {t.reasoning != null && <> ({count(t.reasoning)} reasoning)</>}
      </p>
      {call.truncated && <p className="flag">Cut off at the output limit.</p>}
      {call.empty && <p className="flag">Empty reply.</p>}
      <Slots slots={call.slots} from={call.number} pick={pick} />
      <h4>Reply</h4>
      {call.reply ? <pre className="code">{formatted(call.reply)}</pre> : <p className="muted small">(empty)</p>}
      {call.reasoning && (
        <details open={call.empty}>
          <summary className="small">Reasoning ({call.reasoning.length.toLocaleString()} chars)</summary>
          <pre className="code">{call.reasoning}</pre>
        </details>
      )}
      <details>
        <summary className="small">Prompt ({call.prompt.length.toLocaleString()} chars)</summary>
        <pre className="code">{call.prompt}</pre>
      </details>
    </>
  )
}

function CheckLine({ c }: { c: Check }) {
  const what = c.by === 'reviewer'
    ? (c.passed ? 'the reviewer found it consistent' : 'the reviewer found it inconsistent')
    : `${c.step} version ${c.passed ? 'passed the Gate' : c.stage === 'parse' ? 'could not be read' : 'refused by the Gate'}`
  return (
    <li>
      <span className={c.passed ? 'good' : 'bad'}>{what}</span> <span className="muted small">{time(c.time)}</span>
      {c.reasons.map((r, i) => <p key={i} className="reason">{r}</p>)}
      {c.hints.map((h, i) => <p key={i} className="hint">Hint: {h}</p>)}
    </li>
  )
}

// `back`, when the pending trial was opened from a call's slots: return to that call.
export function PendingReader({ p, back }: { p: PendingTrial | undefined; back?: { call: number; go: () => void } }) {
  const backButton = back && <button className="link back small" onClick={back.go}>← back to call {back.call}</button>
  if (!p) return <>{backButton}<p className="muted">Round written: see the map or list.</p></>
  const item = p.latest
  const candidate = item && typeof item === 'object' ? item as Record<string, unknown> : null
  return (
    <>
      {backButton}
      <div className="muted small">
        pending trial, slot {p.slot + 1} · {p.direction.mechanism_type} · {p.direction.action}
        {p.direction.parent ? ` of ${p.direction.parent}` : ''}
      </div>
      <h3>{candidate?.description != null ? String(candidate.description) : `slot ${p.slot + 1}`}</h3>
      <span className={`badge st-${p.stage}`}>{STAGES[p.stage]}</span>
      <h4>Research direction</h4><p>{p.direction.direction}</p>
      {item == null ? <p className="muted">No version proposed yet.</p> : candidate ? <>
        <h4>Hypothesis</h4><p>{String(candidate.hypothesis ?? '–')}</p>
        <h4>Latest version</h4>
        <pre className="code">{String(candidate.code ?? JSON.stringify(candidate, null, 2))}</pre>
      </> : <><h4>Latest version</h4><pre className="code">{String(item)}</pre></>}
      {p.checks.length > 0 && <>
        <h4>So far</h4>
        <ol className="versions">{p.checks.map((c, i) => <CheckLine key={i} c={c} />)}</ol>
      </>}
    </>
  )
}
