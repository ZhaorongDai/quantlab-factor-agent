// The run list, and a run opened from it (?run=<name>).
import { useEffect, useState } from 'react'
import * as api from './api'
import type { Run, RunSummary } from './api'

function useRunParam(): [string, (name: string) => void] {
  const [name, setName] = useState(new URLSearchParams(location.search).get('run') ?? '')
  useEffect(() => {
    const back = () => setName(new URLSearchParams(location.search).get('run') ?? '')
    addEventListener('popstate', back)
    return () => removeEventListener('popstate', back)
  }, [])
  const open = (next: string) => {
    history.pushState(null, '', next ? `?run=${encodeURIComponent(next)}` : location.pathname)
    setName(next)
  }
  return [name, open]
}

function ago(iso: string): string {
  const seconds = Math.max(0, (Date.now() - Date.parse(iso)) / 1000)
  if (seconds < 90) return 'just now'
  if (seconds < 90 * 60) return `${Math.round(seconds / 60)} min ago`
  if (seconds < 36 * 3600) return `${Math.round(seconds / 3600)} h ago`
  return `${Math.round(seconds / 86400)} days ago`
}

const version = (v: string) => (v.length > 12 && !v.startsWith('unknown') ? v.slice(0, 7) + (v.endsWith('+dirty') ? '+dirty' : '') : v)

function State({ summary }: { summary: RunSummary }) {
  if (!summary.stop) return <span>not stopped</span>
  return <span title={summary.stop.reason}>stopped: {summary.stop.reason}</span>
}

function RunList({ open }: { open: (name: string) => void }) {
  const [runs, setRuns] = useState<RunSummary[] | null>(null)
  const [error, setError] = useState('')
  useEffect(() => { api.runs().then(setRuns, e => setError(String(e))) }, [])
  if (error) return <p className="error">{error}</p>
  if (!runs) return <p className="muted">Loading…</p>
  if (!runs.length) return <p className="muted">No run directory under this folder: a run directory holds a snapshot.json from its campaign's first start.</p>
  return (
    <table className="runs">
      <thead><tr><th>run</th><th>code version</th><th>rounds</th><th>trials</th><th>accepted</th><th>state</th><th>last activity</th></tr></thead>
      <tbody>
        {runs.map(r => (
          <tr key={r.name} onClick={() => open(r.name)}>
            <td><a href={`?run=${encodeURIComponent(r.name)}`} onClick={e => e.preventDefault()}>{r.name}</a></td>
            <td className="mono" title={r.code_version}>{version(r.code_version)}</td>
            <td>{r.rounds}</td><td>{r.trials}</td><td>{r.accepted}</td>
            <td><State summary={r} /></td>
            <td title={new Date(r.updated).toLocaleString()}>{ago(r.updated)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function RunPage({ name }: { name: string }) {
  const [run, setRun] = useState<Run | null>(null)
  const [error, setError] = useState('')
  useEffect(() => { setRun(null); api.run(name).then(setRun, e => setError(String(e))) }, [name])
  if (error) return <p className="error">{error}</p>
  if (!run) return <p className="muted">Loading…</p>
  const s = run.summary
  return (
    <>
      <p className="facts">
        {s.rounds} round{s.rounds === 1 ? '' : 's'} written, {s.trials} trials, {s.accepted} accepted; <State summary={s} />;
        last activity {ago(s.updated)}; code version <span className="mono" title={s.code_version}>{version(s.code_version)}</span>.
      </p>
      <table className="runs">
        <thead><tr><th>round</th><th>slot</th><th>status</th><th>stage</th><th>description</th></tr></thead>
        <tbody>
          {run.trials.map(t => (
            <tr key={`${t.round}.${t.slot}`} className="static">
              <td>{t.round}</td><td>{t.slot + 1}</td><td><span className={`badge ${t.status}`}>{t.status}</span></td><td>{t.stage}</td>
              <td>{String(t.candidate.description ?? '')}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  )
}

export default function App() {
  const [name, open] = useRunParam()
  useEffect(() => { document.title = name ? `${name} · factor-agent` : 'factor-agent' }, [name])
  return (
    <>
      <header className="top">
        <a href="?" onClick={e => { e.preventDefault(); open('') }}>factor-agent</a>
        {name && <> / <b>{name}</b></>}
      </header>
      <main>{name ? <RunPage name={name} /> : <RunList open={open} />}</main>
    </>
  )
}
