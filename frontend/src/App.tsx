// The run list, and a run opened from it (?run=<name>).
import { useEffect, useState } from 'react'
import * as api from './api'
import type { Run, RunSummary } from './api'
import Memory from './Memory'
import Results from './Results'
import Trials from './Trials'

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

// A run's page (&page=<page>): its results by default.
const PAGES = ['results', 'trials', 'memory'] as const
type Page = (typeof PAGES)[number]

const pageParam = (): Page => {
  const page = new URLSearchParams(location.search).get('page') as Page | null
  return page && PAGES.includes(page) ? page : 'results'
}

function usePageParam(): [Page, (page: Page) => void] {
  const [page, setPage] = useState<Page>(pageParam)
  useEffect(() => {
    const back = () => setPage(pageParam())
    addEventListener('popstate', back)
    return () => removeEventListener('popstate', back)
  }, [])
  const open = (next: Page) => {
    history.pushState(null, '', pageHref(next))
    setPage(next)
  }
  return [page, open]
}

function pageHref(page: Page): string {
  const params = new URLSearchParams(location.search)
  if (page === 'results') params.delete('page')
  else params.set('page', page)
  return `?${params}`
}

function Pages({ page, open }: { page: Page; open: (page: Page) => void }) {
  return (
    <nav className="pages">
      {PAGES.map(p => (
        <a key={p} href={pageHref(p)} className={p === page ? 'here' : ''} onClick={e => { e.preventDefault(); open(p) }}>{p}</a>
      ))}
    </nav>
  )
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
  const [page, openPage] = usePageParam()
  useEffect(() => { setRun(null); api.run(name).then(setRun, e => setError(String(e))) }, [name])
  if (error) return <p className="error">{error}</p>
  if (!run) return <p className="muted">Loading…</p>
  return (
    <>
      <Pages page={page} open={openPage} />
      {page === 'memory' ? <Memory versions={run.memory} /> : page === 'trials' ? <Trials run={run} /> : <Results run={run} />}
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
