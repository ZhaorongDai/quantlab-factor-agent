// The run list, and a run opened from it (?run=<name>).
import { useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import * as api from './api'
import type { RunSummary } from './api'
import Health from './Health'
import { useLiveRun, useTabTitle } from './live'
import type { LiveRun } from './live'
import Memory from './Memory'
import Results from './Results'
import Selection from './Selection'
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

// A run's page (&page=<page>): its results by default. The selection view is reached from the results.
const PAGES = ['results', 'selection', 'trials', 'memory'] as const
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

function Pages({ page, open, children }: { page: Page; open: (page: Page) => void; children?: ReactNode }) {
  return (
    <nav className="pages">
      {PAGES.filter(p => p !== 'selection').map(p => (
        <a key={p} href={pageHref(p)} className={p === page || (p === 'results' && page === 'selection') ? 'here' : ''} onClick={e => { e.preventDefault(); open(p) }}>{p}</a>
      ))}
      {children}
    </nav>
  )
}

// The pause of a run that has not stopped: the button, and while paused the bar saying so.
function PauseButton({ live }: { live: LiveRun }) {
  return live.paused == null
    ? <button className="pause" onClick={live.pause} title="Stop the page from updating, to read or click without anything moving">Pause live updates</button>
    : <button className="pause on" onClick={live.resume}>Resume live updates</button>
}

function PausedBar({ live }: { live: LiveRun }) {
  if (live.paused == null) return null
  return (
    <p className="paused">
      Paused since {new Date(live.paused).toLocaleTimeString()}: the page stands as it was then
      {live.changed ? <>; <b>the run has changed since</b></> : <>; the run has not changed since</>}.{' '}
      <button className="link" onClick={live.resume}>Resume</button>
    </p>
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
  const live = useLiveRun(name)
  const { run, error, clock } = live
  const [page, openPage] = usePageParam()
  useTabTitle(name, run)
  if (!run) return error ? <p className="error">{error}</p> : <p className="muted">Loading…</p>
  return (
    <>
      <Pages page={page} open={openPage}>{!run.summary.stop && <PauseButton live={live} />}</Pages>
      <PausedBar live={live} />
      <Health run={run} clock={clock} />
      {error && <p className="error small">{error}</p>}
      {page === 'memory' ? <Memory versions={run.memory} /> : page === 'trials' ? <Trials run={run} name={name} clock={clock} />
        : page === 'selection' ? <Selection run={run} /> : <Results run={run} select={() => openPage('selection')} />}
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
