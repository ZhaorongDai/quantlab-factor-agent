// The run list, and a run opened from it (?run=<name>).
import { useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import * as api from './api'
import type { RunSummary } from './api'
import Health from './Health'
import { useLiveRun, useTabTitle } from './live'
import type { LiveRun } from './live'
import Config from './Config'
import Memory from './Memory'
import Results from './Results'
import Selection from './Selection'
import Trials from './Trials'
import { pausable } from './poll'
import { series } from './series'

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
const PAGES = ['results', 'selection', 'trials', 'memory', 'config'] as const
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
    ? <button className="pause" onClick={live.pause} title="Freeze the page">Pause</button>
    : <button className="pause on" onClick={live.resume}>Resume</button>
}

function PausedBar({ live }: { live: LiveRun }) {
  if (live.paused == null) return null
  return (
    <p className="paused">
      Paused since {new Date(live.paused).toLocaleTimeString()}{live.changed && <> · <b>run changed</b></>}{' '}
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

const shortCommit = (v: string) => (v.length > 12 && !v.startsWith('unknown') ? v.slice(0, 7) + (v.endsWith('+dirty') ? '+dirty' : '') : v)

const STATE: Record<RunSummary['state'], string> = {
  stopped: 'stopped', running: 'running', crashed: 'crashed', unknown: 'not stopped',
}

// One run of a series: what it accepted as the headline, then how far it got and why it stopped.
function RunCard({ run, version, best, open }: { run: RunSummary; version: string; best: number; open: (name: string) => void }) {
  if (run.unreadable) {
    return (
      <div className="run-card unreadable">
        <div className="head"><b>{version}</b><span className="pill crashed">unreadable</span></div>
        <p className="error small">{run.unreadable}</p>
      </div>
    )
  }
  const rate = run.trials ? run.accepted / run.trials : 0
  const state = run.state
  return (
    <a className={`run-card ${state}`} href={`?run=${encodeURIComponent(run.name)}`} title={run.name}
       onClick={e => { e.preventDefault(); open(run.name) }}>
      <div className="head">
        <b>{version}</b>
        <span className={`pill ${state}`} title={run.stop?.reason}>{state === 'running' && <i className="dot" />}{STATE[state]}</span>
      </div>
      <div className="big">{run.accepted}<span> accepted</span></div>
      <div className="rate" title="the acceptance rate against the best of the listed runs"><i style={{ width: `${best ? (rate / best) * 100 : 0}%` }} /></div>
      <div className="muted small">
        {(rate * 100).toFixed(1)}% of {run.trials} trials · round {run.rounds}{run.max_rounds != null && ` of ${run.max_rounds}`}
      </div>
      <div className="muted small">
        {run.stop ? <span title={run.stop.reason}>{run.stop.reason}</span> : state === 'crashed' ? 'its process exited without a stop' : `last activity ${ago(run.updated)}`}
      </div>
      <div className="mono small muted" title={run.code_version}>{shortCommit(run.code_version)} · {ago(run.updated)}</div>
    </a>
  )
}

// The runs under the folder by series (see series.ts), each run a card.
function RunList({ open }: { open: (name: string) => void }) {
  const [runs, setRuns] = useState<RunSummary[] | null>(null)
  const [error, setError] = useState('')
  useEffect(() => { api.runs().then(setRuns, e => setError(String(e))) }, [])
  if (error) return <p className="error">{error}</p>
  if (!runs) return <p className="muted">Loading…</p>
  if (!runs.length) return <p className="muted">No runs in this folder.</p>
  const best = Math.max(0, ...runs.map(r => (r.trials ? r.accepted / r.trials : 0)))
  return (
    <div className="run-list">
      {series(runs).map(s => (
        <section key={s.name}>
          <h2>{s.name} <span className="muted small">
            · {s.runs.length} run{s.runs.length > 1 ? 's' : ''}, {s.runs.reduce((n, r) => n + r.run.accepted, 0)} factors accepted
          </span></h2>
          <div className="run-cards">
            {s.runs.map(r => <RunCard key={r.run.name} run={r.run} version={r.version} best={best} open={open} />)}
          </div>
        </section>
      ))}
    </div>
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
      <Pages page={page} open={openPage}>{pausable(run.summary.stop != null, run.health.crashed) && <PauseButton live={live} />}</Pages>
      <PausedBar live={live} />
      <Health run={run} clock={clock} />
      {error && <p className="error small">{error}</p>}
      {page === 'config' ? <Config name={name} /> : page === 'memory' ? <Memory versions={run.memory} /> : page === 'trials' ? <Trials run={run} name={name} clock={clock} />
        : page === 'selection' ? <Selection run={run} /> : <Results run={run} name={name} select={() => openPage('selection')} />}
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
