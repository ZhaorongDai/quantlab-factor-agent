// The config page: the config of each start and resume, the latest first, each with what changed from the one
// before, then every file the run was set up with to read in full, the YAML coloured.
import { useEffect, useMemo, useState } from 'react'
import * as api from './api'
import type { ConfigCopy, LineChange, RunFiles, SnapshotFile } from './api'
import { folds, highlight, parts, visible } from './yaml'

const when = (iso: string | null) => (iso ? new Date(iso).toLocaleString() : 'at an unknown time')

// The YAML coloured, with line numbers; a key or list item with lines under it folds them away and back.
export function Yaml({ text }: { text: string }) {
  const body = text.replace(/\n$/, '')
  const lines = useMemo(() => highlight(body), [body])
  const ranges = useMemo(() => folds(body), [body])
  const [folded, setFolded] = useState<Set<number>>(new Set())
  useEffect(() => setFolded(new Set()), [body])
  const toggle = (i: number) => setFolded(f => { const next = new Set(f); if (!next.delete(i)) next.add(i); return next })
  // Each line its own constant: a handler must toggle the line it was drawn for.
  const shown = visible(lines.length, ranges, folded).map(line => {
    const end = ranges.get(line)
    return (
      <div key={line}>
        <span className="y-no">{line + 1}</span>
        {end != null
          ? <button className="y-fold" onClick={() => toggle(line)} title={folded.has(line) ? 'Unfold' : 'Fold'}>{folded.has(line) ? '▸' : '▾'}</button>
          : <span className="y-fold" />}
        {lines[line].map((p, k) => <span key={k} className={`y-${p.kind}`}>{p.text}</span>)}
        {end != null && folded.has(line) && <button className="y-folded" onClick={() => toggle(line)}>… {end - line - 1} lines</button>}
      </div>
    )
  })
  return (
    <>
      {ranges.size > 0 && (
        <div className="y-tools small">
          <button onClick={() => setFolded(new Set(ranges.keys()))}>fold all</button>
          <button onClick={() => setFolded(new Set())}>unfold all</button>
        </div>
      )}
      <pre className="code yaml">{shown}</pre>
    </>
  )
}

// The changed lines alone, a config being long; "…" stands for the kept lines between them.
function Changes({ lines }: { lines: LineChange[] }) {
  const changed = lines.flatMap((l, i) => (l.change === 'kept' ? [] : [i]))
  if (!changed.length) return <p className="muted small">Unchanged.</p>
  return (
    <pre className="code yaml">
      {changed.flatMap((i, k) => [
        ...(k > 0 && changed[k - 1] !== i - 1 ? [<div key={`gap ${i}`} className="muted"><span className="y-no">…</span></div>] : []),
        <div key={i} className={`memory-${lines[i].change}`}>
          <span className="y-no">{lines[i].change === 'added' ? '+' : '−'}</span>
          {parts(lines[i].line).map((p, n) => <span key={n} className={`y-${p.kind}`}>{p.text}</span>)}
        </div>,
      ])}
    </pre>
  )
}

const opened = (c: ConfigCopy) => `${c.opening === 1 ? 'started' : 'resumed'} ${when(c.opened_at)}`

function Openings({ configs }: { configs: ConfigCopy[] }) {
  if (!configs.length) {
    return <p className="muted">No config kept (older run).</p>
  }
  return (
    <>
      {configs.map((c, i) => (
        <details key={c.opening} className="memory-version" open={i === 0}>
          <summary>{c.opening === 1 ? 'Started' : `Resumed (opening ${c.opening})`} <span className="muted small">{when(c.opened_at)}</span></summary>
          {c.changes ? <Changes lines={c.changes} />
            : c.opening === 1 ? null
              : <p className="muted small">Earliest kept; changes unknown.</p>}
        </details>
      ))}
    </>
  )
}

const KINDS: Record<string, string> = { fields: 'field file', operators: 'operator file', prompts: 'prompt template' }
const describe = (f: SnapshotFile) => {
  const [folder, ...file] = f.name.split('/')
  return { title: file.join('/'), sub: KINDS[folder] ?? folder }
}

function Browser({ files }: { files: RunFiles }) {
  const entries = [
    ...files.configs.map((c, i) => ({ key: `config/${c.opening}`, title: i === 0 ? 'campaign config' : `campaign config, opening ${c.opening}`, sub: opened(c), text: c.text })),
    ...files.snapshot.map(f => ({ key: f.name, ...describe(f), text: f.text })),
  ]
  const [picked, pick] = useState('')
  const open = entries.find(e => e.key === picked) ?? entries[0]
  if (!open) return null
  return (
    <div className="files">
      <nav>
        {entries.map(e => (
          <button key={e.key} className={e.key === open.key ? 'here' : ''} onClick={() => pick(e.key)}>
            <b>{e.title}</b><span className="muted small">{e.sub}</span>
          </button>
        ))}
      </nav>
      <section>
        <h3>{open.title} <span className="muted small">· {open.sub}, {open.text.split('\n').length} lines</span></h3>
        <Yaml text={open.text} />
      </section>
    </div>
  )
}

export default function Config({ name }: { name: string }) {
  const [files, setFiles] = useState<RunFiles | null>(null)
  const [error, setError] = useState('')
  useEffect(() => { api.files(name).then(setFiles, e => setError(String(e))) }, [name])
  if (error) return <p className="error">{error}</p>
  if (!files) return <p className="muted">Loading…</p>
  return (
    <>
      <h3>Starts and resumes</h3>
      <Openings configs={files.configs} />
      <h3>Files</h3>
      <Browser files={files} />
    </>
  )
}
