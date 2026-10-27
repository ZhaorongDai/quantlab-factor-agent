// The memory page: every version the distiller wrote, the latest first and open, each with what changed from the one before.
import type { MemoryLine, MemoryVersion } from './api'

function Changes({ lines }: { lines: MemoryLine[] }) {
  const added = lines.filter(l => l.change === 'added').length
  const removed = lines.filter(l => l.change === 'removed').length
  if (!added && !removed) return <p className="muted">Unchanged from the version before.</p>
  return (
    <>
      <p className="muted">
        {added} line{added === 1 ? '' : 's'} <span className="memory-added">added</span> and {removed}{' '}
        <span className="memory-removed">removed</span> since the version before.
      </p>
      <pre className="memory">
        {lines.map((l, i) => <div key={i} className={`memory-${l.change}`}>{l.line || ' '}</div>)}
      </pre>
    </>
  )
}

function Version({ version, open }: { version: MemoryVersion; open: boolean }) {
  return (
    <details className="memory-version" open={open}>
      <summary>Written after round {version.round}</summary>
      {version.changes ? (
        <>
          <Changes lines={version.changes} />
          <details><summary className="muted">Text alone</summary><pre className="memory">{version.text}</pre></details>
        </>
      ) : (
        <>
          <p className="muted">The run's first memory.</p>
          <pre className="memory">{version.text}</pre>
        </>
      )}
    </details>
  )
}

export default function Memory({ versions }: { versions: MemoryVersion[] }) {
  if (!versions.length)
    return (
      <p className="muted">
        No memory yet: the distiller folds the trials into a memory every <span className="mono">round.memory_every</span> rounds,
        and has not written one in this run.
      </p>
    )
  return <>{[...versions].reverse().map((v, i) => <Version key={v.round} version={v} open={i === 0} />)}</>
}
