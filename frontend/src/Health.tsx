// The health line every page of a run shows: stopped (and why) or what the run is doing, how long ago the last
// LLM call came back, the round against the round limit with the time left, the tokens used, and a warning when the
// run looks stuck or its process has gone. `clock` is the page's time (ms), moving between two reads of the run.
import type { Run, Tokens } from './api'

const count = (n: number | null) =>
  n == null ? '–' : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e4 ? `${Math.round(n / 1e3)}k` : String(n)

export function duration(seconds: number): string {
  if (seconds < 90) return `${Math.round(seconds)} s`
  if (seconds < 90 * 60) return `${Math.round(seconds / 60)} min`
  if (seconds < 48 * 3600) return `${(seconds / 3600).toFixed(1)} h`
  return `${Math.round(seconds / 86400)} days`
}

const PHASES: Record<string, string> = {
  planning: 'planning',
  asking: 'asking the LLM',
  evaluating: 'evaluating',
}

function TokenLine({ tokens }: { tokens: Tokens }) {
  return (
    <>
      {count(tokens.input)} in{tokens.cache_hits != null && <> ({count(tokens.cache_hits)} cached)</>} /{' '}
      {count(tokens.output)} out{tokens.reasoning != null && <> ({count(tokens.reasoning)} reasoning)</>}
    </>
  )
}

export default function Health({ run, clock }: { run: Run; clock: number }) {
  const s = run.summary, h = run.health
  // Seconds since the run was read: what the server said then has aged by this much.
  const aged = Math.max(0, (clock - Date.parse(run.now)) / 1000)
  const since = (iso: string) => Math.max(0, (clock - Date.parse(iso)) / 1000)
  const written = <>{s.rounds} round{s.rounds === 1 ? '' : 's'} · {s.trials} trial{s.trials === 1 ? '' : 's'} · {s.accepted} accepted</>
  const roles = Object.entries(h.calls)
  return (
    <div className="health">
      <p>
        {s.stop ? <><b>Stopped</b>: {s.stop.reason}</>
          : h.crashed ? <b className="warn">Crashed</b>
          : h.phase ? <><b>Round {h.round}{h.max_rounds != null && <>/{h.max_rounds}</>}</b>: {PHASES[h.phase] ?? h.phase}</>
          : <><b>Not stopped</b>{h.round != null && <>, round {h.round}{h.max_rounds != null && <>/{h.max_rounds}</>}</>}</>}
        {' · '}{written}
        {h.last_call && <> · last call {duration(since(h.last_call))} ago</>}
        {!s.stop && !h.crashed && h.time_left != null && <> · ~{duration(Math.max(0, h.time_left - aged))} left</>}
        {h.tokens && (h.tokens.input != null || h.tokens.output != null) && <> · tokens <TokenLine tokens={h.tokens} /></>}
      </p>
      {h.crashed && (
        <p className="warn">Process exited without a stop. Rerun the campaign to resume.</p>
      )}
      {h.stalled && <p className="warn">No LLM call for over 15 min: may be stuck.</p>}
      {!s.stop && !h.crashed && h.process === 'unknown' && h.phase && (
        <p className="muted small">Process on another host: liveness unknown.</p>
      )}
      {!s.stop && !h.phase && !h.crashed && (
        <p className="muted small">
          No events file: calls only.{' '}
          {roles.length ? <>Round {h.round}: {roles.map(([role, n]) => `${role} ${n}`).join(', ')}</> : 'No call yet.'}
        </p>
      )}
      {h.round_tokens.length > 1 && (
        <details className="small">
          <summary>Tokens by round</summary>
          <table className="tokens">
            <thead><tr><th>round</th><th>in</th><th>cached</th><th>out</th><th>reasoning</th></tr></thead>
            <tbody>
              {h.round_tokens.map(r => (
                <tr key={r.round}>
                  <td>{r.round}</td><td>{count(r.tokens.input)}</td><td>{count(r.tokens.cache_hits)}</td>
                  <td>{count(r.tokens.output)}</td><td>{count(r.tokens.reasoning)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}
    </div>
  )
}
