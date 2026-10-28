// The diagnosis on the results page: at most three findings, each what was seen and what to do,
// the one concerning the most trials first; prominent while fewer than three factors are accepted.
import type { Run } from './api'

export default function Diagnosis({ run }: { run: Run }) {
  const findings = run.diagnosis
  if (!findings.length) return null
  const prominent = run.summary.accepted < 3
  const heading = run.summary.accepted === 0
    ? (run.summary.stop ? 'Why nothing was accepted' : 'Why nothing yet')
    : 'Worth checking'
  return (
    <section className={`diagnosis${prominent ? ' prominent' : ''}`} aria-label="diagnosis">
      <h2>{heading}</h2>
      <ul>
        {findings.map(f => (
          <li key={f.rule}>
            <p className="seen">{f.seen}</p>
            <p className="do">{f.do}</p>
          </li>
        ))}
      </ul>
    </section>
  )
}
