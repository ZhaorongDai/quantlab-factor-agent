// Whether the accepted factors improve the user's model: one sentence over Acceptance's validation comparison,
// the hold-out beside it once opened, and each factor's contribution for its card. Only what the reports hold is shown.
import type { Comparison, Contribution, HoldoutOpen, Run, Stat } from './api'

const STATS: [Stat, string, (v: number) => string][] = [
  ['rank_ic', 'rank IC', v => v.toFixed(4)],
  ['top_k_excess', 'top-k excess return', v => `${(v * 100).toFixed(3)}%`],
  ['top_k_annualised', 'top-k annualised return', v => `${(v * 100).toFixed(1)}%`],
  ['long_short', 'long-short return', v => `${(v * 100).toFixed(3)}%`],
  ['monotonicity', 'decile monotonicity', v => v.toFixed(2)],
]

const show = (v: number | null | undefined, f: (v: number) => string) => (v == null ? '–' : f(v))
const signedWith = (v: number | null | undefined, f: (v: number) => string) =>
  v == null ? '–' : v < 0 ? `−${f(-v)}` : `+${f(v)}`
const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`

// A factor's contribution in one mode, from the validation comparison's contribution table.
export function contributionOf(run: Run, label: string, mode: Contribution['mode']): Contribution | undefined {
  return run.acceptance.validation?.contribution?.find(c => c.factor === label && c.mode === mode)
}

// The drop-one rank IC contribution the cards rank by, when the validation comparison has one for the factor.
export function dropOne(run: Run, label: string): number | null {
  return contributionOf(run, label, 'drop_one')?.difference.rank_ic ?? null
}

const MODES = {
  drop_one: ['drop-one', 'the model with every accepted factor against it without this one'],
  add_one: ['add-one', 'your model with only this factor added against your model'],
} as const

// The card's contribution lines: drop-one and add-one rank IC differences and the seeds agreeing on their sign.
export function ContributionLines({ run, label }: { run: Run; label: string }) {
  const seeds = run.acceptance.validation?.seeds.length ?? 0
  const rows = (['drop_one', 'add_one'] as const).map(mode => [mode, contributionOf(run, label, mode)] as const)
  if (!rows.some(([, c]) => c)) return null
  return (
    <dl className="contribution">
      {rows.map(([mode, c]) => c && (
        <div key={mode} title={MODES[mode][1]}>
          <dt>{MODES[mode][0]}</dt>
          <dd>
            <b className={(c.difference.rank_ic ?? 0) >= 0 ? 'up' : 'down'}>{signedWith(c.difference.rank_ic, v => v.toFixed(4))}</b>{' '}
            rank IC, {c.agreeing.rank_ic ?? '–'}/{seeds} seeds agree
          </dd>
        </div>
      ))}
    </dl>
  )
}

function Figures({ c }: { c: Comparison }) {
  return (
    <table className="comparison">
      <thead><tr><th /><th>without</th><th>with</th><th>difference</th></tr></thead>
      <tbody>
        {STATS.map(([key, name, f]) => (
          <tr key={key}>
            <th>{name}</th>
            <td>{show(c.baseline[key], f)}</td>
            <td>{show(c.with_accepted[key], f)}</td>
            <td className={(c.difference[key] ?? 0) >= 0 ? 'up' : 'down'}>{signedWith(c.difference[key], f)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function Window({ c }: { c: Comparison }) {
  return (
    <p className="muted small">
      Predicted {c.test_window.join(' to ')}, trained up to {c.train_window[1] ?? '–'}; the mean of{' '}
      {plural(c.seeds.length, 'seed')}{c.top_k != null && <>, a plain top-{c.top_k} book</>}; returns fee-free over the label's horizon.
    </p>
  )
}

function sentence(c: Comparison, subject: string): string {
  const rank = (c.difference.rank_ic ?? 0) > 0, excess = (c.difference.top_k_excess ?? 0) > 0
  if (c.verdict === 'improves') return `${subject} improve your model.`
  if (c.verdict === 'does_not_improve') return `${subject} do not improve your model.`
  if (c.verdict === 'mixed') return rank && !excess
    ? `${subject} raise your model's rank IC but not its top-k excess return.`
    : `${subject} raise your model's top-k excess return but not its rank IC.`
  return `Acceptance could not say whether ${subject.toLowerCase()} improve your model.`
}

function Opens({ opens }: { opens: HoldoutOpen[] }) {
  return (
    <ol className="opens small">
      {opens.map(o => (
        <li key={o.opened_at}>{o.opened_at.replace('T', ' ').slice(0, 16)}{o.forced ? ', with force' : ''} <span className="muted mono">{o.code_version.slice(0, 7)}</span></li>
      ))}
    </ol>
  )
}

function Holdout({ c, opens }: { c: Comparison; opens: HoldoutOpen[] }) {
  return (
    <aside className={`holdout${c.forced ? ' forced' : ''}`}>
      <h3>On the hold-out</h3>
      {c.forced && (
        <p className="warning">
          Opened again with force: this hold-out was seen before, so it no longer tests the factors out of sample.
        </p>
      )}
      <p>{sentence(c, c.selection ? `The ${plural(c.factors.length, 'selected factor')}` : `The ${plural(c.factors.length, 'factor')}`)}</p>
      <Figures c={c} />
      <Window c={c} />
      <p className="small">The hold-out lock records {plural(opens.length, 'open')}:</p>
      <Opens opens={opens} />
    </aside>
  )
}

const HOW = `from factor_agent.acceptance import Acceptance
from factor_agent.config import CampaignConfig

Acceptance(CampaignConfig.from_file("<your campaign config>.yaml")).validate()`

export default function Verdict({ run }: { run: Run }) {
  const { validation: v, holdout, holdout_opens: opens } = run.acceptance
  if (!run.factors.length && !v) return null
  if (!v) {
    return (
      <section className="verdict unjudged">
        <h2>Not judged against your model yet</h2>
        <p>
          The accepted factors cleared the campaign's statistical bar, but nothing has compared your model with and
          without them. Run Acceptance on the validation segment, from Python:
        </p>
        <pre className="code">{HOW}</pre>
        <p className="muted small">
          Set <code>acceptance.contribution: both</code> in the config to also measure what each factor adds. The page
          shows the report once it is written.
        </p>
      </section>
    )
  }
  const now = new Set(run.factors.map(f => f.label))
  const judged = v.factors.filter(label => now.has(label)).length
  const subject = v.selection ? `The ${plural(v.factors.length, 'selected factor')}` : `The ${plural(v.factors.length, 'accepted factor')}`
  return (
    <section className={`verdict ${v.verdict ?? 'unknown'}`}>
      <div className="validation">
        <h2>{sentence(v, subject)}</h2>
        {v.selection && <p className="small">This is the validation of a selection: no report judges every accepted factor.</p>}
        {!v.selection && judged < run.factors.length && (
          <p className="small">
            The report judges {judged} of the {run.factors.length} factors accepted now: the run accepted more since.
            Run <code>validate()</code> again to judge them all.
          </p>
        )}
        <h3>On validation, your model without and with them</h3>
        <Figures c={v} />
        <Window c={v} />
        {v.contribution == null && (
          <p className="muted small">This report was written before Acceptance measured each factor's contribution.</p>
        )}
        {v.contribution?.length === 0 && (
          <p className="muted small">
            No contribution per factor: set <code>acceptance.contribution: both</code> in the config and validate again
            to see what each one adds.
          </p>
        )}
      </div>
      {holdout ? <Holdout c={holdout} opens={opens} /> : opens.length > 0 ? (
        <aside className="holdout">
          <h3>On the hold-out</h3>
          <p className="small">
            The hold-out lock records {plural(opens.length, 'open')} but there is no hold-out report: an open stopped before
            writing it. The hold-out may have been seen.
          </p>
          <Opens opens={opens} />
        </aside>
      ) : (
        <aside className="holdout none">
          <h3>On the hold-out</h3>
          <p className="muted small">Not opened yet. Open it once, on the selection you keep, to test it out of sample.</p>
        </aside>
      )}
    </section>
  )
}
