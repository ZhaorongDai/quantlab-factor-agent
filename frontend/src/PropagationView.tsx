// The propagation view under the Lineage graph: why the Trial graph picked a round's parents. For the selected trial,
// one axis per utility with its own value, its neighbours' average and its result after propagation, the gamma
// fraction of the way between them, among the other candidates' results; beside it the rank ladder with the pick line.
// With nothing selected, the round's Picked parents with their rank before and after propagation.
import { useState } from 'react'
import type { GraphNode, RoundGraph } from './api'
import { axisRow, ladder, pickLine } from './propagation'
import type { Utility } from './propagation'

const UTILITIES: Utility[] = ['quality', 'fertility', 'novelty']
const REASONS = { expansion_cap: 'at the expansion cap', max_children: 'at the cap on children' }
const RELATIONS: Record<string, string> = { generation: 'parent or child', structure: 'shared structure', correlation: 'correlated' }
const fmt = (v: number) => v.toFixed(2)

function Rank({ from, to }: { from: number | null; to: number | null }) {
  const way = from == null || to == null || from === to ? '' : to < from ? ' up' : ' down'
  return <span className={`rank-move${way}`}>{from ?? '–'} → {to ?? '–'}</span>
}

function Row({ graph, node, utility, open, setOpen, pick }: {
  graph: RoundGraph; node: GraphNode; utility: Utility; open: boolean; setOpen: (open: boolean) => void
  pick: (trial: string) => void
}) {
  const W = 420
  const others = graph.nodes.filter(n => n.eligible && n.trial !== node.trial)
  const x = axisRow(node, others, utility, graph.gamma, W)
  return (
    <div className="mix">
      <span>{utility}</span>
      <svg viewBox={`0 0 ${W} 30`}>
        <line x1={10} x2={W - 10} y1={15} y2={15} className="axis" />
        {x.ticks.map((t, i) => <line key={others[i].trial} x1={t} x2={t} y1={11} y2={19} className="tick" />)}
        <line x1={x.own} x2={x.neighbours} y1={15} y2={15} className="span" />
        <circle cx={x.own} cy={15} r={5} className="own"><title>{`own ${fmt(node.own[utility])}`}</title></circle>
        <circle cx={x.neighbours} cy={15} r={6} className="neighbours" onMouseEnter={() => setOpen(true)}>
          <title>{`neighbours ${fmt(node.neighbourhood[utility])}`}</title>
        </circle>
        <rect x={x.result - 2} y={4} width={4} height={22} className="result">
          <title>{`after propagation ${fmt(node.propagated[utility])}`}</title>
        </rect>
      </svg>
      {open && node.heaviest.length > 0 && (
        <div className="heaviest">
          {node.heaviest.map(h => (
            <button key={`${h.trial} ${h.relation}`} onClick={() => pick(h.trial)}>
              <b className="mono">{h.trial}</b> <span>{RELATIONS[h.relation] ?? h.relation}</span> <span>{fmt(h.weight)}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

function Key() {
  const mark = (child: React.ReactNode) => <svg viewBox="0 0 22 22" width={22} height={22}>{child}</svg>
  return (
    <div className="mix-key">
      <span>{mark(<circle cx={11} cy={11} r={5} className="own" />)}own</span>
      <span>{mark(<circle cx={11} cy={11} r={6} className="neighbours" />)}neighbours</span>
      <span>{mark(<rect x={9} y={1} width={4} height={20} className="result" />)}after propagation</span>
      <span>{mark(<line x1={11} x2={11} y1={6} y2={16} className="tick" />)}other candidates</span>
    </div>
  )
}

// The ranks after propagation down to the pick line and past it, each with its trial; the trial's rank before
// propagation on the left joined to its rank after on the right, a rank past the ladder as its number below it.
function Ladder({ graph, node, pick }: { graph: RoundGraph; node: GraphNode; pick: (trial: string) => void }) {
  const line = pickLine(graph.nodes, graph.picked)
  const at = ladder(node.rank_own, node.rank_propagated, line)
  const byRank = new Map(graph.nodes.filter(n => n.rank_propagated != null).map(n => [n.rank_propagated!, n.trial]))
  const RH = 16, TOP = 18, W = 200, XB = 26, XA = W - 80
  const y = (row: number) => TOP + (row - 0.5) * RH
  const below = TOP + at.rows * RH + 14
  const H = below + 8
  const yOf = (p: { row: number | null; below: number | null }) => (p.row != null ? y(p.row) : p.below != null ? below - 12 : null)
  const yb = yOf(at.before), ya = yOf(at.after)
  const rose = node.rank_own != null && node.rank_propagated != null && node.rank_propagated < node.rank_own
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="ladder" style={{ maxWidth: W }}>
      <text x={XB} y={10} textAnchor="middle" className="label">before</text>
      <text x={XA - 32} y={10} textAnchor="middle" className="label">after</text>
      {Array.from({ length: at.rows }, (_, i) => i + 1).map(r => {
        const trial = byRank.get(r)
        return (
          <g key={r}>
            <text x={XA - 12} y={y(r) + 3} textAnchor="end" className="rank">{r}</text>
            {trial && (
              <text x={XA + 2} y={y(r) + 3} onClick={() => pick(trial)}
                    className={`rank-id${graph.picked.includes(trial) ? ' picked' : ''}${trial === node.trial ? ' self' : ''}`}>
                {trial}
              </text>
            )}
          </g>
        )
      })}
      {line > 0 && <line x1={4} x2={W - 4} y1={TOP + line * RH} y2={TOP + line * RH} className="pick-line" />}
      {yb != null && ya != null && <line x1={XB + 6} y1={yb} x2={XA - 38} y2={ya} className={`rank-arrow${rose ? ' up' : ' down'}`} />}
      {yb != null && <circle cx={XB} cy={yb} r={5} className="rank-dot" />}
      {at.before.below != null && <text x={XB} y={below + 4} textAnchor="middle" className="rank">{at.before.below}</text>}
      {ya != null && <circle cx={XA - 32} cy={ya} r={5} className={`rank-dot after${graph.picked.includes(node.trial) ? ' picked' : ''}`} />}
      {at.after.below != null && <text x={XA - 32} y={below + 4} textAnchor="middle" className="rank">{at.after.below}</text>}
    </svg>
  )
}

export function Propagation({ graph, selected, pick, close }: {
  graph: RoundGraph; selected: string | null; pick: (trial: string) => void; close: () => void
}) {
  const [open, setOpen] = useState<Utility | null>(null)
  const node = graph.nodes.find(n => n.trial === selected)
  return (
    <section className="propagation">
      <button className="link close small" onClick={close}>close</button>
      {node ? <>
        <h4><span className="mono">{node.trial}</span>{node.reason && <span className="muted"> {REASONS[node.reason]}</span>}</h4>
        <div className="propagation-row">
          <div className="mixes" onMouseLeave={() => setOpen(null)}>
            {UTILITIES.map(u => (
              <Row key={u} graph={graph} node={node} utility={u} open={open === u} pick={pick}
                   setOpen={o => setOpen(o ? u : null)} />
            ))}
            <Key />
          </div>
          {node.eligible && <Ladder graph={graph} node={node} pick={pick} />}
        </div>
      </> : <>
        <h4>Picked parents</h4>
        <div className="picked-list">
          {graph.picked.map(trial => {
            const n = graph.nodes.find(n => n.trial === trial)
            return (
              <button key={trial} onClick={() => pick(trial)}>
                <b className="mono">{trial}</b> <Rank from={n?.rank_own ?? null} to={n?.rank_propagated ?? null} />
              </button>
            )
          })}
        </div>
      </>}
    </section>
  )
}
