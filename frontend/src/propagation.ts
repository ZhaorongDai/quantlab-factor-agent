// The propagation view's pure parts, run with `npm test`: one utility's axis for a trial of a round's Trial graph,
// and the rank ladder with its pick line.

type Values = { quality: number; fertility: number; novelty: number }
export type Utility = keyof Values
export type AxisNode = { trial: string; own: Values; neighbourhood: Values; propagated: Values; rank_propagated: number | null }

// Where a trial's marks sit on the axis of `utility`, `width` wide with `pad` at each end: its own value, its
// neighbours' average, its result after propagation and, as ticks, the results of `others`. Propagation puts the
// result the gamma fraction of the way from the own value to the neighbours'. The axis spans every mark and tick; one
// whose values are all the same puts them in its middle.
export function axisRow(node: AxisNode, others: AxisNode[], utility: Utility, width: number, pad = 10) {
  const own = node.own[utility], neighbours = node.neighbourhood[utility], result = node.propagated[utility]
  const ticks = others.map(n => n.propagated[utility])
  const values = [own, neighbours, result, ...ticks]
  const lo = Math.min(...values), hi = Math.max(...values)
  const x = (v: number) => (hi > lo ? pad + ((v - lo) / (hi - lo)) * (width - 2 * pad) : width / 2)
  return { own: x(own), neighbours: x(neighbours), result: x(result), ticks: ticks.map(x) }
}

// The rank after propagation of the round's last Picked parent, 0 when it has none: the pick line falls under it.
export function pickLine(nodes: AxisNode[], picked: string[]): number {
  const ranks = nodes.filter(n => picked.includes(n.trial)).map(n => n.rank_propagated ?? 0)
  return Math.max(0, ...ranks)
}

// The ladder's rows, at least `rows` and one past the pick line, and where the trial's ranks before and after
// propagation sit: on a row, or below the ladder with their number when they are past it; neither for a rank of none.
// Which way a trial's rank moved with propagation: up when it rose, down when it fell, '' when it stayed or is unknown.
export function rankMove(before: number | null, after: number | null): '' | 'up' | 'down' {
  return before == null || after == null || before === after ? '' : after < before ? 'up' : 'down'
}

export function ladder(before: number | null, after: number | null, line: number, rows = 15) {
  const shown = Math.max(rows, line + 1)
  const place = (rank: number | null) => ({
    row: rank != null && rank <= shown ? rank : null, below: rank != null && rank > shown ? rank : null,
  })
  return { rows: shown, before: place(before), after: place(after) }
}
