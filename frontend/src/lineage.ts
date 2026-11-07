// The Lineage graph's pure parts, run with `npm test`: trial ids, the column layout, a trial's family and the rounds
// the stepper steps through.

// A trial as the graph lays it out: its id (T<round>-<slot from 1, two digits>), its round and its parent's id.
export type LineageNode = { id: string; round: number; parent: string }

export const idOf = (round: number, slot: number) => `T${round}-${String(slot + 1).padStart(2, '0')}`

// The page's key of a trial ("round.slot", the slot from 0) from its id; null for a text that names no trial.
export function keyOf(id: string): string | null {
  const found = /^T(\d+)-(\d+)$/.exec(id)
  return found ? `${found[1]}.${Number(found[2]) - 1}` : null
}

const slotOf = (id: string) => Number(id.slice(id.indexOf('-') + 1))

// Each node's [column, row]: one column per round of `rounds`, in their order. A column's trials sit in the order of
// their parents' rows, so that a parent's children sit together across from it; a trial without a parent in the
// graph comes after them; ties keep slot order. A trial of a round not in `rounds` is left out.
export function layout(nodes: LineageNode[], rounds: number[]): Map<string, [number, number]> {
  const at = new Map<string, [number, number]>()
  rounds.forEach((round, column) => {
    const rowOf = (n: LineageNode) => at.get(n.parent)?.[1] ?? Infinity
    nodes.filter(n => n.round === round)
      .sort((a, b) => rowOf(a) - rowOf(b) || slotOf(a.id) - slotOf(b.id))
      .forEach((n, row) => at.set(n.id, [column, row]))
  })
  return at
}

// A trial's family: itself, its ancestors and every trial built on it, down the generations; empty for none.
export function family(nodes: LineageNode[], id: string | null): Set<string> {
  const out = new Set<string>()
  if (!id) return out
  const parentOf = new Map(nodes.map(n => [n.id, n.parent]))
  for (let at: string | undefined = id; at && !out.has(at); at = parentOf.get(at)) out.add(at)
  const down = [id]
  while (down.length) {
    const parent = down.pop()!
    for (const n of nodes) if (n.parent === parent && !out.has(n.id)) { out.add(n.id); down.push(n.id) }
  }
  return out
}

// The rounds the stepper steps through: the written ones, then the round in progress once its plan is made.
export function steppedRounds(written: number, running: { round: number; planned: boolean } | null): number[] {
  const rounds = Array.from({ length: written }, (_, i) => i + 1)
  return running?.planned ? [...rounds, running.round] : rounds
}
