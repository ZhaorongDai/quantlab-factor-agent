// The rows of one role's lane on the round's timeline, run with `npm test`.

export type Span<T> = { from: number; to: number; item: T }

// Calls that overlap, as a round's proposal groups ask side by side, go on rows of their own so that each can be
// seen and clicked: every span takes the first row it fits on, a span `least` wide at the least (in the same unit).
export function rows<T>(spans: Span<T>[], least: number): T[][] {
  const out: { end: number; items: T[] }[] = []
  for (const s of [...spans].sort((a, b) => a.from - b.from)) {
    const end = Math.max(s.to, s.from + least)
    const row = out.find(r => r.end <= s.from)
    if (row) { row.end = end; row.items.push(s.item) } else out.push({ end, items: [s.item] })
  }
  return out.map(r => r.items)
}
