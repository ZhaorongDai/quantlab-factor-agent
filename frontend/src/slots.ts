// The round plan's pure parts, run with `npm test`: its slots grouped by mechanism type and the group that holds a slot.

// A plan's slots by mechanism type, the types in the order they first appear, each group's slots in plan order.
export function groupSlots<S extends { mechanism_type: string }>(slots: S[]): { type: string; slots: S[] }[] {
  const groups = new Map<string, S[]>()
  for (const s of slots) {
    const group = groups.get(s.mechanism_type)
    if (group) group.push(s); else groups.set(s.mechanism_type, [s])
  }
  return [...groups].map(([type, slots]) => ({ type, slots }))
}

// The mechanism type of the first slot that `is` holds, or null for none.
export function groupOf<S extends { mechanism_type: string }>(slots: S[], is: (s: S) => boolean): string | null {
  return slots.find(is)?.mechanism_type ?? null
}
