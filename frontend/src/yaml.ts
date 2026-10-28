// YAML split into its parts for colouring, run with `npm test`: a line by its own parts (`parts`), or a whole text
// (`highlight`), where a flow list or map carried over several lines and a block scalar's lines keep their kind.

export type Kind = 'indent' | 'dash' | 'key' | 'colon' | 'string' | 'number' | 'bool' | 'null' | 'flow' | 'block' | 'plain' | 'comment'
export type Part = { kind: Kind; text: string }

// Where a comment starts: a # at the line's start or after a space, outside quotes.
function commentAt(line: string): number {
  let quote = ''
  for (let i = 0; i < line.length; i++) {
    const c = line[i]
    if (quote) { if (c === quote) quote = ''; continue }
    if (c === '"' || c === "'") quote = c
    else if (c === '#' && (i === 0 || /\s/.test(line[i - 1]))) return i
  }
  return -1
}

function value(text: string): Kind {
  const t = text.trim()
  if (/^(true|false)$/i.test(t)) return 'bool'
  if (/^(null|~)$/.test(t)) return 'null'
  if (/^[-+]?(\d[\d_]*(\.\d*)?([eE][-+]?\d+)?|\.\d+([eE][-+]?\d+)?|\.inf|\.nan)$/i.test(t)) return 'number'
  if (/^["']/.test(t)) return 'string'
  if (/^[[{]/.test(t)) return 'flow'
  if (/^[|>][-+]?$/.test(t)) return 'block'
  return 'plain'
}

export function parts(line: string): Part[] {
  const out: Part[] = []
  const at = commentAt(line)
  let rest = at >= 0 ? line.slice(0, at) : line
  const comment = at >= 0 ? line.slice(at) : ''
  const indent = rest.match(/^\s*/)![0]
  if (indent) out.push({ kind: 'indent', text: indent })
  rest = rest.slice(indent.length)
  const dash = rest.match(/^-(\s+|$)/)
  if (dash) { out.push({ kind: 'dash', text: dash[0] }); rest = rest.slice(dash[0].length) }
  const key = rest.match(/^("[^"]*"|'[^']*'|[^\s"'{[#][^:#]*?)(:)(\s+|$)/)
  if (key) {
    out.push({ kind: 'key', text: key[1] }, { kind: 'colon', text: key[2] + key[3] })
    rest = rest.slice(key[0].length)
  }
  if (rest) out.push({ kind: value(rest), text: rest })
  if (comment) out.push({ kind: 'comment', text: comment })
  return out
}

// How far ``text`` opens (+) or closes (-) flow collections, brackets inside quotes left out.
function depth(text: string): number {
  let d = 0, quote = ''
  for (const c of text) {
    if (quote) { if (c === quote) quote = ''; continue }
    if (c === '"' || c === "'") quote = c
    else if (c === '[' || c === '{') d++
    else if (c === ']' || c === '}') d--
  }
  return d
}

const indentOf = (line: string) => line.match(/^\s*/)![0].length

export function highlight(text: string): Part[][] {
  const out: Part[][] = []
  let open = 0             // flow collections still open from the lines before
  let block = -1           // the indentation a block scalar's lines must exceed, -1 outside one
  for (const line of text.split('\n')) {
    const indent = line.match(/^\s*/)![0]
    if (block >= 0 && (!line.trim() || indentOf(line) > block)) {
      out.push(line.trim() ? [{ kind: 'indent', text: indent }, { kind: 'string', text: line.slice(indent.length) }] : [])
      continue
    }
    block = -1
    if (open > 0) {
      const at = commentAt(line)
      const body = at >= 0 ? line.slice(indent.length, at) : line.slice(indent.length)
      out.push([
        ...(indent ? [{ kind: 'indent' as const, text: indent }] : []), { kind: 'flow', text: body },
        ...(at >= 0 ? [{ kind: 'comment' as const, text: line.slice(at) }] : []),
      ])
      open = Math.max(0, open + depth(body))
      continue
    }
    const own = parts(line)
    out.push(own)
    const values = own.filter(p => p.kind !== 'comment')
    const last = values[values.length - 1]
    if (last?.kind === 'flow') open = Math.max(0, depth(last.text))
    if (last?.kind === 'block') block = indentOf(line)
  }
  return out
}

// The lines each line folds, as start line -> end line (exclusive), the start kept in view: a key or a list item
// folds the lines indented deeper under it, and a key ending in a colon also a list written at its own indentation.
export function folds(text: string): Map<number, number> {
  const lines = text.split('\n')
  const out = new Map<number, number>()
  lines.forEach((line, i) => {
    const own = parts(line)
    if (!own.some(p => p.kind === 'key' || p.kind === 'dash')) return
    const indent = indentOf(line)
    const opens = own[own.length - 1]?.kind === 'colon'
    let last = i
    for (let j = i + 1; j < lines.length; j++) {
      if (!lines[j].trim()) continue
      const deeper = indentOf(lines[j]) > indent
      const listUnder = opens && indentOf(lines[j]) === indent && /^\s*-(\s|$)/.test(lines[j])
      if (!deeper && !listUnder) break
      last = j
    }
    if (last > i) out.set(i, last + 1)
  })
  return out
}

// The lines of a text of ``count`` lines left in view when the lines ``folded`` start (by ``folds``) are folded.
export function visible(count: number, ranges: Map<number, number>, folded: Set<number>): number[] {
  const out: number[] = []
  for (let line = 0; line < count; line = folded.has(line) ? ranges.get(line) ?? line + 1 : line + 1) out.push(line)
  return out
}
