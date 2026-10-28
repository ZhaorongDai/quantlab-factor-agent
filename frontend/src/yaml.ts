// One line of YAML split into its parts for colouring, run with `npm test`. A line at a time: a block scalar's
// (| or >) continuation lines read as plain values, which is enough to read a config, a field or an operator file.

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
