// The Acceptance calls the selection view writes for the ticked factors, as Python to paste; the page runs none of them.
// A factor is named by its label (`description (accepted in round n)`), which Acceptance accepts wherever it takes a name.

export type Named = { label: string; description: string }

// A Python string literal: a JSON string is one (its escapes are Python's too).
const literal = (text: string) => JSON.stringify(text)

const KEYWORDS = new Set(('False None True and as assert async await break class continue def del elif else except ' +
  'finally for from global if import in is lambda nonlocal not or pass raise return try while with yield').split(' '))

// The names export() is given, one per factor: the description's first words as a lowercase identifier that is
// not a Python keyword, numbered from 2 when two factors' words agree.
function exportNames(selection: Named[]): string[] {
  const taken = new Set<string>()
  return selection.map(n => {
    let name = n.description.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean).slice(0, 4).join('_')
    if (!name || /^[0-9]/.test(name)) name = name ? `factor_${name}` : 'factor'
    if (KEYWORDS.has(name)) name = `${name}_factor`
    let unique = name
    for (let k = 2; taken.has(unique); k++) unique = `${name}_${k}`
    taken.add(unique)
    return unique
  })
}

export function acceptanceCalls(selection: Named[], holdoutOpened: boolean): string {
  const names = exportNames(selection)
  const lines = (f: (n: Named, k: number) => string) => selection.map((n, k) => `    ${f(n, k)},\n`).join('')
  const holdout = holdoutOpened
    ? `# 2. The hold-out was opened before: it is no longer untouched, and a selection judged on it now was chosen
#    knowing an earlier result. Opening it again needs force.
acceptance.open_holdout(selection, force=True)`
    : `# 2. Open the hold-out once, on the selection you keep.
acceptance.open_holdout(selection)`
  return `from factor_agent.acceptance import Acceptance
from factor_agent.config import CampaignConfig

acceptance = Acceptance(CampaignConfig.from_file("<your campaign config>.yaml"))
selection = [
${lines(n => literal(n.label))}]

# 1. Validate the selection: your model without and with it on the validation segment.
acceptance.validate(selection)

${holdout}

# 3. Export it: each factor to a lowercase name for its module and store.
acceptance.export({
${lines((n, k) => `${literal(n.label)}: ${literal(names[k])}`)}})
`
}
