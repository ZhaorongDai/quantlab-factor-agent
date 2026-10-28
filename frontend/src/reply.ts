// An LLM reply as the call reader shows it: the JSON in it laid out to read, the rest as it came.

const FENCED = /(```(?:json)?\n)([\s\S]*?)(\n```)/g

// `text` laid out with two-space indents when it is JSON, `null` when it is not.
function indented(text: string): string | null {
  try {
    return JSON.stringify(JSON.parse(text), null, 2)
  } catch {
    return null
  }
}

// The reply with each fenced JSON block, or the whole reply when it is JSON alone, laid out to read. A block that is
// not JSON (a reply cut off at the output limit, say) is left as it came.
export function formatted(reply: string): string {
  const whole = indented(reply)
  if (whole != null) return whole
  return reply.replace(FENCED, (block, open: string, body: string, close: string) => {
    const laid = indented(body)
    return laid == null ? block : `${open}${laid}${close}`
  })
}
