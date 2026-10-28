// The Acceptance calls the selection view writes, run with `npm test` (Node's own test runner, no browser).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { acceptanceCalls } from './calls.ts'

const QUIET = { label: 'quiet volume (accepted in round 1)', description: 'quiet volume' }
const INTRADAY = { label: 'intraday return (accepted in round 2)', description: 'intraday return' }

test('a selection is validated, opened on the hold-out once and exported, each factor named by its label', () => {
  assert.equal(acceptanceCalls([QUIET, INTRADAY], false), `from factor_agent.acceptance import Acceptance
from factor_agent.config import CampaignConfig

acceptance = Acceptance(CampaignConfig.from_file("<your campaign config>.yaml"))
selection = [
    "quiet volume (accepted in round 1)",
    "intraday return (accepted in round 2)",
]

# 1. Validate the selection: your model without and with it on the validation segment.
acceptance.validate(selection)

# 2. Open the hold-out once, on the selection you keep.
acceptance.open_holdout(selection)

# 3. Export it: each factor to a lowercase name for its module and store.
acceptance.export({
    "quiet volume (accepted in round 1)": "quiet_volume",
    "intraday return (accepted in round 2)": "intraday_return",
})
`)
})

test('once the hold-out was opened, the forced reopen is written commented out, saying it is no longer an untouched test', () => {
  const calls = acceptanceCalls([QUIET], true)
  assert.match(calls, /\n# 2\. The hold-out was already opened\. A forced reopen is no longer an untouched test: a selection judged on\n#    it now was chosen knowing an earlier result\. Uncomment the call only if you accept that\.\n# acceptance\.open_holdout\(selection, force=True\)\n/)
  // No line of the calls opens the hold-out when pasted as it is.
  assert.doesNotMatch(calls, /^acceptance\.open_holdout/m)
})

test('every export name is a distinct lowercase identifier that is not a Python keyword', () => {
  const calls = acceptanceCalls([
    { label: 'Quiet volume (accepted in round 1)', description: 'Quiet volume' },
    { label: 'quiet volume (accepted in round 3)', description: 'quiet volume' },
    { label: '20-day high (accepted in round 4)', description: '20-day high' },
    { label: 'class (accepted in round 5)', description: 'class' },
    { label: 'ρ ≥ 0 (accepted in round 6)', description: 'ρ ≥ 0' },
    { label: 'ρ ≥ 1 (accepted in round 7)', description: 'ρ ≥ 1' },
  ], false)
  const names = [...calls.matchAll(/^ {4}".*": "(.*)",$/gm)].map(m => m[1])
  assert.deepEqual(names, ['quiet_volume', 'quiet_volume_2', 'factor_20_day_high', 'class_factor', 'factor_0', 'factor_1'])
})

test('a label with quotes or backslashes is written as a valid Python string', () => {
  const calls = acceptanceCalls([{ label: 'the "gap" \\ open (accepted in round 1)', description: 'the "gap" \\ open' }], false)
  assert.match(calls, /^ {4}"the \\"gap\\" \\\\ open \(accepted in round 1\)",$/m)
})
