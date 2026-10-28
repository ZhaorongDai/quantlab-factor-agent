// How a line of YAML is split for colouring, run with `npm test`.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { parts } from './yaml.ts'

const kinds = (line: string) => parts(line).map(p => [p.kind, p.text])

test('a key and its value, by the kind of the value', () => {
  assert.deepEqual(kinds('  max_rounds: 30'), [['indent', '  '], ['key', 'max_rounds'], ['colon', ': '], ['number', '30']])
  assert.deepEqual(kinds('early_stopping: true'), [['key', 'early_stopping'], ['colon', ': '], ['bool', 'true']])
  assert.deepEqual(kinds('kwargs: null'), [['key', 'kwargs'], ['colon', ': '], ['null', 'null']])
  assert.deepEqual(kinds("start_date: '1900-01-01'"), [['key', 'start_date'], ['colon', ': '], ['string', "'1900-01-01'"]])
  assert.deepEqual(kinds('training: {start: 2016-10-11, end: 2021-12-31}'),
    [['key', 'training'], ['colon', ': '], ['flow', '{start: 2016-10-11, end: 2021-12-31}']])
})

test('a key that opens a section, a list item and a block scalar', () => {
  assert.deepEqual(kinds('segments:'), [['key', 'segments'], ['colon', ':']])
  assert.deepEqual(kinds('  - zarr_file_path: /data/a.zarr'),
    [['indent', '  '], ['dash', '- '], ['key', 'zarr_file_path'], ['colon', ': '], ['plain', '/data/a.zarr']])
  assert.deepEqual(kinds('- behavioural_bias: retail flow'),
    [['dash', '- '], ['key', 'behavioural_bias'], ['colon', ': '], ['plain', 'retail flow']])
  assert.deepEqual(kinds('description: |'), [['key', 'description'], ['colon', ': '], ['block', '|']])
})

test('a comment, alone or after a value, but not a # inside quotes or a word', () => {
  assert.deepEqual(kinds('# The data as the prompts describe it.'), [['comment', '# The data as the prompts describe it.']])
  assert.deepEqual(kinds('purge_bars: 5  # the label horizon'),
    [['key', 'purge_bars'], ['colon', ': '], ['number', '5  '], ['comment', '# the label horizon']])
  assert.deepEqual(kinds('name: "a # b"'), [['key', 'name'], ['colon', ': '], ['string', '"a # b"']])
  assert.deepEqual(kinds('url: http://x/y#z'), [['key', 'url'], ['colon', ': '], ['plain', 'http://x/y#z']])
})

test('a plain value with a colon inside is not a key, a time is no number', () => {
  assert.deepEqual(kinds('market: one: two'), [['key', 'market'], ['colon', ': '], ['plain', 'one: two']])
  assert.deepEqual(kinds('base_url: "https://api.deepseek.com"'),
    [['key', 'base_url'], ['colon', ': '], ['string', '"https://api.deepseek.com"']])
  assert.deepEqual(kinds('- 2016-10-11'), [['dash', '- '], ['plain', '2016-10-11']])
})
