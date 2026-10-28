// How the call reader lays out an LLM reply, run with `npm test`.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { formatted } from './reply.ts'

test('the JSON in a fenced block is laid out to read and the words around it are kept', () => {
  const reply = 'Here it is.\n```json\n[{"number": 1, "consistent": true}]\n```\nDone.'
  assert.equal(formatted(reply), 'Here it is.\n```json\n[\n  {\n    "number": 1,\n    "consistent": true\n  }\n]\n```\nDone.')
})

test('a reply that is JSON alone is laid out to read', () => {
  assert.equal(formatted('{"a":[1,2]}'), '{\n  "a": [\n    1,\n    2\n  ]\n}')
})

test('a reply cut off before its JSON closes, or without JSON, is shown as it came', () => {
  const cut = 'Here it is.\n```json\n[{"number": 1, "consis'
  assert.equal(formatted(cut), cut)
  assert.equal(formatted('```json\n[{"number": 1,\n```'), '```json\n[{"number": 1,\n```')
  assert.equal(formatted('I cannot answer that.'), 'I cannot answer that.')
})
