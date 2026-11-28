import { test } from 'node:test'
import assert from 'node:assert/strict'
import { normalizeTools, buildToolPreamble, buildToolInstructions, createToolCallParser, slimSchema } from '../lib/tools.mjs'
import { byteLen } from '../lib/translate.mjs'

test('normalizeTools keeps only named function tools', () => {
  const tools = normalizeTools([
    { type: 'function', function: { name: 'get_weather', description: 'w', parameters: { type: 'object' } } },
    { type: 'function', function: { name: 'no_params' } },
    { type: 'bogus' },
    null,
  ])
  assert.equal(tools.length, 2)
  assert.equal(tools[0].name, 'get_weather')
  assert.equal(tools[1].parameters.type, 'object') // defaulted
})

test('the catalog preserves specs and the instructions preserve forced choice', () => {
  const tools = normalizeTools([{ type: 'function', function: { name: 'f', description: 'd', parameters: { type: 'object', properties: { x: { type: 'string' } } } } }])
  const p = buildToolPreamble(tools, 'required')
  assert.ok(p.includes('- f: d'))
  assert.ok(buildToolInstructions(true, true).includes('requires a client tool call'))
  assert.equal(buildToolPreamble([], 'auto'), '')
  const named = buildToolPreamble(tools, { type: 'function', function: { name: 'f' } })
  assert.ok(named.includes('- f: d'))
  assert.ok(buildToolInstructions('f', false).includes('requires the client tool "f"'))
})

test('slimSchema strips annotation keywords, keeps structure', () => {
  const slim = slimSchema({
    type: 'object',
    description: 'verbose docs',
    title: 'T',
    required: ['a'],
    properties: {
      a: { type: 'string', description: 'field doc', enum: ['x', 'y'] },
      b: { type: 'object', properties: { c: { type: 'number', examples: [1] } } },
    },
  })
  assert.equal(slim.description, undefined)
  assert.equal(slim.title, undefined)
  assert.deepEqual(slim.required, ['a'])
  assert.equal(slim.properties.a.description, undefined)
  assert.deepEqual(slim.properties.a.enum, ['x', 'y'])
  assert.equal(slim.properties.b.properties.c.examples, undefined)
})

test('preamble compacts schema annotations and caps giant schemas', () => {
  const bigProps = {}
  for (let i = 0; i < 200; i++) {
    bigProps[`field_${i}`] = { type: 'string', description: 'd'.repeat(80) }
  }
  const tools = normalizeTools([
    { type: 'function', function: { name: 'small', description: 's', parameters: { type: 'object', properties: { x: { type: 'string', description: 'doc' } } } } },
    { type: 'function', function: { name: 'huge', description: 'h', parameters: { type: 'object', properties: bigProps, required: ['field_0'] } } },
  ])
  const p = buildToolPreamble(tools, 'auto')
  // small tool keeps its (slimmed) schema incl. type info
  assert.ok(p.includes('"x":{"type":"string"}'))
  assert.ok(!p.includes('"doc"')) // property description stripped
  // huge tool degrades to bare object + required
  assert.ok(p.includes('"properties":{}'))
  assert.ok(p.includes('"required":["field_0"]'))
  assert.ok(!p.includes('field_1'))
})

test('preamble falls back to name+description when toolset is enormous', () => {
  const tools = normalizeTools(Array.from({ length: 60 }, (_, i) => ({
    type: 'function',
    function: {
      name: `tool_${i}`,
      description: `description ${i} `.repeat(30),
      parameters: { type: 'object', properties: Object.fromEntries(Array.from({ length: 15 }, (_, j) => [`p${j}`, { type: 'string', description: 'x'.repeat(60) }])) },
    },
  })))
  const p = buildToolPreamble(tools, 'auto')
  assert.ok(byteLen(p) < 45 * 1024)
  assert.ok(p.includes('parameter schemas omitted'))
  assert.ok(p.includes('- tool_0:'))
  assert.ok(!p.includes('"properties"'))
})

test('compact preambles retain file delivery and tool search parameter schemas', () => {
  const tools = normalizeTools([
    {
      name: 'SendUserFile',
      input_schema: {
        type: 'object',
        properties: {
          files: { type: 'array', items: { type: 'string' } },
          status: { type: 'string', enum: ['normal', 'proactive'] },
          display: { type: 'string', enum: ['render', 'attach'] },
        },
        required: ['files', 'status'],
      },
    },
    {
      name: 'ToolSearch',
      input_schema: {
        type: 'object',
        properties: { query: { type: 'string' }, max_results: { type: 'number' } },
        required: ['query', 'max_results'],
      },
    },
    ...Array.from({ length: 36 }, (_, index) => ({
      name: `tool_${index}`,
      input_schema: {
        type: 'object',
        properties: Object.fromEntries(Array.from({ length: 45 }, (_, field) => [`generic_field_${field}`, { type: 'string' }])),
      },
    })),
  ])

  const automatic = buildToolPreamble(tools, 'auto')
  assert.ok(automatic.includes('parameter schemas omitted'))
  for (const preamble of [
    automatic,
    buildToolPreamble(tools, 'auto', { lite: true }),
    buildToolPreamble(tools, 'auto', { namesOnly: true }),
    buildToolPreamble(tools, 'auto', { reminder: true }),
  ]) {
    const schemaFor = (name) => {
      const prefix = `- ${name} parameters: `
      const line = preamble.split('\n').find(line => line.startsWith(prefix))
      assert.ok(line, `Expected compact parameters for ${name}`)
      return JSON.parse(line.slice(prefix.length))
    }
    const delivery = schemaFor('SendUserFile')
    assert.deepEqual(delivery.required, ['files', 'status'])
    assert.deepEqual(delivery.properties.files, { type: 'array', items: { type: 'string' } })
    assert.deepEqual(delivery.properties.status.enum, ['normal', 'proactive'])
    assert.deepEqual(delivery.properties.display.enum, ['render', 'attach'])
    const search = schemaFor('ToolSearch')
    assert.deepEqual(search.required, ['query', 'max_results'])
    assert.equal(search.properties.max_results.type, 'number')
    assert.ok(!preamble.includes('generic_field_0'))
  }

  const forced = buildToolPreamble(tools, { type: 'tool', name: 'ToolSearch' }, { reminder: true })
  assert.ok(forced.includes('- ToolSearch parameters: '))
  assert.ok(!forced.includes('SendUserFile'))
})

test('parser extracts blocks split across deltas', () => {
  const p = createToolCallParser()
  let r = p.push('Hello <tool_call>{"name":"get_weather","arg')
  assert.equal(r.text, 'Hello ')
  assert.equal(r.calls.length, 0)
  r = p.push('uments":{"city":"Paris"}}</tool_call> done')
  assert.equal(r.text, ' done')
  assert.equal(r.calls.length, 1)
  assert.equal(r.calls[0].name, 'get_weather')
  assert.deepEqual(r.calls[0].arguments, { city: 'Paris' })
  const end = p.end()
  assert.equal(end.text, '')
  assert.equal(end.dropped, undefined)
})

test('parser handles multiple blocks and surrounding text', () => {
  const p = createToolCallParser()
  const r = p.push('pre <tool_call>{"name":"a","arguments":{}}</tool_call> mid <tool_call>{"name":"b","arguments":{"x":1}}</tool_call> post')
  assert.equal(r.text, 'pre  mid  post')
  assert.deepEqual(r.calls.map((c) => c.name), ['a', 'b'])
})

test('parser holds back tag split mid-token across deltas', () => {
  // Live Quick streams "<" then "tool_call>{...}</tool_call>" as separate deltas.
  const p = createToolCallParser()
  let r = p.push('pre<')
  assert.equal(r.text, 'pre') // '<' held, not flushed
  assert.equal(r.calls.length, 0)
  r = p.push('tool_call>{"name":"f","arguments":{}}</tool_call>post')
  assert.equal(r.text, 'post')
  assert.equal(r.calls.length, 1)
  assert.equal(r.calls[0].name, 'f')
})

test('parser holds back split close tag', () => {
  const p = createToolCallParser()
  p.push('<tool_call>{"name":"f","arguments":{}}</tool_')
  const r = p.push('call>tail')
  assert.equal(r.text, 'tail')
  assert.equal(r.calls.length, 1)
})

test('parser preserves backticked mentions and examples across deltas', () => {
  const call = '<tool_call>{"name":"f","arguments":{}}</tool_call>'
  const prose = 'Mention `<tool_call>` here.\n```xml\n' + call + '\n```\n'
  const parser = createToolCallParser()
  let visible = ''
  for (const char of prose + call) visible += parser.push(char).text
  const tail = parser.end()
  assert.equal(visible + tail.text, prose)
  assert.equal(tail.dropped, undefined)
  assert.deepEqual(parser.calls, [{ name: 'f', arguments: {} }])
})

test('parser drops unterminated block without leaking it', () => {
  const p = createToolCallParser()
  p.push('text <tool_call>{"name":"x"')
  const end = p.end()
  assert.equal(end.text, '')
  assert.equal(end.dropped, true)
})

test('parser tolerates arguments as JSON string and missing args', () => {
  const p = createToolCallParser()
  const r = p.push('<tool_call>{"name":"f","arguments":"{\\"a\\":1}"}</tool_call><tool_call>{"name":"g"}</tool_call>')
  assert.equal(r.calls.length, 2)
  assert.deepEqual(r.calls[0].arguments, { a: 1 })
  assert.deepEqual(r.calls[1].arguments, {})
})

test('parser ignores malformed JSON blocks', () => {
  const p = createToolCallParser()
  const r = p.push('<tool_call>not json</tool_call>ok')
  assert.equal(r.calls.length, 0)
  assert.equal(r.text, 'ok')
  const end = p.end()
  assert.equal(end.dropped, true) // malformed block counts as dropped
})
