// OpenAI tools ⇄ Quick text convention.
//
// Quick's bidi protocol has no client-tool channel: its agent runtime owns the
// tool loop and executes ITS tools server-side (verified live — tool_start/
// tool_arguments/tool_result events, message_complete.tool_calls stays empty).
// But the underlying model will happily emit a caller-defined convention as
// plain text when the instruction is strong and inline (verified: a MANDATORY
// block in the user text produced exactly `<tool_call>{json}</tool_call>`).
//
// So we inject tool specs + rules into the message text, parse
// <tool_call>{"name":…,"arguments":{…}}</tool_call> blocks out of the streamed
// text, and surface them as OpenAI tool_calls. Results come back as role:'tool'
// messages, rendered into the next turn's text by renderDelta.

import { byteLen } from './translate.mjs'

/** OpenAI tools[] → compact spec list (function tools only). */
export function normalizeTools(tools) {
  if (!Array.isArray(tools)) return []
  const out = []
  for (const t of tools) {
    const fn = t?.function ?? t
    if (!fn || typeof fn.name !== 'string' || !fn.name) continue
    // OpenAI uses `parameters`; Anthropic-style shims may send `input_schema`.
    const schema = fn.parameters ?? fn.input_schema
    out.push({
      name: fn.name,
      description: typeof fn.description === 'string' ? fn.description : '',
      parameters: schema && typeof schema === 'object' ? schema : { type: 'object', properties: {} },
    })
  }
  return out
}

export function requiredToolChoice(choice) {
  return choice?.function?.name ?? (choice?.type === 'tool' ? choice.name : undefined)
    ?? (choice === 'required' || choice?.type === 'any')
}

export function canonicalJson(value) {
  return JSON.stringify(value, (_key, entry) => {
    if (entry === null || typeof entry !== 'object' || Array.isArray(entry)) return entry
    return Object.fromEntries(Object.keys(entry).sort().map((name) => [name, entry[name]]))
  })
}

export function toolArguments(value) {
  if (typeof value === 'string') {
    try { value = JSON.parse(value) } catch { return undefined }
  }
  return value && typeof value === 'object' && !Array.isArray(value) ? value : undefined
}

function toolSpecLine(t) {
  const params = JSON.stringify(boundedParameters(t))
  return `- ${t.name}: ${t.description || '(no description)'}\n  parameters: ${params}`
}

// JSON-Schema annotation keywords that cost bytes but carry no structure.
const SCHEMA_ANNOTATIONS = new Set([
  'description', 'title', 'examples', 'example', 'default', '$comment',
  'deprecated', 'readOnly', 'writeOnly', 'markdownDescription',
])
const SCHEMA_NESTED = new Set(['properties', 'patternProperties', '$defs', 'definitions'])
const SCHEMA_SINGLE = new Set(['items', 'additionalProperties', 'contains', 'propertyNames', 'if', 'then', 'else', 'not'])
const SCHEMA_ARRAY = new Set(['anyOf', 'oneOf', 'allOf', 'prefixItems'])

/** Strip annotation keywords recursively; keeps types/required/enum/structure. */
export function slimSchema(schema, depth = 0) {
  if (!schema || typeof schema !== 'object' || Array.isArray(schema) || depth > 8) return schema
  const out = {}
  for (const [k, v] of Object.entries(schema)) {
    if (SCHEMA_ANNOTATIONS.has(k)) continue
    if (SCHEMA_NESTED.has(k) && v && typeof v === 'object') {
      out[k] = Object.fromEntries(Object.entries(v).map(([pk, pv]) => [pk, slimSchema(pv, depth + 1)]))
    } else if (SCHEMA_SINGLE.has(k) && v && typeof v === 'object') {
      out[k] = slimSchema(v, depth + 1)
    } else if (SCHEMA_ARRAY.has(k) && Array.isArray(v)) {
      out[k] = v.map((x) => slimSchema(x, depth + 1))
    } else {
      out[k] = v
    }
  }
  return out
}

/** Per-tool schema cap: beyond this, degrade to a bare object schema. */
const SCHEMA_CAP = 3072

function boundedParameters(t) {
  const slim = slimSchema(t.parameters)
  const json = JSON.stringify(slim)
  if (json.length <= SCHEMA_CAP) return slim
  // Still too big: keep only the top-level shape so the model knows it takes an object.
  const required = Array.isArray(t.parameters?.required) ? t.parameters.required : []
  return { type: 'object', properties: {}, ...(required.length ? { required } : {}) }
}

/**
 * The instruction block prepended to the text we send upstream.
 * @param {Array} tools normalized tools (may be empty → returns '')
 * @param {string|object} [toolChoice] body.tool_choice
 */
// Quick rejects user_message.text over 64 KB; a huge toolset alone can eat
// most of that, so the preamble degrades to name+description when needed.
const PREAMBLE_CAP = 40 * 1024
const COMPACT_SCHEMA_TOOLS = new Set(['SendUserFile', 'ToolSearch'])

export function buildToolPreamble(tools, toolChoice, { lite = false, namesOnly = false, reminder = false, parallel = true } = {}) {
  if (!tools.length) return ''
  const forced = requiredToolChoice(toolChoice)
  const only = typeof forced === 'string' ? tools.filter((t) => t.name === forced) : tools
  const shown = only.length ? only : tools
  const retainedSchemas = shown
    .filter(tool => COMPACT_SCHEMA_TOOLS.has(tool.name))
    .map(tool => `- ${tool.name} parameters: ${JSON.stringify(boundedParameters(tool))}`)
  const compactSchemas = retainedSchemas.length ? ['Selected client tool parameter schemas:', ...retainedSchemas].join('\n') : ''
  const must = forced
    ? only.length === 1 && typeof forced === 'string'
      ? `This request requires the client tool "${forced}" — reply with only its <tool_call> block.`
      : 'This request requires a client tool — reply with <tool_call> block(s) only.'
    : ''
  // Reused threads already carry the full spec in Quick's server-side
  // history, and everything sent in `text` stays there: re-sending a 30 KB
  // spec every turn grew the reported context ~10k tokens per request and
  // made Claude Code auto-compact after a couple of questions. Keep only a
  // short reminder so the convention stays salient.
  if (reminder) {
    return [
      '[Client tool calling — reminder]',
      `The client tools described earlier in this conversation are still available: ${shown.slice(0, 80).map((t) => t.name).join(', ')}.`,
      'To call one, reply with ONLY <tool_call>{"name":"<tool_name>","arguments":{<json arguments>}}</tool_call> block(s), no other text; after a [tool_result …] message arrives, continue in plain text.',
      parallel ? '' : 'Emit at most one client tool call per step.',
      compactSchemas,
      must,
    ].filter(Boolean).join('\n')
  }
  const assemble = (specs, header, schemas) => [
    '[Client tool calling]',
    'The application you are assisting exposes the tools below; it executes them on the user\'s machine and returns each result as a [tool_result …] message.',
    'This channel has no native function-calling API — the <tool_call> marker below is the only way to invoke these tools.',
    'When the task needs one of them, reply with ONLY a block in this format (no other text):',
    '<tool_call>{"name":"<tool_name>","arguments":{<json arguments>}}</tool_call>',
    'Guidelines:',
    parallel ? '- one block per call; several blocks are allowed for parallel calls;' : '- at most one tool call block per step;',
    '- "arguments" is a JSON object matching the parameters schema;',
    '- after a [tool_result …] message arrives, continue the task in plain text;',
    '- earlier [tool_result …] blocks may be truncated or omitted; when the task needs current file or command output, call the tools again instead of guessing from memory;',
    must,
    header,
    specs,
    schemas,
  ].filter(Boolean).join('\n')
  // Lite/names tiers cap each description — verbose tool docs otherwise keep
  // the "lite" preamble in the tens of KB and it still gets shed downstream.
  const desc = (t) => String(t.description || '(no description)').slice(0, 200)
  const liteSpecs = shown.map((t) => `- ${t.name}: ${desc(t)}`).join('\n')
  const namesSpecs = shown.map((t) => `- ${t.name}`).join('\n')
  const liteHeader = 'Tools provided by the client (some parameter schemas omitted — follow any schemas shown):'
  const namesHeader = 'Tools provided by the client (names and selected parameter schemas):'
  let out = namesOnly
    ? assemble(namesSpecs, namesHeader, compactSchemas)
    : lite
      ? assemble(liteSpecs, liteHeader, compactSchemas)
      : assemble(shown.map(toolSpecLine).join('\n'), 'Tools provided by the client:')
  if (!lite && !namesOnly && byteLen(out) > PREAMBLE_CAP) {
    // 20+ Claude-Code-sized schemas can exceed the 64 KB message cap on their
    // own — keep names + descriptions and selected parameter schemas.
    out = assemble(liteSpecs, liteHeader, compactSchemas)
  }
  return out
}

const OPEN_TAG = '<tool_call>'
const CLOSE_TAG = '</tool_call>'
const OPEN_RE = /<tool_call>/g
const CLOSE_RE = /<\/tool_call>/g

/** Longest suffix of `buf` that is a proper prefix of `tag` (0 if none). */
function heldPrefixLen(buf, tag) {
  const max = Math.min(buf.length, tag.length - 1)
  for (let k = max; k >= 1; k--) {
    if (buf.endsWith(tag.slice(0, k))) return k
  }
  return 0
}

/**
 * Incremental parser: feed text_delta strings, get back visible text and
 * completed tool calls. Blocks split across deltas are handled — including a
 * tag split mid-token (`<` + `tool_call>…`), which is held back rather than
 * flushed. A dangling <tool_call> at end() is reported via `dropped`.
 */
export function createToolCallParser() {
  let buf = ''
  let inBlock = false
  let blockJson = ''
  let dropped = false
  const pending = [] // parsed calls waiting for id assignment

  const drain = () => {
    const text = []
    const calls = []
    for (;;) {
      if (!inBlock) {
        OPEN_RE.lastIndex = 0
        const m = OPEN_RE.exec(buf)
        if (!m) {
          // Hold back a suffix that could be the start of <tool_call>.
          const hold = heldPrefixLen(buf, OPEN_TAG)
          text.push(buf.slice(0, buf.length - hold))
          buf = buf.slice(buf.length - hold)
          break
        }
        text.push(buf.slice(0, m.index))
        buf = buf.slice(m.index + m[0].length)
        inBlock = true
        blockJson = ''
        continue
      }
      CLOSE_RE.lastIndex = 0
      const m = CLOSE_RE.exec(buf)
      if (!m) {
        // Same holdback for a split </tool_call> — keep it out of blockJson.
        const hold = heldPrefixLen(buf, CLOSE_TAG)
        blockJson += buf.slice(0, buf.length - hold)
        buf = buf.slice(buf.length - hold)
        break
      }
      blockJson += buf.slice(0, m.index)
      buf = buf.slice(m.index + m[0].length)
      inBlock = false
      const call = parseCall(blockJson)
      if (call) calls.push(call)
      else dropped = true
      blockJson = ''
    }
    return { text: text.join(''), calls }
  }

  return {
    /** Feed a delta; returns {text, calls:[{name, arguments(object)}]}. */
    push(delta) {
      if (!delta) return { text: '', calls: [] }
      buf += delta
      const r = drain()
      for (const c of r.calls) pending.push(c)
      return { text: r.text, calls: r.calls }
    },
    /** Flush at stream end. Returns {text, calls, dropped}. */
    end() {
      const r = drain()
      for (const c of r.calls) pending.push(c)
      let tail = r.text
      if (inBlock) {
        // unterminated block → not shown to the user
        dropped = true
        inBlock = false
        blockJson = ''
        buf = ''
      } else if (buf) {
        // held-back tag prefix at EOS is just literal text
        tail += buf
        buf = ''
      }
      return { text: tail, calls: r.calls, dropped: dropped || undefined }
    },
    /** All calls seen so far (for session bookkeeping). */
    get calls() { return pending },
    get sawCall() { return inBlock || dropped || pending.length > 0 },
  }
}

function parseCall(json) {
  let obj
  try { obj = JSON.parse(json.trim()) } catch { return null }
  const name = obj?.name ?? obj?.tool ?? obj?.function?.name
  if (typeof name !== 'string' || !name) return null
  const argField = ['arguments', 'args', 'input', 'parameters'].find((field) => Object.hasOwn(obj, field))
  const args = toolArguments(argField === undefined ? {} : obj[argField])
  return args ? { name, arguments: args } : null
}

/** Assign OpenAI-style ids to parsed calls: call_<rand>. */
export function withCallIds(calls) {
  return calls.map((c) => ({
    id: `call_${crypto.randomUUID().replace(/-/g, '').slice(0, 24)}`,
    type: 'function',
    function: { name: c.name, arguments: JSON.stringify(c.arguments ?? {}) },
  }))
}
