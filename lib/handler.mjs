// Serve one OpenAI Chat Completions request through an Amazon Quick chat turn.
//
// Quick has no OpenAI-shaped HTTP API: the desktop client talks to a bidi
// WebSocket (`session_connect` → `user_message` → text_delta/thinking_delta/…
// → message_complete). This module renders the OpenAI message list into Quick's
// text protocol, runs the turn on a cached thread, and re-emits the event
// stream as chat.completion / chat.completion.chunk.
//
// OpenAI `tools` have no wire channel either, but the model emits a
// caller-defined <tool_call>{json}</tool_call> convention as plain text when
// the instruction is inline and strong (verified live). lib/tools.mjs owns the
// preamble + streaming parser; results return as role:'tool' messages rendered
// into the next turn's text.
import { createHash } from 'node:crypto'
import { QuickSocket, QuickWsError } from './quickws.mjs'
import { OutpostError } from './outpost.mjs'
import { uploadImage } from './attach.mjs'
import { normalizeTools, requiredToolChoice, canonicalJson, toolArguments, buildToolPreamble, createToolCallParser } from './tools.mjs'
import {
  AGENT_MODES, MODELS, modeFor, effortFor, renderTranscript, renderDelta, renderToolCalls, textOf,
  byteLen, capText, UPSTREAM_TEXT_LIMIT, canonRole, pickImages,
} from './translate.mjs'
import {
  chainHashes, sessionScope, findContinuation, extendHash, replyMessage,
  replyAnchor, tailAnchor,
} from './sessions.mjs'

const SURFACE = 'QUICK_WORK_DESKTOP_WINDOWS'
const APP_VERSION = '1.2.4660'
// system_prompt shares the user_message frame's ~64 KB budget with `text`;
// cap it so the transcript/tool preamble always keeps real room.
const SYSTEM_PROMPT_CAP = 32 * 1024
// Text room kept free when inline images claim the frame.
const IMAGE_TEXT_RESERVE = 8 * 1024
// Numeric plugin option with a default; 0 is a real value (disables the timer).
const msOpt = (v, fallback) => {
  if (v == null || v === '') return fallback
  const n = Number(v)
  return Number.isFinite(n) ? n : fallback
}
// Sentinel for "headerWaitMs elapsed while waiting for first output".
const HDR_WAIT = Symbol('header-wait')

export const PLAIN_TEXT_NOTE = 'This application renders plain text and Markdown only. Don\'t use AWS Quick UI markup (decision cards, artifacts, a sources block); when the user needs to choose, ask in plain text with a short numbered list. This doesn\'t apply to code or file contents the user asks for.'

function errorResponse(status, message, type = 'api_error') {
  return new Response(JSON.stringify({ error: { message, type, code: status } }), {
    status,
    headers: { 'content-type': 'application/json' },
  })
}

function errorType(err) {
  if (err?.kind === 'auth') return 'authentication_error'
  if (err?.status === 429 || err?.kind === 'quota') return 'rate_limit_error'
  return 'api_error'
}

function httpStatus(err) {
  if (err instanceof OutpostError || err instanceof QuickWsError) {
    if (err.kind === 'auth') return 401
    if (err.kind === 'abort') return 499
    if (err.kind === 'spawn' || err.kind === 'timeout') return 503
  }
  return err?.status && err.status >= 400 && err.status < 600 ? err.status : 502
}

const sockets = new WeakMap() // Outpost → QuickSocket

async function socketFor(outpost, options) {
  let s = sockets.get(outpost)
  if (!s) {
    const outpostId = await outpost.outpostId()
    // OAuth-provisioned sessions (direct backend) have no local outpost; the
    // server accepts an empty outpostId. A *local* outpost returning none is a
    // real auth failure though.
    if (!outpostId && !outpost.direct) throw new OutpostError('outpost 未返回设备 id（/local/outpost-id），请确认已登录 Amazon Quick', { kind: 'auth', status: 401 })
    s = new QuickSocket({
      getSession: () => outpost.session(),
      outpostId,
      extra: {
        surfaceType: options?.surfaceType || SURFACE,
        appVersion: options?.appVersion || APP_VERSION,
        channel: options?.channel,
      },
      log: options?.log,
    })
    sockets.set(outpost, s)
  }
  return s
}

/** Usage object from a turn_usage event payload (snake_case wire fields). */
function usageFrom(u) {
  if (!u || typeof u !== 'object') return null
  const input = Number(u.input_tokens ?? u.inputTokens) || 0
  const cached = Number(u.cache_read_tokens ?? u.cached_input_tokens ?? u.cachedInputTokens) || 0
  const written = Number(u.cache_write_tokens ?? u.cacheWriteTokens) || 0
  const completion = Number(u.output_tokens ?? u.outputTokens) || 0
  // Quick reports non-cached input, cache reads and cache writes separately;
  // OpenAI prompt_tokens is the whole prompt side.
  const prompt = input + cached + written
  const total = Number(u.total_tokens ?? u.totalTokens) || prompt + completion
  const cost = Number(u.cost_usd ?? u.costUsd)
  const ctx = Number(u.context_window ?? u.contextWindow)
  const resolved = u.resolved_model_id ?? u.served_model_id ?? u.resolvedModelId
  return {
    prompt_tokens: prompt,
    completion_tokens: completion,
    total_tokens: total || prompt + completion,
    ...(cached ? { prompt_tokens_details: { cached_tokens: cached } } : {}),
    // Quick-reported extras; OpenAI clients ignore unknown usage fields.
    ...(Number.isFinite(cost) && cost > 0 ? { cost } : {}),
    ...(Number.isFinite(ctx) && ctx > 0 ? { context_window: ctx } : {}),
    ...(typeof resolved === 'string' && resolved ? { resolved_model_id: resolved } : {}),
  }
}

function addUsage(acc, u) {
  if (!u) return acc
  acc = acc ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
  // Each agentic step re-reads the whole thread, so summing prompts counted
  // the same context once per step — clients size their context window from
  // prompt_tokens (Claude Code auto-compacted on it). The prompt is the
  // latest step's; only completions accumulate.
  if (u.prompt_tokens) {
    acc.prompt_tokens = u.prompt_tokens
    if (u.prompt_tokens_details?.cached_tokens) acc.prompt_tokens_details = { cached_tokens: u.prompt_tokens_details.cached_tokens }
    else delete acc.prompt_tokens_details
  }
  acc.completion_tokens += u.completion_tokens || 0
  acc.total_tokens = acc.prompt_tokens + acc.completion_tokens
  if (u.cost) acc.cost = (acc.cost || 0) + u.cost
  // Window and model are per-turn facts, not sums: keep the latest.
  if (u.context_window) acc.context_window = u.context_window
  if (u.resolved_model_id) acc.resolved_model_id = u.resolved_model_id
  return acc
}

const newCallId = () => `call_${crypto.randomUUID().replace(/-/g, '').slice(0, 24)}`
const toolProtocolError = (message) => new QuickWsError(message, { kind: 'tool_protocol' })
const callKey = ({ name, arguments: args }) => canonicalJson([name, args])
const batchKey = (calls) => JSON.stringify(calls.map(callKey).sort())

/**
 * Normalized turn output:
 *   {type:'meta', threadId, reused, mode}
 *   {type:'text', text}                 visible answer delta
 *   {type:'reasoning', text}            thinking_delta
 *   {type:'tool_call', id, name, arguments}   parsed <tool_call> block (arguments = JSON string)
 *   {type:'usage', usage}               turn_usage (per message)
 *   {type:'note', text}                 non-fatal oddities worth showing
 *   {type:'done', …}                    terminal bookkeeping
 */
async function* runTurn({ outpost, socket, body, tools, requiredTool, parallel, sessions, options, signal }) {
  const messages = Array.isArray(body.messages) ? body.messages : []
  const convo = messages.filter((m) => m.role !== 'system' && m.role !== 'developer')
  const systemText = messages
    .filter((m) => m.role === 'system' || m.role === 'developer')
    .map((m) => textOf(m.content))
    .join('\n\n')
  const mode = modeFor(body.model)
  const effort = effortFor(options?.effortHeader, body)
  // Quick can emit client calls via either structured events or our text convention.
  // The client receives one validated batch per model step.
  const clientTools = new Map(tools.map((tool) => [tool.name, tool]))
  const findClientTool = (name) => {
    if (!tools.length || typeof name !== 'string') return undefined
    const exact = clientTools.get(name)
    if (exact) return exact
    const matches = tools.filter((tool) => tool.name.toLowerCase() === name.toLowerCase())
    return matches.length === 1 ? matches[0] : undefined
  }
  const validateCall = (call) => {
    const tool = findClientTool(call.name)
    if (!tool) throw toolProtocolError(`Quick 调用了未注册的客户端工具 ${call.name}，未执行`)
    if (typeof requiredTool === 'string' && requiredTool !== tool.name) {
      throw toolProtocolError(`Quick 未调用指定工具 ${requiredTool}，未执行`)
    }
    const args = toolArguments(call.arguments)
    if (!args) throw toolProtocolError(`Quick 工具 ${tool.name} 的参数必须是 JSON 对象，未执行`)
    const required = tool.parameters.required
    const missing = Array.isArray(required) ? required.filter((field) => !Object.hasOwn(args, field)) : []
    if (missing.length) throw toolProtocolError(`Quick 工具 ${tool.name} 缺少必填参数 ${missing.join(', ')}，未执行`)
    return { name: tool.name, arguments: args }
  }
  // The client's system prompt rides the dedicated user_message.system_prompt
  // field (verified live: the server honors it as a per-turn system override,
  // even at ~54 KB). It shares the same ~64 KB frame budget as `text`, so both
  // are capped together — the transcript budget shrinks by whatever the system
  // prompt occupies.
  const systemPrompt = systemText ? capText(systemText, SYSTEM_PROMPT_CAP) : ''
  const stats = {}
  if (systemText && byteLen(systemText) > SYSTEM_PROMPT_CAP) stats.truncated = true
  const sessionKey = typeof options?.sessionKey === 'string' ? options.sessionKey.slice(0, 128) : ''
  // Quick's own tool_start/tool_progress/tool_result events are agent-internal;
  // showQuickTools surfaces each tool_start as a [Quick: name] reasoning line.
  const showQuickTools = options?.showQuickTools !== false
  // custom_prompt is a real per-turn instruction channel (verified live). Use
  // it to tell the model this session is app-driven: with client tools, steer
  // local-environment work to the <tool_call> protocol; on direct backends
  // (OAuth / Credential Manager — no outpost, hence no connected device) also
  // say built-in device tools cannot run, so the model doesn't burn agentic
  // turns on guaranteed failures. The local-outpost path may genuinely have a
  // device attached, so the claim stays off there.
  const directBackend = !!outpost?.direct
  const baseCustomPrompt = tools.length
    ? directBackend
      ? 'Calling context: this chat is driven by an external application. The client tools listed in the message are executed by that application on the user\'s machine; this session has no connected device, so built-in device tools (local files, apps) cannot run here. For questions about the user\'s project, files or code, call the client tools to read current contents — earlier tool results may be partial, and a fresh read beats guessing.'
      : 'Calling context: this chat is driven by an external application. The client tools listed in the message are executed by that application on the user\'s machine — for questions about the user\'s project, files or code, call them to read current contents; earlier tool results may be partial, and a fresh read beats guessing.'
    : directBackend
      ? 'Calling context: this chat is driven by an external application and has no connected device. Built-in tools that need the user\'s device (local files, apps) cannot run in this session — say so plainly rather than attempting them.'
      : ''
  // Quick's own system prompt asks for Quick-app UI markup (decision cards,
  // artifacts). External clients render plain text/Markdown, so that markup
  // leaks as raw tags. Scoped so code/file output (e.g. HTML <option>) is unaffected.
  const customPrompt = [baseCustomPrompt, PLAIN_TEXT_NOTE].filter(Boolean).join(' ')
  const textCap = Math.max(8 * 1024, UPSTREAM_TEXT_LIMIT - byteLen(systemPrompt) - byteLen(customPrompt) - 1024)

  // A Quick thread can learn new tools. A definition it has already seen
  // still needs to be available and unchanged to continue that thread.
  const identity = options?.identity || 'local'
  const offered = typeof requiredTool === 'string' ? [clientTools.get(requiredTool)] : tools
  const currentSpecs = Object.fromEntries(offered.map((tool) => [
    tool.name, createHash('sha256').update(canonicalJson(tool)).digest('hex'),
  ]))
  const scope = sessionScope(identity, mode, systemText)
  const hashes = chainHashes(scope, convo)
  const cont = sessions ? findContinuation(sessions, hashes, convo) : undefined
  const taken = cont ? [[cont.key, cont.entry]] : []
  const restoreTaken = () => {
    for (const [key, entry] of taken) if (!sessions.has(key)) sessions.set(key, entry)
    taken.length = 0
  }
  // Fuzzy resume: context extensions (pi-magic-context, rewind hooks, …) can
  // prune or rewrite old messages, breaking the strict prefix chain. Anchor on
  // our last reply echo + its 3 predecessors — content hashes survive the
  // position shifts such edits cause. Only the LAST assistant can yield an
  // appendable delta anyway.
  let fuzzy
  if (!cont && sessions) {
    let a = -1
    for (let i = convo.length - 1; i >= 0; i--) {
      if (canonRole(convo[i]?.role) === 'assistant') { a = i; break }
    }
    if (a >= 3) {
      const rk = replyAnchor(scope, convo[a])
      const tk = tailAnchor(scope, convo.slice(a - 3, a))
      const hit = sessions.findEntry((e) => e.replyKey === rk && e.tailKey === tk)
      if (hit) {
        const delta = convo.slice(a + 1)
        if (delta.length && delta.every((m) => ['user', 'tool', 'function'].includes(canonRole(m.role)))) {
          const entry = sessions.take(hit.key)
          if (entry) {
            taken.push([hit.key, entry])
            fuzzy = { entry, delta }
          }
        }
      }
    }
  }
  // Session-keyed fallback: gateways like magpie stamp every request of a
  // chat with a stable session id (x-magpie-session). When content matching
  // fails entirely — heavy client-side compaction rewrites even our reply
  // echo — the session boundary still identifies the conversation, so trust
  // it and append whatever follows the last assistant message. Verified
  // matches (strict/fuzzy) always win; this is the last resort.
  let sessHit
  if (!cont && !fuzzy && sessions && sessionKey) {
    const sk = `sess:${scope}:${sessionKey}`
    if (sessions.has(sk)) {
      let a = -1
      for (let i = convo.length - 1; i >= 0; i--) {
        if (canonRole(convo[i]?.role) === 'assistant') { a = i; break }
      }
      if (a >= 0) {
        const delta = convo.slice(a + 1)
        if (delta.length && delta.every((m) => ['user', 'tool', 'function'].includes(canonRole(m.role)))) {
          const entry = sessions.take(sk)
          if (entry) {
            // Subagents can share a session key. Confirm whose calls these are.
            const calls = new Set(entry.callIds ?? [])
            const results = delta.filter((m) => canonRole(m.role) === 'tool')
            if (results.every((m) => calls.has(m.tool_call_id ?? m.id))) {
              taken.push([sk, entry])
              sessHit = { entry, delta }
            } else sessions.set(sk, entry)
          }
        }
      }
    }
  }
  let reuse = cont ?? fuzzy ?? sessHit
  if (reuse?.entry.toolSpecs && Object.entries(reuse.entry.toolSpecs).some(([name, spec]) => currentSpecs[name] !== spec)) {
    restoreTaken()
    reuse = undefined
  }
  // A tool preamble that would leave under ~8 KB for the actual message
  // degrades to names+descriptions first — otherwise the frame hard-cap
  // (a tail cut) would eat the user's question and the model would see
  // only the tool list. Names-only is the last resort: verbose descriptions
  // can keep even the lite preamble over budget, and a shed preamble means
  // zero tool instructions and guaranteed plain-text answers.
  const fitPreamble = (list = offered) => {
    let pre = buildToolPreamble(list, body.tool_choice, { parallel })
    if (byteLen(pre) > textCap - 8192) {
      const lite = buildToolPreamble(list, body.tool_choice, { lite: true, parallel })
      if (byteLen(lite) < byteLen(pre)) { pre = lite; stats.truncated = true }
    }
    if (byteLen(pre) > textCap - 4096) {
      const names = buildToolPreamble(list, body.tool_choice, { namesOnly: true, parallel })
      if (byteLen(names) < byteLen(pre)) { pre = names; stats.truncated = true }
    }
    return pre
  }
  const known = reuse?.entry.toolSpecs ?? {}
  const added = reuse ? offered.filter(({ name }) => !Object.hasOwn(known, name)) : []
  const preamble = reuse
    ? [fitPreamble(added), buildToolPreamble(tools, body.tool_choice, { reminder: true, parallel })].filter(Boolean).join('\n\n')
    : fitPreamble()
  let toolSpecs = { ...known, ...currentSpecs }
  const textBudget = Math.max(4 * 1024, textCap - byteLen(preamble))

  // Small images ride inline; larger ones can use Quick's cloud attachments.
  // Only this turn's images are sent. Inline bytes shrink the text budget.
  const lastAsst = convo.findLastIndex((m) => canonRole(m.role) === 'assistant')
  const imageSlice = reuse ? reuse.delta : convo.slice(lastAsst + 1)
  const images = pickImages(imageSlice, Math.max(0, textBudget - IMAGE_TEXT_RESERVE), {
    upload: directBackend || options?.uploadLargeImagesWithOutpost === true,
  })
  const turnBudget = textBudget - images.bytes
  const frameCap = textCap - images.bytes

  let threadId = reuse ? reuse.entry.threadId : crypto.randomUUID()
  const imageNoteFor = (uploadFailed) => {
    const n = images.skipped + uploadFailed
    return n ? `\n\n[${n} image(s) not delivered: too large, upload failed, or not inline data]` : ''
  }
  // Deliver the queued uploads into tid's cloud workspace, in parallel.
  // Failures degrade to the not-delivered note; aborts propagate.
  const uploadAll = async (tid) => {
    if (!images.toUpload.length) return { entries: [], failed: 0 }
    const sess = await outpost.session()
    const results = await Promise.allSettled(images.toUpload.map((img) =>
      uploadImage({
        session: sess, image: img, name: img.name, threadId: tid,
        fetchImpl: options?.fetchImpl ?? globalThis.fetch, signal,
      })
    ))
    const entries = []
    let failed = 0
    for (const r of results) {
      if (r.status === 'fulfilled') { entries.push(r.value); continue }
      if (signal?.aborted || r.reason?.name === 'AbortError') throw r.reason
      failed++
      options?.log?.(`[aws-quick] image upload failed: ${String(r.reason?.message ?? r.reason).slice(0, 200)}`)
    }
    return { entries, failed }
  }
  let uploaded
  try {
    uploaded = await uploadAll(threadId)
  } finally {
    if (!uploaded) restoreTaken()
  }
  const imageNote = imageNoteFor(uploaded.failed)
  const attachments = images.attachments.concat(uploaded.entries)

  let text
  let reused = false
  if (reuse) {
    // A turn that ended in native-tool salvage was stopped server-side: the
    // reused thread's history holds an empty assistant message + "You stopped
    // the response." — NOT the call the client is now answering. Re-state it
    // (same rendering transcript history uses for tool_calls) ahead of the
    // delta, once: the entry written after this turn carries no flag.
    const previousCalls = convo[convo.length - reuse.delta.length - 1].tool_calls ?? []
    const hasResults = reuse.delta.some((message) => canonRole(message.role) === 'tool')
    let toolPrefix = ''
    if (reuse.entry.salvageCalls?.length) {
      toolPrefix = `[The client is returning results for these tool calls:]\nassistant: ${renderToolCalls(reuse.entry.salvageCalls)}\n\n`
    } else if (hasResults && previousCalls.length) {
      toolPrefix = `[Client tool result IDs in previous call order: ${previousCalls.map((call) => `${call.function.name}=${call.id}`).join(', ')}]\n\n`
    }
    text = toolPrefix + renderDelta(reuse.delta, { maxBytes: turnBudget - byteLen(toolPrefix), stats }) + imageNote
    reused = true
  } else {
    text = renderTranscript(convo, '', { maxBytes: turnBudget, stats }) + imageNote
  }
  // The tool convention rides inside the user_message text (there is no
  // client-tool channel); on reused threads only the short reminder rides,
  // since every byte of text stays in the thread and counts toward context.
  let upstreamText = preamble ? `${preamble}\n\n${text}` : text
  if (byteLen(upstreamText) > frameCap) {
    // Last resort before the tail-cutting hard cap: shed the preamble so the
    // user's message survives intact (tool calling degrades, the answer doesn't).
    upstreamText = byteLen(text) <= frameCap ? text : capText(text, frameCap)
    toolSpecs = known
    stats.truncated = true
  }
  if (!upstreamText.trim()) upstreamText = '(empty)'

  const state = { text: '', reasoning: '', usage: null, calls: [] }
  // Calls Quick's own history may not have retained, for the next reused turn.
  const salvageCalls = []
  let sawToolOutput = false

  const payload = {
    text: upstreamText,
    message_id: crypto.randomUUID(),
    agent_mode: mode,
    user_timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC',
    user_locale: (Intl.DateTimeFormat().resolvedOptions().locale || 'en-US').replace('_', '-'),
  }
  if (effort) payload.thinking_effort = effort
  if (systemPrompt) payload.system_prompt = systemPrompt
  if (customPrompt) payload.custom_prompt = customPrompt
  if (tools.length) payload.web_search_enabled = options?.webSearchWithClientTools === true && !requiredTool
  if (options?.chatAgentId) payload.chat_agent_id = options.chatAgentId
  if (attachments.length) payload.attachments = attachments

  async function* consume(turnIt, tid) {
    let parser = tools.length ? createToolCallParser() : null
    let previousText = ''
    let streamedText = ''
    const textCalls = []
    const nativeCalls = []
    const nativeIds = new Map()
    const requireCall = () => {
      if (requiredTool) throw toolProtocolError('Quick 未调用请求要求的客户端工具，未执行')
    }
    function* emitText(parsed) {
      textCalls.push(...parsed.calls)
      if (parsed.text) {
        state.text += parsed.text
        yield { type: 'text', text: parsed.text }
      }
    }
    function* consumeText(delta) {
      streamedText += delta
      const parsed = parser ? parser.push(delta) : { text: delta, calls: [] }
      if (parser?.sawCall) sawToolOutput = true
      yield* emitText(parsed)
    }
    function* finishStep({ interrupt, terminal }) {
      if (turnCtl.signal.aborted) throw new QuickWsError('aborted', { kind: 'abort' })
      if (parser) {
        const tail = parser.end()
        if (tail.dropped) throw toolProtocolError('Quick 工具调用格式不完整或无效，未执行')
        yield* emitText(tail)
      }
      const quoted = textCalls.map(validateCall)
      if (nativeCalls.length && quoted.length && batchKey(nativeCalls) !== batchKey(quoted)) {
        throw toolProtocolError('Quick 的结构化工具调用和文本调用冲突，未执行')
      }
      const calls = nativeCalls.length ? nativeCalls : quoted
      if (!calls.length && terminal) requireCall()
      if (!parallel && calls.length > 1) throw toolProtocolError('本轮不允许并行工具调用，Quick 返回了多个调用，未执行')
      if (calls.length && turnCtl.signal.aborted) throw new QuickWsError('aborted', { kind: 'abort' })
      if (interrupt && calls.length) {
        stopTurn(tid)
        turnCtl.abort()
      }
      for (const call of calls) {
        const id = newCallId()
        const args = JSON.stringify(call.arguments)
        state.calls.push({ id, name: call.name })
        if (nativeCalls.length || interrupt) salvageCalls.push({ id, name: call.name, arguments: args })
        yield { type: 'tool_call', id, name: call.name, arguments: args }
      }
      return calls.length
    }
    for await (const evt of turnIt) {
      if (turnCtl.signal.aborted) throw new QuickWsError('aborted', { kind: 'abort' })
      if (progressEvent(evt)) bumpWatchdog()
      const t = evt.type
      const p = evt.payload || {}
      if (t === 'text_delta' && typeof p.delta === 'string') {
        yield* consumeText(p.delta)
      } else if (t === 'thinking_delta' && typeof p.delta === 'string') {
        state.reasoning += p.delta
        yield { type: 'reasoning', text: p.delta }
      } else if (t === 'tool_start' && showQuickTools && typeof p.tool_name === 'string' && p.tool_name && !findClientTool(p.tool_name)) {
        // Quick runs its own tools server-side; surface each one as a
        // reasoning line so long agentic turns don't look dead. Client-tool
        // names are skipped — the salvaged call itself is the surface.
        const note = `\n[Quick: ${p.tool_name}]\n`
        state.reasoning += note
        yield { type: 'reasoning', text: note }
      } else if (t === 'tool_arguments') {
        if (findClientTool(p.tool_name)) {
          sawToolOutput = true
          const call = validateCall({ name: p.tool_name, arguments: p.arguments })
          if (p.tool_id) {
            const key = callKey(call)
            if (nativeIds.has(p.tool_id)) {
              if (nativeIds.get(p.tool_id) !== key) throw toolProtocolError('Quick 重复使用了同一工具调用 ID，参数冲突，未执行')
              continue
            }
            nativeIds.set(p.tool_id, key)
          }
          nativeCalls.push(call)
        }
      } else if (t === 'turn_usage') {
        const u = usageFrom(p)
        if (u) { state.usage = addUsage(state.usage, u); yield { type: 'usage', usage: u } }
        if (nativeCalls.length) {
          yield* finishStep({ interrupt: true, terminal: true })
          return
        }
      } else if (t === 'message_complete') {
        if (typeof p.text === 'string' && (!streamedText || parser)) {
          const wholePrefix = previousText + streamedText
          const prefix = p.text.startsWith(wholePrefix) ? wholePrefix : streamedText
          if (p.text.startsWith(prefix)) yield* consumeText(p.text.slice(prefix.length))
        }
        const continuation = p.continuation_expected === true
        const count = yield* finishStep({ interrupt: continuation, terminal: !continuation })
        if (count || !continuation) return
        previousText += streamedText
        streamedText = ''
        if (parser) parser = createToolCallParser()
      } else if (t === 'message_start' || t === 'connected' || t === 'session_established' || t === 'keepalive') {
        // noise
      } else if (t === 'status') {
        const s = p.status || p.message
        if (s && !/pong|ok/i.test(String(s))) yield { type: 'note', text: `[quick] ${s}` }
      } else if (t === 'turn_aborted') {
        throw new QuickWsError('Quick 已中止本轮', { kind: 'abort' })
      } else if (t === 'guardrail_blocked' || t === 'error') {
        // Error details may live in payload OR at the event's top level;
        // dump the whole event (truncated) when nothing readable is found.
        const detail = p.message || p.reason || p.error || evt.message || evt.reason || evt.error || evt.code
        const extra = detail ? '' : ` ${JSON.stringify(evt).slice(0, 400)}`
        throw new QuickWsError(`${detail || `quick ${t}`}${extra}`, { kind: 'stream' })
      }
      // Quick's tool_result and tool_progress do not seal the model step.
    }
    if (turnCtl.signal.aborted) throw new QuickWsError('aborted', { kind: 'abort' })
    const tail = parser?.end()
    if (nativeCalls.length || textCalls.length || tail?.calls.length || tail?.dropped) {
      throw toolProtocolError('Quick 工具调用中断，未执行')
    }
    if (tail) yield* emitText(tail)
    requireCall()
  }

  // Idle watchdog: every Quick event is progress except transport noise
  // (keepalive/ping/pong and ok-status pings). The timer aborts an internal
  // controller linked to the request signal, so socket.turn wakes even when
  // Quick sends nothing at all; watchdogFired distinguishes it from a client
  // abort when translating the error below.
  const idleMs = msOpt(options?.idleTimeoutMs, 300_000)
  const turnCtl = new AbortController()
  const linkRequestAbort = () => turnCtl.abort()
  if (signal) {
    if (signal.aborted) turnCtl.abort()
    else signal.addEventListener('abort', linkRequestAbort, { once: true })
  }
  let watchdogFired = false
  let watchdogTimer = null
  const bumpWatchdog = () => {
    if (!idleMs) return
    if (watchdogTimer) clearTimeout(watchdogTimer)
    watchdogTimer = setTimeout(() => { watchdogFired = true; turnCtl.abort() }, idleMs)
    watchdogTimer.unref?.()
  }
  const progressEvent = (evt) => {
    const t = evt?.type
    if (t === 'keepalive' || t === 'ping' || t === 'pong') return false
    // Quick sends these even when the request itself has stalled.
    if (t === 'runs_updated' || t === 'task_list_update' || t === 'title_generated' || t === 'session_summary_updated' || t === 'session_sync') return false
    if (t === 'status') {
      const s = evt?.payload?.status || evt?.payload?.message || ''
      if (/pong|ok/i.test(String(s))) return false
    }
    return true
  }
  // Quick's own stop message (the frame the desktop sends on stop): socket.stop
  // sends it AND arms the per-thread settle window, so the next turn() on this
  // thread waits out Quick's stale terminal-event tail instead of losing its
  // user_message. Covers watchdog timeout, client abort and tool salvage.
  const stopTurn = (tid) => {
    try { socket.stop(tid) } catch {}
  }
  // Translate a watchdog/client abort into the error the caller should see;
  // returns null for ordinary turn errors (which may still hit the retry).
  const turnTermination = (e, tid) => {
    if (watchdogFired) {
      stopTurn(tid)
      return new QuickWsError(`Quick 已 ${idleMs / 1000}s 没有进展，已中止本轮 (idleTimeoutMs)`, { kind: 'timeout' })
    }
    if (signal?.aborted || e?.kind === 'abort') {
      stopTurn(tid)
      return e
    }
    return null
  }
  const produced = () => sawToolOutput || state.text.length > 0 || state.calls.length > 0 || state.reasoning.length > 0
  let turnOk = false
  try {
    yield { type: 'meta', threadId, reused, mode }
    if (stats.truncated) yield { type: 'note', text: '上下文超过 Quick 64KB 上限，已截断较早内容' }
    bumpWatchdog()
    for await (const o of consume(socket.turn(threadId, payload, { signal: turnCtl.signal }), threadId)) yield o
    turnOk = true
  } catch (e) {
    const term = turnTermination(e, threadId)
    if (term) throw term
    if (!(reuse && sessions) || produced() || e?.kind === 'auth' || e?.kind === 'timeout' || e?.kind === 'tool_protocol') throw e
    // Cached thread may be gone server-side: retry once on a fresh thread with
    // the full transcript (same policy as magpie-comate). Uploaded files live
    // in the dead thread's workspace, so they are delivered again for `fresh`.
    const fresh = crypto.randomUUID()
    taken.length = 0
    toolSpecs = currentSpecs
    const up2 = await uploadAll(fresh)
    const retryAttachments = images.attachments.concat(up2.entries)
    const full = renderTranscript(convo, '', { maxBytes: turnBudget, stats }) + imageNoteFor(up2.failed)
    const fullPreamble = tools.length ? fitPreamble() : ''
    const fullText = fullPreamble ? `${fullPreamble}\n\n${full}` : full
    const capped = byteLen(fullText) > frameCap ? capText(fullText, frameCap) : fullText
    state.text = ''
    state.reasoning = ''
    state.usage = null
    state.calls = []
    salvageCalls.length = 0
    threadId = fresh
    reused = false
    yield { type: 'meta', threadId: fresh, reused: false, mode }
    payload.text = capped.trim() || '(empty)'
    payload.message_id = crypto.randomUUID()
    if (retryAttachments.length) payload.attachments = retryAttachments
    else delete payload.attachments
    try {
      for await (const o of consume(socket.turn(fresh, payload, { signal: turnCtl.signal }), fresh)) yield o
      turnOk = true
    } catch (e2) {
      const term2 = turnTermination(e2, fresh)
      if (term2) throw term2
      throw e2
    }
  } finally {
    if (watchdogTimer) clearTimeout(watchdogTimer)
    signal?.removeEventListener?.('abort', linkRequestAbort)
    if (!turnOk) restoreTaken()
  }

  // Remember where this conversation stands for the next request.
  if (sessions && (state.text || state.calls.length)) {
    const reply = replyMessage(state.text, state.calls)
    const key = extendHash(hashes[convo.length], reply)
    // salvageCalls ride the entry so the next reused turn can re-state the
    // interrupted call (memory only — the file store keeps hashes + threadIds,
    // never message content, so a reloaded entry simply lacks the prefix).
    const salvaged = salvageCalls.length ? { salvageCalls } : {}
    sessions.set(key, {
      threadId,
      toolSpecs,
      replyKey: replyAnchor(scope, reply),
      tailKey: tailAnchor(scope, convo.slice(-3)),
      ...salvaged,
    })
    // Session-keyed entry self-perpetuates: rewritten every turn so the
    // fallback survives however much the client rewrites its history.
    if (sessionKey) sessions.set(`sess:${scope}:${sessionKey}`, {
      threadId, toolSpecs, callIds: state.calls.map(({ id }) => id), ...salvaged,
    })
  } else restoreTaken()
  yield { type: 'done', threadId, text: state.text, reasoning: state.reasoning, usage: state.usage }
}

function makeId(threadId) {
  return `chatcmpl-quick-${threadId?.slice(0, 8) ?? 'x'}-${Date.now().toString(36)}`
}

/**
 * @param {object} p
 * @param {object} p.body       parsed Chat Completions request
 * @param {object} p.outpostMgr createOutpostManager() result
 * @param {object} [p.sessions] session store (lib/sessions.mjs)
 * @param {object} [p.options]  plugin options
 * @param {string} [p.effortHeader] X-Magpie-Effort style header value
 * @param {AbortSignal} [p.signal]
 */
export async function handleChatCompletions({ body, outpostMgr, sessions, options, effortHeader, signal }) {
  if (!body || typeof body !== 'object') return errorResponse(400, 'request body must be JSON', 'invalid_request_error')
  if (!Array.isArray(body.messages) || !body.messages.length) {
    return errorResponse(400, 'messages is required', 'invalid_request_error')
  }
  const choice = body.tool_choice
  const tools = choice === 'none' || choice?.type === 'none' ? [] : normalizeTools(body.tools)
  const requiredTool = requiredToolChoice(choice)
  if (requiredTool && (typeof requiredTool === 'string' ? !tools.some((tool) => tool.name === requiredTool) : !tools.length)) {
    return errorResponse(400, 'tool_choice requires a tool present in tools', 'invalid_request_error')
  }
  const parallel = body.parallel_tool_calls !== false && choice?.disable_parallel_tool_use !== true
  let outpost
  try {
    outpost = await outpostMgr.ensure()
  } catch (e) {
    return errorResponse(httpStatus(e), e?.message ?? String(e), errorType(e))
  }
  let socket
  try {
    socket = await socketFor(outpost, { ...options, effortHeader, log: options?.log })
  } catch (e) {
    return errorResponse(httpStatus(e), e?.message ?? String(e), errorType(e))
  }

  const model = typeof body.model === 'string' && body.model.trim() ? body.model.trim() : 'smart'
  const it = runTurn({ outpost, socket, body: { ...body, model }, tools, requiredTool, parallel, sessions, options: { ...options, effortHeader }, signal })

  // Pull up to the first real output before answering so auth/session failures
  // surface as real HTTP statuses (magpie marks the account on 401, fails over
  // on 429/5xx) instead of a 200 stream that dies immediately. For streaming,
  // headerWaitMs caps that wait: agentic turns can sit silent while Quick runs
  // its own tools, and the gateway only waits so long for response headers.
  // pull() keeps at most one it.next() in flight — after the deadline the same
  // pending promise is what items awaits, never a second next() call.
  const headerWaitMs = msOpt(options?.headerWaitMs, 10_000)
  const giveUpAt = body.stream && headerWaitMs > 0 ? Date.now() + headerWaitMs : 0
  const pre = []
  let done = false
  let meta
  let pendingNext = null
  const pull = () => {
    if (!pendingNext) {
      const p = it.next()
      pendingNext = p
      const clear = () => { if (pendingNext === p) pendingNext = null }
      p.then(clear, clear)
    }
    return pendingNext
  }
  try {
    for (;;) {
      let r
      if (giveUpAt) {
        const remaining = giveUpAt - Date.now()
        if (remaining <= 0) break
        let timer
        const timedOut = new Promise((res) => { timer = setTimeout(() => res(HDR_WAIT), remaining) })
        r = await Promise.race([pull(), timedOut])
        clearTimeout(timer)
        if (r === HDR_WAIT) break
      } else {
        r = await pull()
      }
      if (r.done) { done = true; break }
      if (r.value.type === 'meta') { meta = r.value; continue }
      if (r.value.type === 'note') continue // notes ride inside the stream
      pre.push(r.value)
      break
    }
  } catch (e) {
    return errorResponse(httpStatus(e), e?.message ?? String(e), errorType(e))
  }
  const items = (async function* () {
    yield* pre
    if (done) return
    for (;;) {
      const r = await pull()
      if (r.done) return
      yield r.value
    }
  })()

  const created = Math.floor(Date.now() / 1000)
  const id = makeId(meta?.threadId)

  if (!body.stream) {
    let text = ''
    let reasoning = ''
    const calls = []
    let usage
    try {
      for await (const o of items) {
        if (o.type === 'text') text += o.text
        else if (o.type === 'reasoning') reasoning += o.text
        else if (o.type === 'tool_call') calls.push({ id: o.id, type: 'function', function: { name: o.name, arguments: o.arguments } })
        else if (o.type === 'usage') usage = addUsage(usage, o.usage)
      }
    } catch (e) {
      return errorResponse(httpStatus(e), e?.message ?? String(e), errorType(e))
    }
    const message = { role: 'assistant', content: text }
    if (reasoning) message.reasoning_content = reasoning
    if (calls.length) message.tool_calls = calls
    return new Response(
      JSON.stringify({
        id, object: 'chat.completion', created, model,
        choices: [{ index: 0, message, finish_reason: calls.length ? 'tool_calls' : 'stop' }],
        usage: usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
      }),
      { status: 200, headers: { 'content-type': 'application/json' } },
    )
  }

  const enc = new TextEncoder()
  const includeUsage = !!body.stream_options?.include_usage
  const stream = new ReadableStream({
    async start(controller) {
      const send = (obj) => controller.enqueue(enc.encode(`data: ${JSON.stringify(obj)}\n\n`))
      const chunk = (delta, finish = null) => send({ id, object: 'chat.completion.chunk', created, model, choices: [{ index: 0, delta, finish_reason: finish }] })
      let sentRole = false
      const role = () => {
        if (!sentRole) {
          sentRole = true
          chunk({ role: 'assistant', content: '' })
        }
      }
      let toolIndex = 0
      let usage
      try {
        for await (const o of items) {
          if (o.type === 'text') {
            role()
            chunk({ content: o.text })
          } else if (o.type === 'reasoning') {
            role()
            chunk({ reasoning_content: o.text })
          } else if (o.type === 'tool_call') {
            role()
            chunk({ tool_calls: [{ index: toolIndex++, id: o.id, type: 'function', function: { name: o.name, arguments: o.arguments } }] })
          } else if (o.type === 'usage') usage = addUsage(usage, o.usage)
          else if (o.type === 'note') {
            role()
            chunk({ content: `\n\n> ${o.text}` })
          }
        }
        role()
        chunk({}, toolIndex > 0 ? 'tool_calls' : 'stop')
        if (includeUsage) send({ id, object: 'chat.completion.chunk', created, model, choices: [], usage: usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 } })
        controller.enqueue(enc.encode('data: [DONE]\n\n'))
      } catch (e) {
        send({ error: { message: e?.message ?? String(e), type: errorType(e), code: httpStatus(e) } })
        controller.enqueue(enc.encode('data: [DONE]\n\n'))
      } finally {
        controller.close()
      }
    },
    cancel() {
      items.return?.()
      it.return?.()
    },
  })
  return new Response(stream, {
    status: 200,
    headers: { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache' },
  })
}

/** /v1/models in OpenAI shape. */
export function handleModels() {
  return new Response(JSON.stringify({ object: 'list', data: MODELS }), {
    headers: { 'content-type': 'application/json' },
  })
}

export { AGENT_MODES }
