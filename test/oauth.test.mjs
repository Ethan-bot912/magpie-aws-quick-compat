import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, rmSync, readFileSync, writeFileSync, existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import {
  createSessionStore, runOAuthLogin, startOAuthLogin, refreshSession, buildSession,
  keyForSession, createDirectBackend, OAuthError, rememberDiscovery, takeDiscovery,
} from '../lib/oauth.mjs'

const tmp = () => mkdtempSync(path.join(tmpdir(), 'oauth-test-'))

/** Mock fetch answering the OIDC endpoints the flow calls. */
const mockFetch = (log = []) => async (url, init = {}) => {
  const u = String(url)
  log.push({ url: u, body: init.body ? JSON.parse(init.body) : null })
  const reply = (data, status = 200) => new Response(JSON.stringify(data), { status, headers: { 'content-type': 'application/json' } })
  if (u.endsWith('/client/register')) return reply({ clientId: 'cid-1', clientSecret: 'csec-1' })
  if (u.endsWith('/token')) {
    const b = JSON.parse(init.body)
    if (b.grantType === 'authorization_code') {
      return reply({ accessToken: 'at-1', refreshToken: 'rt-1', idToken: 'idt-1', expiresIn: 3600 })
    }
    if (b.grantType === 'refresh_token') {
      return reply({ accessToken: 'at-2', refreshToken: 'rt-2', idToken: 'idt-2', expiresIn: 3600 })
    }
    return reply({ error: 'bad grant' }, 400)
  }
  if (u.endsWith('/integration/quick-work/user-info')) {
    return reply({ email: 'u@x.com', userArn: 'arn:aws:quicksight::1:user/u', awsAccountId: '111', accountName: 'acct' })
  }
  return reply({ error: 'unknown' }, 404)
}

/** Fake "browser": decodes the discovery URL and walks the redirect chain. */
async function fakeBrowser(url) {
  const surface = new URL(url).searchParams.get('qwSurface')
  const [nonce, port] = surface.split(':').slice(1)
  const cb = `http://127.0.0.1:${port}/callback?nonce=${encodeURIComponent(nonce)}` +
    `&issuerUrl=${encodeURIComponent('https://idc.example.com/issuer')}` +
    `&tenantUrl=${encodeURIComponent('https://qbs-cell.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev')}` +
    `&idcRegion=us-east-1`
  const r1 = await fetch(cb, { redirect: 'manual' })
  assert.equal(r1.status, 302)
  const loc = new URL(r1.headers.get('location'))
  const state = loc.searchParams.get('state')
  assert.ok(state)
  const r2 = await fetch(`http://127.0.0.1:${port}/?code=authcode-1&state=${encodeURIComponent(state)}`)
  assert.equal(r2.status, 200)
}

test('session store save/get/remove round-trip', () => {
  const dir = tmp()
  try {
    const store = createSessionStore({ dir })
    assert.equal(store.get('k'), undefined)
    store.set('k', { jwt: 'j', ws_host: 'h' })
    assert.equal(store.get('k').jwt, 'j')
    assert.deepEqual(store.list(), ['k'])
    store.remove('k')
    assert.equal(store.get('k'), undefined)
    // file written under dir with private name
    assert.ok(existsSync(path.join(dir, 'oauth-sessions.json')))
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('buildSession derives ws_host + jwt + usable_until', () => {
  const sess = buildSession({
    tenant_url: 'https://qbs-x.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev/',
    tokens: { access_token: 'a', refresh_token: 'r', id_token: 'i', expires_in: 3600 },
    user_info: { user_arn: 'arn:aws:quicksight::1:user/u', aws_account_id: '1', email: 'u@x' },
    client_id: 'c', client_secret: 's',
    discovery: { idc_region: 'us-east-1', issuer_url: 'https://idc/x' },
  })
  assert.equal(sess.ws_host, 'qbs-x.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev')
  assert.equal(sess.jwt, 'i')
  assert.equal(sess.actor_arn, 'arn:aws:quicksight::1:user/u')
  assert.ok(sess.usable_until > Math.floor(Date.now() / 1000))
  assert.equal(sess.client.idc_region, 'us-east-1')
})

test('full OAuth login flow: callback server + register + exchange + user-info', async () => {
  const calls = []
  const sess = await runOAuthLogin({
    fetchImpl: mockFetch(calls),
    openBrowser: fakeBrowser,
    log: () => {},
    timeoutMs: 15000,
  })
  assert.equal(sess.jwt, 'idt-1')
  assert.equal(sess.ws_host, 'qbs-cell.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev')
  assert.equal(sess.email, 'u@x.com')
  assert.equal(sess.tokens.refresh_token, 'rt-1')
  // called register → token → user-info in order
  assert.deepEqual(calls.map((c) => c.url.split('/').pop()), ['register', 'token', 'user-info'])
  const reg = calls[0].body
  assert.deepEqual(reg.grantTypes, ['authorization_code', 'refresh_token'])
  assert.equal(reg.clientName, 'AmazonQuick-Desktop')
  const exch = calls[1].body
  assert.equal(exch.grantType, 'authorization_code')
  assert.equal(exch.code, 'authcode-1')
  assert.ok(exch.codeVerifier)
})

test('startOAuthLogin returns the sign-in URL before any browser action; done resolves via the flow', async () => {
  const { url, done, cancel } = await startOAuthLogin({
    fetchImpl: mockFetch(),
    log: () => {},
    timeoutMs: 15000,
  })
  assert.equal(typeof cancel, 'function')
  assert.ok(url.includes('qwSurface=desktop%3A'), `url carries the encoded surface: ${url}`)
  // driving the mock browser against the returned url completes `done`
  await fakeBrowser(url)
  const sess = await done
  assert.equal(sess.jwt, 'idt-1')
  assert.equal(sess.ws_host, 'qbs-cell.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev')
  assert.equal(sess.email, 'u@x.com')
})

test('startOAuthLogin cancel() rejects done with kind abort and closes the server', async () => {
  const { url, done, cancel } = await startOAuthLogin({
    fetchImpl: mockFetch(),
    log: () => {},
    timeoutMs: 15000,
  })
  cancel()
  await assert.rejects(done, (e) => e instanceof OAuthError && e.kind === 'abort' && /cancelled/.test(e.message))
  // the callback server is closed: its port refuses connections
  const port = new URL(url).searchParams.get('qwSurface').split(':')[2]
  let refused = false
  for (let i = 0; i < 40 && !refused; i++) {
    try {
      await fetch(`http://127.0.0.1:${port}/`)
    } catch {
      refused = true
    }
    if (!refused) await new Promise((r) => setTimeout(r, 50))
  }
  assert.ok(refused, 'callback server still accepting connections after cancel()')
})

test('a stale-nonce or wrong-state callback does not end the login', async () => {
  const { url, done, cancel } = await startOAuthLogin({
    fetchImpl: mockFetch(),
    log: () => {},
    timeoutMs: 15000,
  })
  const port = new URL(url).searchParams.get('qwSurface').split(':')[2]
  // stray requests get the fail page but the login keeps waiting
  const r1 = await fetch(`http://127.0.0.1:${port}/callback?nonce=wrong&issuerUrl=x`)
  assert.equal(r1.status, 400)
  const r2 = await fetch(`http://127.0.0.1:${port}/?code=x&state=wrong`)
  assert.equal(r2.status, 400)
  await fakeBrowser(url)
  const sess = await done
  assert.equal(sess.jwt, 'idt-1')
  cancel()
})

test('submit() completes the login from a pasted hop-2 URL', async () => {
  const { url, done, submit, cancel } = await startOAuthLogin({
    fetchImpl: mockFetch(),
    log: () => {},
    timeoutMs: 15000,
  })
  const surface = new URL(url).searchParams.get('qwSurface')
  const [nonce, port] = surface.split(':').slice(1)
  // hop 1 hits the server (client registration runs) but the browser never
  // follows the 302 — its final address is pasted back instead
  const cb = `http://127.0.0.1:${port}/callback?nonce=${encodeURIComponent(nonce)}` +
    `&issuerUrl=${encodeURIComponent('https://idc.example.com/issuer')}` +
    `&tenantUrl=${encodeURIComponent('https://qbs-cell.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev')}` +
    `&idcRegion=us-east-1`
  const r1 = await fetch(cb, { redirect: 'manual' })
  assert.equal(r1.status, 302)
  const state = new URL(r1.headers.get('location')).searchParams.get('state')
  const sess = await submit(`http://127.0.0.1:${port}/?code=authcode-1&state=${encodeURIComponent(state)}`)
  assert.equal(sess.jwt, 'idt-1')
  assert.equal((await done).email, 'u@x.com')
  cancel()
})

test('submit() with a pasted hop-1 URL saves discovery; takeDiscovery is one-shot', async () => {
  const dir = tmp()
  try {
    const { url, submit, cancel } = await startOAuthLogin({
      fetchImpl: mockFetch(),
      log: () => {},
      timeoutMs: 15000,
      dir,
    })
    const surface = new URL(url).searchParams.get('qwSurface')
    const [nonce, port] = surface.split(':').slice(1)
    const hop1 = `http://127.0.0.1:${port}/callback?nonce=${encodeURIComponent(nonce)}` +
      `&issuerUrl=${encodeURIComponent('https://idc.example.com/issuer')}` +
      `&tenantUrl=${encodeURIComponent('https://tenant.example.com')}` +
      `&account=acct-1&idcInstanceId=inst-1&idcRegion=us-east-1`
    await assert.rejects(submit(hop1), (e) => e instanceof OAuthError && e.kind === 'discovery-saved')
    const file = path.join(dir, 'oauth-discovery.json')
    const saved = JSON.parse(readFileSync(file, 'utf8'))
    assert.equal(saved.issuer_url, 'https://idc.example.com/issuer')
    assert.equal(saved.tenant_url, 'https://tenant.example.com')
    assert.equal(saved.account, 'acct-1')
    assert.equal(saved.idc_instance_id, 'inst-1')
    assert.equal(saved.idc_region, 'us-east-1')
    assert.ok(saved.saved_at > 0)
    assert.equal(saved.refresh_token, undefined)
    // one-shot: the first take returns it, the second finds nothing
    const d = takeDiscovery(dir)
    assert.equal(d.issuer_url, 'https://idc.example.com/issuer')
    assert.equal(d.idc_region, 'us-east-1')
    assert.equal(takeDiscovery(dir), null)
    cancel()
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('takeDiscovery returns null for a stale saved_at; rememberDiscovery round-trips', () => {
  const dir = tmp()
  try {
    rememberDiscovery(dir, {
      issuer_url: 'https://idc.example.com/i',
      tenant_url: 'https://t.example.com',
      account: 'a',
      idc_instance_id: 'i',
      idc_region: 'us-west-2',
    })
    // age the entry past the 15-minute window
    const file = path.join(dir, 'oauth-discovery.json')
    const data = JSON.parse(readFileSync(file, 'utf8'))
    data.saved_at = Date.now() - 16 * 60 * 1000
    writeFileSync(file, JSON.stringify(data))
    assert.equal(takeDiscovery(dir), null)
    assert.equal(takeDiscovery(dir), null) // file is gone even when stale
    rememberDiscovery(dir, { issuer_url: 'https://idc.example.com/i2' })
    const d = takeDiscovery(dir, 1000)
    assert.equal(d.issuer_url, 'https://idc.example.com/i2')
    assert.equal(d.tenant_url, '')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('startOAuthLogin with saved discovery skips hop 1 and returns the authorize URL', async () => {
  const calls = []
  const { url, done, submit, cancel } = await startOAuthLogin({
    fetchImpl: mockFetch(calls),
    log: () => {},
    timeoutMs: 15000,
    discovery: {
      issuer_url: 'https://idc.example.com/issuer',
      tenant_url: 'https://qbs-cell.dp.appintegrations.us-east-1.prod.plato.ai.aws.dev',
      account: 'acct-1',
      idc_instance_id: 'inst-1',
      idc_region: 'us-east-1',
    },
  })
  assert.ok(url.startsWith('https://oidc.us-east-1.amazonaws.com/authorize'), url)
  const u = new URL(url)
  assert.equal(u.searchParams.get('client_id'), 'cid-1')
  // the client was registered before any browser action
  assert.deepEqual(calls.map((c) => c.url.split('/').pop()), ['register'])
  // and the pasted hop-2 address still completes the login
  const port = new URL(u.searchParams.get('redirect_uri')).port
  const sess = await submit(`http://127.0.0.1:${port}/?code=authcode-1&state=${encodeURIComponent(u.searchParams.get('state'))}`)
  assert.equal(sess.jwt, 'idt-1')
  assert.equal((await done).email, 'u@x.com')
  assert.deepEqual(calls.map((c) => c.url.split('/').pop()), ['register', 'token', 'user-info'])
  cancel()
})

test('submit() after the flow already completed returns the session', async () => {
  const { url, done, submit } = await startOAuthLogin({
    fetchImpl: mockFetch(),
    log: () => {},
    timeoutMs: 15000,
  })
  await fakeBrowser(url)
  const sess = await done
  // whatever the user pastes now — typically the success page's own address —
  // just hands the finished session back
  const again = await submit('http://127.0.0.1:1/?code=whatever&state=whatever')
  assert.equal(again.jwt, sess.jwt)
})

test('login fails cleanly when the callback gets an error', async () => {
  await assert.rejects(() => runOAuthLogin({
    fetchImpl: mockFetch(),
    openBrowser: async (url) => {
      const [nonce, port] = new URL(url).searchParams.get('qwSurface').split(':').slice(1)
      await fetch(`http://127.0.0.1:${port}/callback?nonce=${encodeURIComponent(nonce)}&error=access_denied`)
    },
    log: () => {},
    timeoutMs: 15000,
  }), /nonce|mismatch|access_denied|timed out/i)
})

test('refreshSession mints new tokens and keeps rotation', async () => {
  const sess = buildSession({
    tenant_url: 'https://t/x', tokens: { access_token: 'a', refresh_token: 'rt-1', id_token: 'i', expires_in: 3600 },
    user_info: {}, client_id: 'c', client_secret: 's', discovery: { idc_region: 'us-east-1' },
  })
  await refreshSession(sess, { fetchImpl: mockFetch() })
  assert.equal(sess.jwt, 'idt-2')
  assert.equal(sess.tokens.refresh_token, 'rt-2') // rotated token persisted
  assert.equal(sess.tokens.access_token, 'at-2')
})

test('DirectBackend: sncs shape + refresh-on-expiry writes back', async () => {
  const dir = tmp()
  try {
    const store = createSessionStore({ dir })
    const sess = buildSession({
      tenant_url: 'https://t/x',
      tokens: { access_token: 'a', refresh_token: 'rt-1', id_token: 'i', expires_in: 10 }, // ~expired
      user_info: { user_arn: 'arn:u' }, discovery: { idc_region: 'us-east-1' },
      client_id: 'c', client_secret: 's',
    })
    store.set('k', sess)
    const backend = createDirectBackend({
      load: () => store.get('k'),
      persist: (s) => store.set('k', s),
      fetchImpl: mockFetch(),
    })
    assert.equal(backend.direct, true)
    assert.equal(await backend.outpostId(), '')
    const got = await backend.session()
    assert.equal(got.id_token, 'idt-2') // refreshed via refresh_token grant
    assert.equal(got.ws_host, 't')
    assert.equal(got.actor_arn, 'arn:u')
    // rotated refresh token persisted to the file
    assert.equal(store.get('k').tokens.refresh_token, 'rt-2')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('DirectBackend errors 401/auth when session is missing', async () => {
  const backend = createDirectBackend({ load: () => null })
  await assert.rejects(() => backend.ensure(), OAuthError)
  await assert.rejects(() => backend.session(), /missing|again/i)
})

test('keyForSession is stable per account', () => {
  const a = keyForSession({ email: 'u@x.com' })
  const b = keyForSession({ email: 'u@x.com' })
  const c = keyForSession({ email: 'v@x.com' })
  assert.equal(a, b)
  assert.notEqual(a, c)
  assert.ok(a.startsWith('oauth-'))
})
