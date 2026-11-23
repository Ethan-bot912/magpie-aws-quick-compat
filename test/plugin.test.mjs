import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { AwsQuickAuthPlugin } from '../index.mjs'
import { rememberDiscovery } from '../lib/oauth.mjs'

const tmp = () => mkdtempSync(path.join(tmpdir(), 'plugin-test-'))

/** Mock fetch answering the OIDC endpoints the login flow calls. */
const mockFetch = async (url) => {
  const u = String(url)
  const reply = (data, status = 200) => new Response(JSON.stringify(data), { status, headers: { 'content-type': 'application/json' } })
  if (u.endsWith('/client/register')) return reply({ clientId: 'cid-1', clientSecret: 'csec-1' })
  if (u.endsWith('/token')) return reply({ accessToken: 'at-1', refreshToken: 'rt-1', idToken: 'idt-1', expiresIn: 3600 })
  if (u.endsWith('/integration/quick-work/user-info')) return reply({ email: 'u@x.com' })
  return reply({ error: 'unknown' }, 404)
}

test('auth.methods: two browser-OAuth methods + the local-credentials api method', async () => {
  const dir = tmp()
  try {
    const plugin = await AwsQuickAuthPlugin(null, { sessionDir: dir, log: () => {}, fetchImpl: mockFetch })
    const methods = plugin.auth.methods
    assert.equal(methods.length, 3)
    assert.equal(methods[0].type, 'oauth')
    assert.equal(methods[1].type, 'oauth')
    assert.equal(methods[2].type, 'api')
    assert.equal(methods[2].keyless, true)
    assert.match(methods[2].labels?.zh ?? methods[2].label, /本地凭据/)

    // method 1 (copy sign-in link): magpie shows a paste field, opens nothing
    const ac1 = new AbortController()
    const r1 = await methods[1].authorize(undefined, { signal: ac1.signal })
    assert.equal(r1.opened, true)
    assert.equal(r1.method, 'code')
    assert.ok(r1.url.includes('qwSurface='), `sign-in url: ${r1.url}`)
    assert.equal(typeof r1.callback, 'function')
    assert.ok(r1.instructions)

    // method 0 (browser OAuth): no opened flag, magpie opens the browser itself
    const ac0 = new AbortController()
    const r0 = await methods[0].authorize(undefined, { signal: ac0.signal })
    assert.notEqual(r0.opened, true)
    assert.equal(r0.method, 'auto')
    assert.ok(r0.url.includes('qwSurface='), `sign-in url: ${r0.url}`)
    assert.equal(typeof r0.callback, 'function')

    // aborting the sign-in signal cancels the wait; callback reports it
    ac1.abort()
    const c1 = await r1.callback('whatever', { signal: ac1.signal })
    assert.equal(c1.type, 'failed')
    assert.equal(c1.error, '登录已取消')
    ac0.abort()
    const c0 = await r0.callback(undefined, { signal: ac0.signal })
    assert.equal(c0.type, 'failed')
    assert.equal(c0.error, '登录已取消')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('copy-link method uses a saved discovery to skip the discovery hop', async () => {
  const dir = tmp()
  try {
    const plugin = await AwsQuickAuthPlugin(null, { sessionDir: dir, log: () => {}, fetchImpl: mockFetch })
    const method = plugin.auth.methods[1]
    rememberDiscovery(dir, {
      issuer_url: 'https://idc.example.com/issuer',
      tenant_url: 'https://tenant.example.com',
      account: 'acct-1',
      idc_instance_id: 'inst-1',
      idc_region: 'us-east-1',
    })
    const ac = new AbortController()
    const r = await method.authorize(undefined, { signal: ac.signal })
    assert.equal(r.method, 'code')
    assert.equal(r.opened, true)
    assert.ok(r.url.startsWith('https://oidc.us-east-1.amazonaws.com/authorize'), r.url)
    assert.match(r.instructions, /已记住上次的账号信息/)
    ac.abort()
    const c = await r.callback('whatever', { signal: ac.signal })
    assert.equal(c.type, 'failed')
    assert.equal(c.error, '登录已取消')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
