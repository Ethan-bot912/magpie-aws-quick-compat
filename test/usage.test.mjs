import { test } from 'node:test'
import assert from 'node:assert/strict'
import { quickUsage, usageFromEntitlements, planName } from '../lib/usage.mjs'

const RESP = {
  entitlements: [
    { unitType: 'AUTOMATION_AGENT_HOUR_UNIT', numMinsAvailable: 529, numMinsProvisioned: 600 },
    { unitType: 'RESEARCH_AGENT_HOUR_UNIT', numMinsAvailable: 529, numMinsProvisioned: 600 },
  ],
  subscriptionMetadata: { subscriptionType: 'QUICK_LITE_FREE_TRIAL', freeTrialMetadata: { status: 'ACTIVE', expiryDate: 1797865753000 } },
}

test('planName humanizes subscription types', () => {
  assert.equal(planName('QUICK_LITE_FREE_TRIAL'), 'Quick Lite (free trial)')
  assert.equal(planName('QUICK_PRO'), 'Quick Pro')
  assert.equal(planName(''), '')
})

test('entitlements map to one agent-hours window per pool, plan and trial expiry', () => {
  const u = usageFromEntitlements(RESP, { user: 'me@example.com', now: new Date(2026, 10, 22) })
  assert.equal(u.plan, 'Quick Lite (free trial)')
  assert.equal(u.until, new Date(1797865753000).toISOString())
  assert.equal(u.user, 'me@example.com')
  assert.equal(u.windows.length, 1) // identical pools collapse
  const w = u.windows[0]
  assert.equal(w.name, 'Agent hours')
  assert.equal(w.amount, 1.18)
  assert.equal(w.limit, 10)
  assert.equal(w.unit, 'h')
  assert.ok(Math.abs(w.used - (71 / 600) * 100) < 1e-9)
  assert.equal(w.resetsAt, new Date(2026, 11, 1).toISOString())
})

test('quickUsage posts the qbs-event and reads the account from the actor ARN', async () => {
  let seen
  const fetchImpl = async (url, init) => {
    seen = { url, init }
    return new Response(JSON.stringify({ errorMessage: null, eventResponse: { getAvailableEntitlementUnitsResponse: RESP } }), { status: 200 })
  }
  const sess = { id_token: 'tok', tenant_url: 'https://tenant.example/', aws_account_id: '123456789012', actor_arn: 'arn:aws:quicksight:us-east-1:123456789012:user/default/me@example.com' }
  const u = await quickUsage(async () => sess, { fetchImpl })
  assert.equal(seen.url, 'https://tenant.example/integration/quick-work/qbs-event')
  assert.equal(seen.init.headers.Authorization, 'Bearer tok')
  assert.deepEqual(JSON.parse(seen.init.body), { eventType: 'getAvailableEntitlementUnits', eventBody: { getAvailableEntitlementUnits: { awsAccountId: '123456789012' } } })
  assert.equal(u.user, 'me@example.com')
  assert.equal(u.windows[0].limit, 10)
})

test('quickUsage reports expired sign-in on 401 and errors without leaking the token', async () => {
  const sess = { id_token: 'secret-token', tenant_url: 'https://t.example' }
  const expired = await quickUsage(async () => sess, { fetchImpl: async () => new Response('', { status: 401 }) })
  assert.equal(expired.signIn, 'expired')
  const bad = await quickUsage(async () => sess, { fetchImpl: async () => new Response(JSON.stringify({ errorMessage: 'boom' }), { status: 400 }) })
  assert.match(bad.error, /boom/)
  assert.ok(!JSON.stringify(bad).includes('secret-token'))
})
