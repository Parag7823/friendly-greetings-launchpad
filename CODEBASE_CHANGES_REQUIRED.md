# Codebase Changes Required After Schema Migrations

**Date**: December 3, 2025  
**Status**: Required for full functionality  
**Priority**: HIGH - Must implement before deploying Tasks #4, #5, #6

---

## Overview

The 2 schema migrations fix database issues but require **3 codebase changes** to fully enable functionality:

1. **Backend Code**: Populate `sync_run_id` when inserting into `external_items`
2. **Backend Code**: Populate `created_at` when inserting into `webhook_events`
3. **Frontend Code**: No changes needed (schema is backward compatible)

---

## Change #1: Populate `sync_run_id` in `external_items` Inserts

### Location
File: `core_infrastructure/fastapi_backend_v2.py`

### Current Code Pattern
All sync functions (`_gmail_sync_run`, `_dropbox_sync_run`, etc.) insert into `external_items` without `sync_run_id`.

**Example** (around line 10143):
```python
ext_res = supabase.table('external_items').select('id').eq('user_id', user_id).eq('hash', file_hash).limit(1).execute()
```

### Required Changes

**Find all locations** where `external_items` is inserted:
```bash
grep -n "table('external_items').insert" core_infrastructure/fastapi_backend_v2.py
```

**Pattern to find**:
```python
supabase.table('external_items').insert({
    'user_id': user_id,
    'user_connection_id': user_connection_id,
    'provider_id': ...,
    'kind': ...,
    'hash': ...,
    'metadata': ...,
    'status': 'fetched'
    # ❌ MISSING: 'sync_run_id': sync_run_id
}).execute()
```

**Fix**: Add `sync_run_id` to every `external_items` insert:
```python
supabase.table('external_items').insert({
    'user_id': user_id,
    'user_connection_id': user_connection_id,
    'provider_id': message_id,  # or file_id, transaction_id, etc.
    'kind': 'email',  # or 'file', 'txn', etc.
    'sync_run_id': sync_run_id,  # ✅ ADD THIS
    'hash': file_hash,
    'metadata': metadata,
    'status': 'fetched'
}).execute()
```

### Affected Functions
- `_gmail_sync_run()` - Gmail email syncing
- `_dropbox_sync_run()` - Dropbox file syncing
- `_gdrive_sync_run()` - Google Drive file syncing
- `_zohomail_sync_run()` - Zoho Mail syncing
- `_quickbooks_sync_run()` - QuickBooks transaction syncing
- `_xero_sync_run()` - Xero transaction syncing
- `_zoho_books_sync_run()` - Zoho Books syncing
- `_stripe_sync_run()` - Stripe payment syncing
- `_razorpay_sync_run()` - Razorpay payment syncing
- `_paypal_sync_run()` - PayPal payment syncing
- Any webhook delta processing functions

### Impact
- ✅ Enables Task #4: Sync Results Endpoint (query by sync_run_id)
- ✅ Enables Task #6: Sync Resume Capability (checkpoint tracking)
- ✅ Enables Task #5: Cross-Provider Duplicate Detection (track source sync)
- ✅ Enables accurate sync statistics

### Backward Compatibility
- ✅ Existing `external_items` rows will have `sync_run_id = NULL` (safe)
- ✅ New rows will have `sync_run_id` populated
- ✅ Queries without `sync_run_id` filter still work

---

## Change #2: Populate `created_at` in `webhook_events` Inserts

### Location
File: `core_infrastructure/fastapi_backend_v2.py`

### Current Code Pattern
Around line 12303, webhook events are inserted without explicit `created_at`:

```python
supabase.table('webhook_events').insert({
    'user_id': user_id or 'unknown',
    'user_connection_id': webhook_user_connection_id,
    'event_type': event_type,
    'payload': payload,
    'signature_valid': signature_valid,
    'received_at': pendulum.now().to_iso8601_string(),
    # ❌ MISSING: 'created_at': pendulum.now().to_iso8601_string()
    'event_id': event_id
}).execute()
```

### Required Changes

**Find the webhook insert** (around line 12303):
```python
supabase.table('webhook_events').insert({
    'user_id': user_id or 'unknown',
    'user_connection_id': webhook_user_connection_id,
    'event_type': event_type,
    'payload': payload,
    'signature_valid': signature_valid,
    'received_at': pendulum.now().to_iso8601_string(),
    'created_at': pendulum.now().to_iso8601_string(),  # ✅ ADD THIS
    'event_id': event_id
}).execute()
```

### Impact
- ✅ Enables proper audit trail (created_at vs received_at vs processed_at)
- ✅ Enables webhook lifecycle tracking
- ✅ Enables compliance/audit requirements

### Backward Compatibility
- ✅ Database migration backfills `created_at = received_at` for existing rows
- ✅ New rows will have explicit `created_at`
- ✅ Queries without `created_at` filter still work

---

## Change #3: Update Sync Functions to Use `sync_run_id`

### Location
File: `core_infrastructure/fastapi_backend_v2.py`

### Pattern
All sync functions receive `sync_run_id` as parameter. Ensure it's passed to `external_items` inserts.

**Example** (Gmail sync function signature):
```python
async def _gmail_sync_run(
    user_id: str,
    connection_id: str,
    sync_run_id: str,  # ← Already available
    ...
):
    # When inserting external_items:
    supabase.table('external_items').insert({
        ...
        'sync_run_id': sync_run_id,  # ✅ Use this parameter
        ...
    }).execute()
```

### Verification Checklist
- [ ] All sync functions pass `sync_run_id` to `external_items.insert()`
- [ ] All webhook delta processing populates `sync_run_id`
- [ ] All webhook events populate `created_at`
- [ ] No breaking changes to existing queries
- [ ] Backward compatible with existing data

---

## Change #4: Frontend - No Changes Needed

The schema changes are **fully backward compatible** with the frontend:

- ✅ New columns have default values
- ✅ Existing queries still work
- ✅ No API contract changes
- ✅ No UI changes required

---

## Verification Steps

### Step 1: Verify Schema Changes Applied
```sql
-- Check webhook_events has created_at
SELECT column_name FROM information_schema.columns 
WHERE table_name = 'webhook_events' AND column_name = 'created_at';
-- Should return: created_at

-- Check external_items has sync_run_id
SELECT column_name FROM information_schema.columns 
WHERE table_name = 'external_items' AND column_name = 'sync_run_id';
-- Should return: sync_run_id
```

### Step 2: Verify Code Changes
```bash
# Find all external_items inserts
grep -n "table('external_items').insert" core_infrastructure/fastapi_backend_v2.py

# Verify each has sync_run_id
# Pattern: should see 'sync_run_id': sync_run_id in each insert
```

### Step 3: Test Sync Functions
```python
# After code changes, test:
1. Start a Gmail sync
2. Verify external_items rows have sync_run_id populated
3. Query: SELECT COUNT(*) FROM external_items WHERE sync_run_id IS NOT NULL
4. Should return > 0
```

### Step 4: Test Webhook Processing
```python
# After code changes, test:
1. Send test webhook
2. Verify webhook_events rows have created_at populated
3. Query: SELECT COUNT(*) FROM webhook_events WHERE created_at IS NOT NULL
4. Should return > 0
```

---

## Timeline

| Phase | Task | Time |
|-------|------|------|
| 1 | Apply Migration 1 (20251203000001) | 5 min |
| 2 | Apply Migration 2 (20251203000002) | 5 min |
| 3 | Update sync functions (Change #1) | 30 min |
| 4 | Update webhook handler (Change #2) | 10 min |
| 5 | Test all sync functions | 30 min |
| 6 | Deploy to production | 10 min |
| **Total** | | **90 min** |

---

## Rollback Plan

If issues occur:

1. **Rollback migrations**:
   ```sql
   -- Drop migration 2
   DROP MIGRATION 20251203000002;
   -- Drop migration 1
   DROP MIGRATION 20251203000001;
   ```

2. **Revert code changes**:
   - Remove `sync_run_id` from `external_items` inserts
   - Remove `created_at` from `webhook_events` inserts

3. **Data integrity**:
   - ✅ No data loss (migrations are additive only)
   - ✅ Existing data unaffected
   - ✅ Safe to rollback anytime

---

## Summary

**Required Codebase Changes**: 2 changes in `fastapi_backend_v2.py`
- Add `sync_run_id` to all `external_items` inserts
- Add `created_at` to `webhook_events` insert

**Frontend Changes**: None (fully backward compatible)

**Time to Implement**: ~40 minutes

**Impact**: Enables Tasks #4, #5, #6 and full audit trail functionality
