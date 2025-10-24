# 🔄 COMPLETE CONNECTOR FLOW - Gmail Example

## **Visual Flow Diagram**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          USER PERSPECTIVE                                    │
└─────────────────────────────────────────────────────────────────────────────┘

Step 1: Click "Connect" Button
   ↓
Step 2: Popup Window Opens (Gmail OAuth)
   ↓
Step 3: User Authorizes Gmail Access
   ↓
Step 4: Popup Closes Automatically
   ↓
Step 5: Button Changes to "Connected" (Green)
   ↓
Step 6: Click "Sync Now" Button
   ↓
Step 7: Data Syncs from Gmail
   ↓
Step 8: Chat Can Access Gmail Data
```

---

## **🎯 DETAILED STEP-BY-STEP FLOW**

### **STEP 1: User Clicks "Connect Gmail"** 🖱️

**Location:** Data Sources Panel

**What User Sees:**
- Button says "Connect" (default color)
- Button is clickable

**What Happens in Code:**

**Frontend (`DataSourcesPanel.tsx` line 292):**
```typescript
const handleConnect = async (provider: string) => {
  setConnecting(provider); // Show loading state
  
  // Call backend to initiate connection
  const response = await fetch(`${config.apiUrl}/api/connectors/initiate`, {
    method: 'POST',
    body: JSON.stringify({
      provider: 'google-mail', // Gmail
      user_id: user?.id,
      session_token: sessionToken
    })
  });
}
```

**Backend (`fastapi_backend.py` line 12983):**
```python
@app.post("/api/connectors/initiate")
async def initiate_connector(req: dict):
    provider = req.get('provider')  # 'google-mail'
    user_id = req.get('user_id')
    
    # Map provider to Nango integration ID
    integ = NANGO_GMAIL_INTEGRATION_ID  # 'gmail'
    
    # Create Nango Connect session
    session = await nango.create_connect_session(
        end_user={'id': user_id},
        allowed_integrations=[integ]
    )
    
    # Return OAuth URL to frontend
    return {
        'connect_session': {
            'url': session['url']  # Gmail OAuth URL
        }
    }
```

---

### **STEP 2: Popup Window Opens** 🪟

**What User Sees:**
- New window opens (600x700px)
- Gmail login page appears
- URL is from Google OAuth

**What Happens in Code:**

**Frontend (`DataSourcesPanel.tsx` line 332):**
```typescript
const connectUrl = data?.connect_session?.url;

// Open popup window
const popup = window.open(
  connectUrl,  // Gmail OAuth URL from Nango
  '_blank',
  'width=600,height=700,noopener,noreferrer'
);

// Show toast notification
toast({
  title: 'Connection Started',
  description: 'Complete the authorization in the popup window'
});
```

**What the URL Looks Like:**
```
https://connect.nango.dev/oauth/connect?
  session_token=abc123...
  &integration_id=gmail
  &end_user_id=user_uuid
```

---

### **STEP 3: User Authorizes Gmail** ✅

**What User Sees:**
1. Gmail login page (if not logged in)
2. Permission request screen:
   - "Finley AI wants to access your Gmail"
   - "Read, send, delete emails"
   - "Manage labels and settings"
3. "Allow" button

**What Happens When User Clicks "Allow":**

**Nango (External Service):**
1. Receives authorization code from Google
2. Exchanges code for access token
3. Stores access token securely
4. Creates connection record
5. Sends webhook to your backend (optional)
6. Redirects popup to success page
7. **Popup closes automatically**

---

### **STEP 4: Popup Closes** 🔄

**What User Sees:**
- Popup window closes automatically
- Back to main app
- Brief loading state (3 seconds)

**What Happens in Code:**

**Frontend (`DataSourcesPanel.tsx` line 340-392):**
```typescript
// Poll to detect when popup closes
const pollTimer = setInterval(() => {
  if (popup.closed) {
    clearInterval(pollTimer);
    
    // Wait 3 seconds for Nango to process
    setTimeout(async () => {
      // Verify connection was created
      const verifyResponse = await fetch(
        `${config.apiUrl}/api/connectors/verify-connection`,
        {
          method: 'POST',
          body: JSON.stringify({
            user_id: user?.id,
            provider: 'google-mail',
            session_token: sessionToken
          })
        }
      );
      
      if (verifyResponse.ok) {
        // Show success toast
        toast({
          title: 'Connected!',
          description: 'Gmail connected successfully'
        });
      }
      
      // Refresh connections list
      const response = await fetch(
        `${config.apiUrl}/api/connectors/user-connections`
      );
      const data = await response.json();
      setConnections(data.connections); // Update UI
    }, 3000);
  }
}, 500); // Check every 500ms
```

**Backend (`fastapi_backend.py` line 13060):**
```python
@app.post("/api/connectors/verify-connection")
async def verify_connection(req: dict):
    user_id = req.get('user_id')
    provider = req.get('provider')  # 'google-mail'
    
    # Map to integration ID
    integration_id = NANGO_GMAIL_INTEGRATION_ID  # 'gmail'
    
    # Generate connection_id (Nango format)
    connection_id = f"{user_id}_{integration_id}"
    
    # Lookup connector_id from database
    connector = supabase.table('connectors')\
        .select('id')\
        .eq('integration_id', integration_id)\
        .limit(1)\
        .execute()
    
    # Create/update user_connection record
    supabase.table('user_connections').upsert({
        'user_id': user_id,
        'nango_connection_id': connection_id,
        'connector_id': connector.data[0]['id'],
        'status': 'active',  # ✅ ACTIVE!
        'sync_frequency_minutes': 60,
        'created_at': datetime.utcnow().isoformat(),
        'updated_at': datetime.utcnow().isoformat()
    }).execute()
    
    return {'status': 'ok', 'connection_id': connection_id}
```

---

### **STEP 5: Button Changes to "Connected"** 🟢

**What User Sees:**
- Button text changes from "Connect" to "Connected"
- Button color changes to **GREEN** (success state)
- "Sync Now" button appears next to it

**What Happens in Code:**

**Frontend (`DataSourcesPanel.tsx` line 377-389):**
```typescript
// Refresh connections list
const response = await fetch(
  `${config.apiUrl}/api/connectors/user-connections`,
  {
    method: 'POST',
    body: JSON.stringify({
      user_id: user?.id,
      session_token: sessionToken
    })
  }
);

const data = await response.json();
setConnections(data.connections); // ← This triggers UI update
```

**UI Rendering Logic:**
```typescript
// Check if this integration is connected
const isConnected = connections.some(
  conn => conn.connector_id === integration.id && 
          conn.status === 'active'
);

// Render button
{isConnected ? (
  <Button variant="outline" className="bg-green-500/20 text-green-400">
    ✓ Connected
  </Button>
) : (
  <Button onClick={() => handleConnect(integration.provider_key)}>
    Connect
  </Button>
)}
```

**Button States:**
- **Default**: "Connect" (gray/default color)
- **Loading**: "Connecting..." (disabled, spinner)
- **Connected**: "✓ Connected" (green background, green text)
- **Error**: "Connect" (red border if failed)

---

### **STEP 6: User Clicks "Sync Now"** 🔄

**What User Sees:**
- "Sync Now" button next to "Connected" button
- Button shows loading spinner when clicked
- Toast notification: "Syncing Gmail data..."

**What Happens in Code:**

**Frontend (`DataSourcesPanel.tsx` line 409):**
```typescript
const handleSync = async (connectionId: string, integrationId: string) => {
  setSyncing(connectionId); // Show loading state
  
  const response = await fetch(`${config.apiUrl}/api/connectors/sync`, {
    method: 'POST',
    body: JSON.stringify({
      connection_id: connectionId,
      integration_id: integrationId,
      user_id: user?.id,
      session_token: sessionToken
    })
  });
  
  if (response.ok) {
    toast({
      title: 'Sync Started',
      description: 'Gmail data is being synced...'
    });
  }
}
```

**Backend (`fastapi_backend.py` - Gmail Sync):**
```python
@app.post("/api/connectors/sync")
async def sync_connector(req: dict):
    connection_id = req.get('connection_id')
    integration_id = req.get('integration_id')
    user_id = req.get('user_id')
    
    # Route to appropriate sync function
    if integration_id == NANGO_GMAIL_INTEGRATION_ID:
        await sync_gmail_attachments(user_id, connection_id)
    
    return {'status': 'success', 'message': 'Sync started'}

async def sync_gmail_attachments(user_id: str, connection_id: str):
    # 1. Get Gmail messages with attachments
    messages = await nango.list_gmail_messages(
        provider_config_key='gmail',
        connection_id=connection_id,
        q='has:attachment newer_than:365d'
    )
    
    # 2. Download each attachment
    for msg in messages:
        attachment_bytes = await nango.get_gmail_attachment(
            provider_config_key='gmail',
            connection_id=connection_id,
            message_id=msg['id'],
            attachment_id=attachment['id']
        )
        
        # 3. Store in external_items table
        supabase.table('external_items').insert({
            'user_id': user_id,
            'connection_id': connection_id,
            'item_type': 'gmail_attachment',
            'external_id': msg['id'],
            'payload': {
                'filename': attachment['filename'],
                'size': attachment['size'],
                'data': base64.b64encode(attachment_bytes)
            },
            'status': 'pending'
        }).execute()
    
    # 4. Process through unified pipeline
    # (This happens in background job)
    await process_external_items(user_id, connection_id)
```

---

### **STEP 7: Data Syncs from Gmail** 📥

**What Happens (Background Process):**

**Backend (`fastapi_backend.py` - Processing Pipeline):**
```python
async def process_external_items(user_id: str, connection_id: str):
    # 1. Get pending items
    items = supabase.table('external_items')\
        .select('*')\
        .eq('user_id', user_id)\
        .eq('connection_id', connection_id)\
        .eq('status', 'pending')\
        .execute()
    
    for item in items.data:
        # 2. Convert to CSV format
        csv_data = _convert_api_data_to_csv_format(
            data=[item['payload']],
            source='gmail'
        )
        
        # 3. Process through ExcelProcessor
        result = await _process_api_data_through_pipeline(
            user_id=user_id,
            csv_bytes=csv_data,
            filename=f"gmail_{item['external_id']}.csv",
            source='gmail'
        )
        
        # 4. Data now in raw_events table!
        # - Platform detected: Gmail
        # - Document classified: Invoice/Receipt/etc.
        # - Entities extracted: Vendor names, amounts
        # - Currency normalized: All to USD
        # - Duplicates detected: Skipped if duplicate
        
        # 5. Mark item as processed
        supabase.table('external_items')\
            .update({'status': 'processed'})\
            .eq('id', item['id'])\
            .execute()
```

**What Gets Stored in Database:**

**Table: `raw_events`**
```sql
INSERT INTO raw_events (
    user_id,
    source_platform,  -- 'Gmail'
    source_filename,  -- 'gmail_msg_12345.csv'
    payload,          -- {vendor: 'Acme Corp', amount: 1500, ...}
    classification_metadata,  -- {kind: 'invoice', confidence: 0.95}
    entities,         -- [{type: 'vendor', name: 'Acme Corp'}]
    ingest_ts,        -- '2025-10-24T10:00:00Z'
    status            -- 'processed'
)
```

---

### **STEP 8: Chat Can Access Gmail Data** 💬

**What User Sees:**
- User asks: "Show me my Gmail invoices"
- Chat responds with actual data!

**What Happens in Code:**

**Backend (`intelligent_chat_orchestrator.py` line 1010-1018):**
```python
async def _fetch_user_data_context(self, user_id: str) -> str:
    # Query user's connections
    connections = self.supabase.table('user_connections')\
        .select('*')\
        .eq('user_id', user_id)\
        .eq('status', 'active')\
        .execute()
    
    # Query transactions (includes Gmail data!)
    events = self.supabase.table('raw_events')\
        .select('id, source_platform, ingest_ts, payload')\
        .eq('user_id', user_id)\
        .gte('ingest_ts', ninety_days_ago)\
        .order('ingest_ts', desc=True)\
        .limit(1000)\
        .execute()
    
    # Build context for AI
    context = f"""
    CONNECTED DATA SOURCES: Gmail, QuickBooks, Xero
    TOTAL TRANSACTIONS: {len(events.data)}
    PLATFORMS DETECTED: Gmail, QuickBooks, Xero
    
    FINANCIAL SUMMARY:
    - Total Revenue: $125,432.00
    - Total Expenses: $89,127.00
    - Net Income: $36,305.00
    """
    
    return context
```

**AI Response:**
```
💰 I found 23 invoices from your Gmail in the last 90 days!

**Key Findings:**
• Total invoice value: $45,230
• Top vendor: Acme Corp ($12,500 - 5 invoices)
• Average invoice: $1,966
• 3 invoices overdue >30 days

**🎯 Recommended Actions:**
1. Follow up on overdue invoices ($8,450 total)
2. Acme Corp is your biggest vendor - negotiate volume discount
3. Set up automatic payment reminders

**What's next?** Want me to analyze payment patterns?
```

---

## **🎨 BUTTON COLOR STATES**

### **Connect Button:**
```
┌─────────────────────────────────────────────┐
│ State          │ Color      │ Text          │
├─────────────────────────────────────────────┤
│ Default        │ Gray       │ "Connect"     │
│ Hover          │ Light Gray │ "Connect"     │
│ Loading        │ Gray       │ "Connecting..." │
│ Connected      │ GREEN      │ "✓ Connected" │
│ Error          │ Red border │ "Connect"     │
└─────────────────────────────────────────────┘
```

### **Sync Button:**
```
┌─────────────────────────────────────────────┐
│ State          │ Color      │ Text          │
├─────────────────────────────────────────────┤
│ Default        │ Blue       │ "Sync Now"    │
│ Syncing        │ Blue       │ "Syncing..."  │
│ Success        │ Green      │ "Synced ✓"    │
│ Error          │ Red        │ "Retry Sync"  │
└─────────────────────────────────────────────┘
```

---

## **📊 DATA FLOW SUMMARY**

```
┌──────────────┐
│ User Clicks  │
│ "Connect"    │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ Frontend calls       │
│ /api/connectors/     │
│ initiate             │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Backend creates      │
│ Nango session        │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Popup opens with     │
│ Gmail OAuth          │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ User authorizes      │
│ Gmail access         │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Nango stores token   │
│ & creates connection │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Popup closes         │
│ automatically        │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Frontend verifies    │
│ connection           │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Backend creates      │
│ user_connection      │
│ record (status:      │
│ 'active')            │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Frontend refreshes   │
│ connections list     │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Button changes to    │
│ "✓ Connected"        │
│ (GREEN)              │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ User clicks          │
│ "Sync Now"           │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Backend fetches      │
│ Gmail data via       │
│ Nango API            │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Data stored in       │
│ external_items       │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Processed through    │
│ unified pipeline     │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Data in raw_events   │
│ table (enriched)     │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Chat can access      │
│ Gmail data!          │
└──────────────────────┘
```

---

## **🔍 TROUBLESHOOTING**

### **Button Stays "Connect" (Doesn't Turn Green)**

**Possible Causes:**
1. Webhook from Nango failed
2. `verify-connection` endpoint not called
3. Database connection failed
4. User closed popup before authorizing

**Solution:**
- Frontend polls popup closure and calls `verify-connection`
- This creates the connection record even if webhook fails

### **Button Turns Green But No Data**

**Possible Causes:**
1. User hasn't clicked "Sync Now"
2. Sync is running in background
3. No attachments found in Gmail

**Solution:**
- Click "Sync Now" button
- Wait 30-60 seconds for sync to complete
- Check sync history

### **Sync Fails**

**Possible Causes:**
1. Nango token expired
2. Gmail API rate limit
3. Network error

**Solution:**
- Disconnect and reconnect
- Wait and retry
- Check backend logs

---

## **✅ SUCCESS INDICATORS**

**User Knows Connection Worked When:**
1. ✅ Popup closes automatically
2. ✅ Toast shows "Connected!"
3. ✅ Button turns GREEN with "✓ Connected"
4. ✅ "Sync Now" button appears
5. ✅ After sync, chat can answer questions about Gmail data

---

**This is the complete end-to-end flow!** 🎉
