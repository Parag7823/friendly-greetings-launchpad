# Real-Time File Processing Updates Implementation
## Phase 1 & 2: Backend Infrastructure + Frontend State Management

---

## 📋 Overview

This implementation enables real-time, conversational file processing updates. Users receive friendly, human-readable status messages as their files are processed, creating a premium, responsive experience.

### Architecture
```
Backend (FastAPI)
    ↓
[File Processing] → [Friendly Message Generation] → [Socket.IO Emit]
    ↓
WebSocket (Socket.IO)
    ↓
Frontend (React)
    ↓
[WebSocket Hook] → [Zustand Store] → [UI Components]
```

---

## ✅ Phase 1: Backend Infrastructure

### Step 1.1: WebSocket Events (Already Implemented)

**Status:** ✅ COMPLETE

**File:** `core_infrastructure/fastapi_backend_v2.py` (lines 698-821)

**What exists:**
- `SocketIOWebSocketManager` class handles all WebSocket routing
- `send_update(job_id, data)` emits to Socket.IO rooms
- Job-based room system enables targeted updates
- Redis adapter for multi-server scaling

**Key Methods:**
```python
class SocketIOWebSocketManager:
    async def send_update(self, job_id: str, data: Dict[str, Any]):
        """Send update via Socket.IO to job room"""
        await self.merge_job_state(job_id, data)
        await sio.emit('job_update', data, room=job_id)
    
    async def send_error(self, job_id: str, error_message: str):
        """Send error via Socket.IO"""
        await sio.emit('job_error', payload, room=job_id)
```

---

### Step 1.2: Conversational Message Generation

**Status:** ✅ COMPLETE

**File:** `core_infrastructure/utils/helpers.py` (lines 204-314)

**What was built:**

#### Function: `generate_friendly_status(step: str, context: Optional[Dict] = None) -> str`

Converts technical processing steps into human-readable messages using:
1. **Fast Path** (Predefined Messages): No LLM needed, instant response
2. **LLM Path** (Groq + Instructor): For unknown steps, generates custom messages

**Predefined Messages (18 total):**
```python
{
    'initializing_streaming': "Getting ready to read your file...",
    'duplicate_check': "Checking if I've seen this file before...",
    'field_detection': "Analyzing the structure of your data...",
    'platform_detection': "Figuring out where this data came from...",
    'document_classification': "Understanding what type of document this is...",
    'starting_transaction': "Setting up secure storage for your data...",
    'storing': "Saving your file details...",
    'extracting': "Reading through your data...",
    'processing_decision': "Processing your request...",
    'duplicate_found': "Found an exact match - I've processed this before",
    'near_duplicate_found': "Found a similar file - let me compare them",
    'content_duplicate_found': "This data overlaps with something I already have",
    'delta_analysis_complete': "Spotted the differences in your data",
    'entity_resolution': "Matching entities across your data...",
    'classification': "Categorizing your transactions...",
    'enrichment': "Adding context to your data...",
    'complete': "Done! I've processed your file successfully",
    'error': "Oops, something went wrong"
}
```

**Context-Aware Enhancement:**
```python
# Example 1: With filename
generate_friendly_status('duplicate_check', {'filename': 'expenses.csv'})
# Returns: "Checking if I've seen expenses.csv before..."

# Example 2: With row counts
generate_friendly_status('delta_analysis_complete', {
    'new_rows': 42,
    'existing_rows': 156
})
# Returns: "Spotted the differences: 42 new rows, 156 I already know"

# Example 3: With similarity score
generate_friendly_status('near_duplicate_found', {'similarity_score': 0.87})
# Returns: "Found a 87% match with something I processed earlier"
```

**LLM Integration (Groq + Instructor):**
```python
import instructor

client = get_groq_client()
client_with_instructor = instructor.from_groq(client)

response = client_with_instructor.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{
        "role": "system",
        "content": "Convert technical steps into friendly messages..."
    }, {
        "role": "user",
        "content": f"Step: {step}, Context: {context_str}"
    }],
    max_tokens=50
)
```

**Error Handling:**
- ImportError: Falls back to predefined messages if instructor not installed
- LLM Error: Uses generic fallback message
- Non-blocking: Errors don't interrupt file processing

---

### Step 1.3: WebSocket Progress Helper

**Status:** ✅ COMPLETE

**File:** `core_infrastructure/utils/helpers.py` (lines 317-366)

**Function:** `send_websocket_progress(manager, job_id, step, progress, context, extra_data)`

**Purpose:** Single function to emit progress with friendly messages

**Usage:**
```python
# Before (verbose):
message = generate_friendly_status('duplicate_check', {'filename': 'file.csv'})
await manager.send_update(job_id, {
    "step": "duplicate_check",
    "message": message,
    "progress": 15
})

# After (clean):
await send_websocket_progress(
    manager, job_id, 'duplicate_check', 15,
    context={'filename': 'file.csv'}
)
```

**Payload Structure:**
```json
{
  "step": "duplicate_check",
  "message": "Checking if I've seen expenses.csv before...",
  "progress": 15,
  "timestamp": "2025-01-24T10:30:45.123Z",
  "extra": {
    "filename": "expenses.csv",
    "duplicate_info": {...}
  }
}
```

---

## ✅ Phase 2: Frontend State Management

### Step 2.1: File Status Store (Zustand)

**Status:** ✅ COMPLETE

**File:** `src/stores/useFileStatusStore.ts` (NEW)

**What was built:**

#### Store Structure:
```typescript
interface ProcessingStep {
  step: string;
  message: string;
  status: 'in_progress' | 'complete' | 'error';
  timestamp: number;
  progress?: number;
  extra?: Record<string, any>;
}

interface FileStatus {
  fileId: string;
  filename?: string;
  steps: ProcessingStep[];
  currentStep?: string;
  overallProgress: number;
  startedAt: number;
  completedAt?: number;
  error?: string;
}
```

#### Store Actions:
```typescript
const store = useFileStatusStore();

// Add a processing step
store.addStep(fileId, {
  step: 'duplicate_check',
  message: 'Checking if I\'ve seen this file before...',
  status: 'in_progress',
  timestamp: Date.now(),
  progress: 15
});

// Update overall progress
store.updateProgress(fileId, 25);

// Mark as complete
store.markComplete(fileId);

// Set error
store.setError(fileId, 'Processing failed');

// Clear specific file
store.clearStatus(fileId);

// Clear all
store.clearAll();
```

#### Store Queries:
```typescript
// Get specific file status
const status = store.getStatus(fileId);

// Get all statuses
const all = store.getAllStatuses();

// Get active files (in-progress)
const active = store.getActiveFiles();

// Get completed files
const completed = store.getCompletedFiles();
```

**Features:**
- ✅ Multiple concurrent files
- ✅ Step history tracking
- ✅ Progress calculation
- ✅ Error handling
- ✅ Auto-cleanup support
- ✅ TypeScript support

---

### Step 2.2: WebSocket Connection Hook

**Status:** ✅ COMPLETE

**File:** `src/hooks/useFileStatusSocket.ts` (NEW)

**What was built:**

#### Hook: `useFileStatusSocket(userId, sessionToken)`

**Purpose:** Connect to backend Socket.IO and update store in real-time

**Usage:**
```typescript
import { useFileStatusSocket } from '@/hooks/useFileStatusSocket';
import { useAuth } from '@/components/AuthProvider';

export const MyComponent = () => {
  const { user } = useAuth();
  const { isConnected, emit } = useFileStatusSocket(
    user?.id,
    sessionToken
  );

  return (
    <div>
      {isConnected ? '✅ Connected' : '⚠️ Disconnected'}
    </div>
  );
};
```

**Event Handlers:**
```typescript
// Listens for:
socket.on('file_progress', handleFileProgress);
socket.on('job_update', handleFileProgress);      // Alias
socket.on('job_complete', handleJobComplete);
socket.on('job_error', handleJobError);
```

**Auto-Updates Store:**
```typescript
// When backend emits:
{
  "fileId": "file-123",
  "step": "duplicate_check",
  "message": "Checking if I've seen expenses.csv before...",
  "progress": 15
}

// Hook automatically calls:
store.addStep('file-123', {
  step: 'duplicate_check',
  message: 'Checking if I\'ve seen expenses.csv before...',
  status: 'in_progress',
  timestamp: Date.now(),
  progress: 15
});
```

**Connection Features:**
- ✅ Auto-reconnection with exponential backoff
- ✅ Multiple transport fallback (WebSocket → Polling)
- ✅ Auth via query parameters
- ✅ Error handling
- ✅ Auto-cleanup on unmount

**Connection Configuration:**
```typescript
const socket = io(wsUrl, {
  query: {
    user_id: userId,
    session_token: sessionToken,
  },
  reconnection: true,
  reconnectionDelay: 1000,
  reconnectionDelayMax: 5000,
  reconnectionAttempts: 5,
  transports: ['websocket', 'polling'],
});
```

---

## 🔗 Integration Points

### Backend → Frontend Flow:

1. **Backend processes file**
   ```python
   await send_websocket_progress(
       manager, job_id, 'duplicate_check', 15,
       context={'filename': 'expenses.csv'}
   )
   ```

2. **Generates friendly message**
   ```
   "Checking if I've seen expenses.csv before..."
   ```

3. **Emits via Socket.IO**
   ```json
   {
     "step": "duplicate_check",
     "message": "Checking if I've seen expenses.csv before...",
     "progress": 15,
     "timestamp": "2025-01-24T10:30:45.123Z"
   }
   ```

4. **Frontend receives via WebSocket**
   ```typescript
   socket.on('file_progress', (data) => {
     // data = { step, message, progress, timestamp }
   })
   ```

5. **Hook updates store**
   ```typescript
   store.addStep(fileId, {
     step: 'duplicate_check',
     message: 'Checking if I\'ve seen expenses.csv before...',
     status: 'in_progress',
     timestamp: Date.now(),
     progress: 15
   })
   ```

6. **UI components subscribe to store**
   ```typescript
   const status = useFileStatusStore((state) => state.getStatus(fileId));
   // Re-renders automatically when status changes
   ```

---

## 📦 Dependencies

### Backend
- ✅ `groq` - LLM API client
- ✅ `instructor` - Structured AI responses
- ✅ `python-socketio` - WebSocket server
- ✅ `redis` - State persistence

### Frontend
- ✅ `zustand` - State management
- ✅ `socket.io-client` - WebSocket client
- ✅ `react` - UI framework

---

## 🚀 Next Steps (Phase 2.3+)

### Step 2.3: UI Components
- [ ] FileStatusDisplay component
- [ ] FileProcessingTimeline component
- [ ] ProgressBar with friendly messages
- [ ] ErrorBoundary for error states

### Step 2.4: Integration Testing
- [ ] E2E test: File upload → Real-time updates
- [ ] WebSocket reconnection test
- [ ] Store state verification
- [ ] Error handling test

### Step 3: Advanced Features
- [ ] Pause/Resume processing
- [ ] Cancel processing
- [ ] Batch file processing
- [ ] Processing history/analytics

---

## 📊 Current Implementation Status

| Component | Status | File |
|-----------|--------|------|
| SocketIOWebSocketManager | ✅ | `fastapi_backend_v2.py` |
| generate_friendly_status | ✅ | `helpers.py` |
| send_websocket_progress | ✅ | `helpers.py` |
| useFileStatusStore | ✅ | `useFileStatusStore.ts` |
| useFileStatusSocket | ✅ | `useFileStatusSocket.ts` |
| UI Components | ⏳ | TBD |
| Integration Tests | ⏳ | TBD |

---

## 🎯 Benefits

✅ **User Experience**
- Real-time feedback during file processing
- Friendly, conversational messages
- Progress visibility
- Error clarity

✅ **Developer Experience**
- Single function to emit progress
- Automatic message generation
- Type-safe state management
- Easy to extend

✅ **Performance**
- Fast predefined message path (no LLM)
- Async WebSocket (non-blocking)
- Efficient state updates
- Auto-reconnection

✅ **Reliability**
- Error handling at every layer
- Graceful fallbacks
- State persistence
- Multi-server support via Redis
