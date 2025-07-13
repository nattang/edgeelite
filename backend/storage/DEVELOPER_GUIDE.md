# EdgeElite Storage Backend - Developer Guide

This guide shows how to integrate with the EdgeElite storage and semantic search system.

## 📋 Overview

The storage backend handles:

- **OCR/Audio data ingestion** → SQLite database
- **Session processing** → cleaning + chunking + embeddings
- **Semantic search** → RAG retrieval from FAISS vector store

## 🚀 Quick Setup

```python
# Import the storage interface
from backend.storage.interface import (
    store_raw_ocr_event,
    store_raw_audio_event,
    process_session,
    search_similar
)
```

---

## 📷 1. OCR Data Ingestion

**Function:** `store_raw_ocr_event()`

**When to use:** Every time OCR detects text on screen

### Parameters:

- `session_id` (str): Unique session identifier
- `source` (str): Always `'ocr'`
- `ts` (float): Unix timestamp when OCR occurred
- `text` (str): The extracted text content
- `metadata` (dict, optional): Additional context data

### Example:

```python
# Store OCR text when detected
event_id = store_raw_ocr_event(
    session_id="user_work_session_123",
    source="ocr",
    ts=1703123456.789,
    text="Visual Studio Code - main.py",
    metadata={
        "window": "vscode",
        "confidence": 0.95,
        "coordinates": {"x": 100, "y": 50}
    }
)
print(f"Stored OCR event: {event_id}")
```

---

## 🎤 2. Audio Data Ingestion

**Function:** `store_raw_audio_event()`

**When to use:** When audio transcription is complete (supports batch processing)

### Parameters:

- `session_id` (str): Unique session identifier
- `source` (str): Always `'audio'`
- `audio_data` (list): List of transcription segments

### Audio Data Format:

Each segment must contain:

- `timestamp` (float): Unix timestamp of the audio
- `text` (str): Transcribed text content
- Any additional fields become metadata

### Example:

```python
# Store batch of audio transcriptions
audio_segments = [
    {
        "timestamp": 1703123456.789,
        "text": "Let me open this file and edit the function"
    },
    {
        "timestamp": 1703123461.234,
        "text": "I need to fix this bug in the authentication logic",
        "confidence": 0.92,
        "speaker": "user"
    }
]

event_ids = store_raw_audio_event(
    session_id="user_work_session_123",
    source="audio",
    audio_data=audio_segments
)
print(f"Stored {len(event_ids)} audio events")
```

---

## 🔄 3. Session Processing

**Function:** `process_session()`

**When to use:** When user session ends (triggers the complete processing pipeline)

### What it does:

1. Retrieves all raw OCR/Audio events for the session
2. Creates chronological stream: `[OCR @ 10:15:23] text` / `[AUDIO @ 10:15:24] text`
3. Uses semantic chunking to create coherent segments
4. Generates embeddings and stores in FAISS vector database

### Parameters:

- `session_id` (str): The session to process

### Example:

```python
# Call when user session ends
try:
    node_ids = process_session("user_work_session_123")
    print(f"Session processed successfully! Created {len(node_ids)} chunks")
except Exception as e:
    print(f"Session processing failed: {e}")
```

### Important Notes:

- ⚠️ **Only call once per session** - processing is idempotent
- ⚠️ **Wait until session is complete** - don't process partial sessions
- ✅ **Session must have events** - empty sessions will raise an error

---

## 🔍 4. Semantic Search & Retrieval

**Function:** `search_similar()`

**When to use:** When user asks questions or needs to find relevant past content

### Parameters:

- `query` (str): Natural language search query
- `k` (int): Number of results to return (default: 5)
- `filter` (dict, optional): Metadata filters (reserved for future use)

### Returns:

List of tuples: `(summary, full_content)`

- `summary`: Brief description of the chunk
- `full_content`: Complete chronological text

### Example:

```python
# Search for relevant content
results = search_similar(
    query="authentication code and login functions",
    k=3
)

print(f"Found {len(results)} relevant chunks:")
for i, (summary, content) in enumerate(results, 1):
    print(f"\n{i}. {summary}")
    print(f"Content: {content[:200]}...")
```

### Search Tips:

- 🎯 **Use descriptive queries**: "email client settings" vs "email"
- 🔍 **Mix keywords with context**: "react component error debugging"
- 📊 **Semantic search works across sessions** - finds related content anywhere
- ⚡ **Fast retrieval**: Typically < 100ms response time

---

## 📋 Complete Integration Example

```python
from backend.storage.interface import (
    store_raw_ocr_event,
    store_raw_audio_event,
    process_session,
    search_similar
)
import time

# During user session - store events as they happen
session_id = f"session_{int(time.time())}"

# 1. Store OCR events
store_raw_ocr_event(
    session_id=session_id,
    source="ocr",
    ts=time.time(),
    text="Gmail - Compose New Message"
)

# 2. Store audio events (batch)
audio_data = [
    {"timestamp": time.time(), "text": "Let me send an email to the client"},
    {"timestamp": time.time() + 2, "text": "I need to update them on project status"}
]
store_raw_audio_event(session_id, "audio", audio_data)

# 3. Process when session ends
node_ids = process_session(session_id)
print(f"Session processed: {len(node_ids)} chunks created")

# 4. Later - search for relevant content
results = search_similar("client email communication", k=2)
for summary, content in results:
    print(f"Found: {summary}")
```

---

## 🛠️ Error Handling

### Common Errors:

```python
# Invalid session ID
try:
    process_session("")
except ValueError as e:
    print(f"Error: {e}")  # "Invalid session_id: must be a non-empty string"

# Empty text
try:
    store_raw_ocr_event("session1", "ocr", time.time(), "")
except ValueError as e:
    print(f"Error: {e}")  # "Invalid text: must be a non-empty string"

# Wrong audio format
try:
    store_raw_audio_event("session1", "audio", [{"ts": 123, "text": "hello"}])
except ValueError as e:
    print(f"Error: {e}")  # "must contain 'timestamp' and 'text' keys"
```

---

## 📊 System Stats & Monitoring

```python
from backend.storage.interface import get_system_stats, get_session_stats

# Overall system status
stats = get_system_stats()
print(f"Total sessions: {stats['total_sessions_processed']}")
print(f"Total chunks: {stats['total_nodes']}")

# Specific session info
session_stats = get_session_stats("session_123")
print(f"OCR events: {session_stats['ocr_events']}")
print(f"Audio events: {session_stats['audio_events']}")
print(f"Processed: {session_stats['is_processed']}")
```

---

## 🚨 Best Practices

### ✅ Do:

- Use meaningful session IDs (e.g., `user_123_work_20240115`)
- Include metadata for debugging (window names, confidence scores)
- Process sessions promptly after they end
- Use descriptive search queries

### ❌ Don't:

- Process the same session multiple times
- Store empty or meaningless text
- Use very short session IDs
- Search with empty queries

### 🔄 Data Flow:

```
OCR/Audio Events → store_raw_*_event() → SQLite Database
                                                ↓
User Session Ends → process_session() → Cleaning → Chunking → Embeddings → FAISS
                                                                              ↓
User Asks Question → search_similar() → Semantic Search → Relevant Chunks
```

---

## 🧪 Testing

```python
# Quick test of the complete pipeline
from backend.storage.test_files.simple_e2e_test import test_complete_pipeline

# Run end-to-end test
success = test_complete_pipeline()
print(f"Pipeline test: {'✅ PASSED' if success else '❌ FAILED'}")
```

---

**Questions?** Check the comprehensive tests in `backend/storage/test_files/` for more examples!
