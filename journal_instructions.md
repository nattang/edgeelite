EdgeElite Journaling – "Walk-Without-Phone" R-A-G Showcase
(Business narrative ➔ user journey ➔ full-stack implementation blueprint)

1 Business Value & Demo Story (Why it matters)
Pain-point (today) EdgeElite outcome Benefit to highlight
Knowledge-worker stress journals pile up but are never re-read. EdgeElite retrieves past, self-authored coping tactics at the exact moment they are relevant. • Proof of personalized memory → differentiates vs. one-shot chatbots.
• Higher stickiness → user feels "It really knows me."
Wellness platforms give generic tips. Advice is contextual & empirical ("A 15-min phoneless walk worked for you on 5 May"). • Drives trust & conversion to paid tiers.
• Showcases Snapdragon on-device smarts → partner appeal.

Demo narrative (60-second pitch)
5 May 2025 – Session #27
User says: "I'm burnt out… going for a 15-min walk without my phone."
EdgeElite logs it.

10 Jun 2025 – Session #42 (live demo)
User says: "Huge headache, calendar is insane." – shows screenshot.
EdgeElite responds:

"You felt the same on 5 May and a short phoneless walk restored your calm.
You have a 30-min gap at 14:30 – take that same walk and breathe deeply."

Take-away for judges / investors: EdgeElite resurfaces a forgotten, self-proven remedy in real time → tangible well-being lift, powered by our R-A-G pipeline on-device.

2 Functional Requirements
Session capture

Multimodal: continuous ASR + on-demand screenshot → OCR.

Session boundaries: control:start / control:end.

Persistent memory

Raw events stored in SQLite.

Summaries & embeddings indexed in FAISS.

Retrieval-Augmented Guidance

When a new session ends, system must:

Embed the current transcript + OCR.

Retrieve K = 3 most similar past nodes from the same user.

Detect matching remedy patterns (here: phoneless walk).

Inject that evidence into the LLM prompt.

User-facing output

Journal card with:

Summary of feelings.

Action referencing past success ("as on 5 May").

Expandable "Related Memory" chip linking to Session #27.

3 Data & Control Flow (end-to-end)

```
Renderer (Nextron) Electron main.ts FastAPI backend
┌──────────────┐ ┌────────────┐ ┌─────────────────────┐
│ Start/Stop │─IPC──────────►│ edge-event │─HTTP──►│ /capture (OCR) │
│ Screenshot │ └────────────┘ │ (store_raw_*_event) │ /asr (Audio) │
└──────────────┘ │ │
▲ │ │
│ poll /api/journal │ async task: │
└──────────────────────────────────────────────┤ run_journal_pipeline│
│ ↳ process_session │
│ ↳ search_similar │
│ ↳ LLM inference │
└─────────┬───────────┘
│ ready JSON
Renderer fetch ‹entry› ◄───────────────────────────────────────┘
```

4 Technical Implementation (step-by-step)

## 4.1 Storage System Architecture (IMPLEMENTED ✅)

The EdgeElite storage system is already implemented with:

- **Interface Functions**: `store_raw_ocr_event()`, `store_raw_audio_event()`, `process_session()`, `search_similar()`
- **Database**: SQLite with structured event storage
- **Vector Store**: FAISS for semantic similarity search
- **Embeddings**: Sentence transformers for text embedding

**Key Files**:

- `backend/storage/interface.py` - Main API functions
- `backend/storage/db.py` - SQLite database management
- `backend/storage/faiss_store.py` - Vector storage and search
- `backend/storage/DEVELOPER_GUIDE.md` - Complete usage documentation

## 4.2 Current Backend Endpoints (NEEDS UPDATE ⚠️)

**Existing Endpoints**:

- `POST /capture` - OCR processing (needs database storage)
- `POST /asr` - Audio processing (needs database storage)

**Missing Endpoints**:

- `POST /api/session/end` - End session and trigger journal processing
- `POST /api/journal` - Poll for journal results

## 4.3 Backend Implementation Updates Needed

### ➊ Fix OCR Endpoint (10 LOC)

```python
@app.post("/capture")
async def capture(data: CaptureRequest):
    from backend.storage.interface import store_raw_ocr_event
    import time

    result = process_image(data.filename)

    # Store in database using correct function
    store_raw_ocr_event(
        session_id=data.session_id,
        source="ocr",
        ts=time.time(),
        text=result,
        metadata={"image_file": data.filename}
    )

    return {"message": f"OCR processed and stored for session {data.session_id}"}
```

### ➋ Fix ASR Endpoint (15 LOC)

```python
@app.post("/asr")
async def asr(request: ASRRequest):
    from backend.storage.interface import store_raw_audio_event
    import time

    result = process_audio(latest_audio_file)

    # Store using correct function (audio expects list format)
    audio_data = [{
        "timaestamp": time.time(),
        "text": result,
        "audio_file": latest_audio_file
    }]
    store_raw_audio_event(
        session_id=request.session_id,
        source="audio",
        audio_data=audio_data
    )

    return {"message": result}
```

### ➌ Add Journal Pipeline (60 LOC)

````python
async def run_journal_pipeline(session_id: str):
    """Process session and generate journal entry with RAG."""

    # 1. Process session (clean, chunk, embed)
    from backend.storage.interface import process_session, search_similar
    from backend.storage.db import StorageDB

    node_ids = process_session(session_id)

    # 2. Get current session text for RAG
    db = StorageDB()
    raw_events = db.get_raw_events_by_session(session_id)
    full_doc = "\n".join(event["text"] for event in raw_events
                        if event["source"] in ("asr", "ocr"))

    # 3. Search for similar past sessions
    similar_results = search_similar(full_doc, k=3)

    # 4. Detect remedy pattern (keyword matching)
    remedy_context = ""
    remedy_session_id = None
    for summary, content in similar_results:
        if "walk without" in content.lower():
            remedy_context = content
            remedy_session_id = "2025-05-05-demo"  # For demo
            break

    # 5. Generate journal entry with LLM
    prompt = f"""
    User journal (current session):
    ```{full_doc}```

    Past similar experience:
    ```{remedy_context}```

    Task: 1) Summarize emotions in 2 sentences.
          2) Recommend ONE concrete action referencing the past success explicitly.
          Limit to 120 words total.
    """

    response = llm_service.generate_response(prompt, [])

    # 6. Cache for frontend polling
    journal_cache[session_id] = {
        "summary_action": response,
        "related": {
            "session_id": remedy_session_id,
            "snippet": remedy_context[:200] if remedy_context else None
        }
    }
````

### ➍ Add Missing Endpoints (20 LOC)

```python
@app.post("/api/session/end")
async def end_session(request: SessionEndRequest):
    """End a session and trigger journal processing."""
    session_id = request.session_id

    # Trigger journal pipeline asynchronously
    import asyncio
    asyncio.create_task(run_journal_pipeline(session_id))

    return {"status": "processing", "session_id": session_id}

@app.post("/api/journal")
async def get_journal(request: JournalRequest):
    """Poll for journal processing status and results."""
    session_id = request.session_id
    entry = journal_cache.get(session_id)

    if entry:
        return {"status": "done", "session_id": session_id, **entry}
    else:
        return {"status": "processing", "session_id": session_id}
```

## 4.4 Frontend Implementation (renderer)

**pages/journal.jsx** - Complete journal interface with:

- Session management (start/stop)
- Screenshot and audio capture
- Journal result display
- Related memory modal

**lib/api.js** - API functions for:

- `endSession(sessionId)` - End session
- `pollJournal(sessionId)` - Poll for results

## 4.5 Demo Data Setup (WORKING ✅)

**backend/seed_demo_data.py** - Creates demo session with walk remedy:

```python
from backend.storage.interface import store_raw_audio_event, process_session

def seed_demo_data():
    session_id = "2025-05-05-demo"

    # Create audio event with the remedy
    audio_data = [{
        "timestamp": 1715000000,  # May 5, 2025
        "text": "I'm extremely stressed... going for a 15-minute walk without my phone.",
        "demo": True,
        "remedy_type": "walk_without_phone"
    }]

    # Store using correct storage function
    store_raw_audio_event(session_id, "audio", audio_data)

    # Process session to make it searchable
    process_session(session_id)
```

**backend/test_journal_pipeline.py** - Complete test suite that verifies:

- ✅ OCR/ASR data storage
- ✅ Session processing (embedding generation)
- ✅ RAG retrieval (finds walk remedy)
- ✅ End-to-end pipeline functionality

## 5 Testing Status

**CURRENT STATUS**: Core storage and retrieval pipeline **WORKING** ✅

**Test Results**:

1. ✅ OCR/ASR data can be stored using storage functions
2. ✅ Sessions can be processed (cleaned, chunked, embedded)
3. ✅ Similarity search can find relevant past experiences
4. ✅ RAG retrieval can match current stress with past remedies
5. ✅ The 'walk without phone' remedy is discoverable

**Next Steps**:

1. Update backend endpoints to use storage functions
2. Add journal pipeline endpoints
3. Implement frontend journal interface
4. Connect end-to-end workflow

## 6 Risks & Mitigations

Risk Quick mitigation (hackathon scope)
False retrieval (irrelevant past entry) Use simple keyword filter (walk AND phone) before accepting remedy match.
Long LLM latency on-device Quantize model (ggml INT4) and limit max_new_tokens = 120.
Privacy concerns Point out everything stays local (SQLite + FAISS on disk), no cloud calls.

## 7 Deliverables Checklist

- ✅ Storage system with SQLite + FAISS
- ✅ Demo data seeding script
- ✅ Complete pipeline test suite
- ⚠️ Backend endpoints need storage integration
- ⚠️ Frontend journal interface needs implementation
- ⚠️ Journal pipeline endpoints need implementation

**Total estimated remaining work**: ≈ 4-6 hours for endpoints + frontend integration.

The core **RAG retrieval system is working** - we can find the walk remedy when users express stress. The foundation is solid! 🎯
