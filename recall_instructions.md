---

# EdgeElite Recall — "Context Recall" R-A-G Demo

> **Use case**: Seamlessly recall prior context in a multi-modal, multi-app environment—via a simple voice query—powered by EdgeElite’s on-device OCR, ASR, and RAG pipeline.

---

## 1. Business Story & Value

| Current Problem                                                        | EdgeElite Demo Outcome                                                   | Value Add                                                                            |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| Professionals constantly lose context across tools (e.g. chat → call). | User can say “What did I say about X earlier?” and get an instant reply. | Showcases real-world assistant behavior — proactive, grounded, multi-modal, private. |

### Demo Flow

1. **1:00 PM — Chat on Teams (Person 1)**
   User says “I’m delaying Project X” and screenshots their calendar. EdgeElite captures:

   - ASR: audio note
   - OCR: screenshot of calendar window

2. **3:00 PM — On video call (Person 2)**
   User says: _“EdgeElite, what did I say about Project X earlier?”_
   EdgeElite responds: _“You mentioned delaying Project X by 2 weeks due to scheduling conflicts. Here’s the screenshot from earlier.”_

---

## 2. Functional Breakdown

### Input Modalities

- **Audio (ASR)** — via mic; stored in 3s chunks.
- **Visual (OCR)** — user screenshot or window frame (e.g. calendar, chat).

### Trigger Point

- When user explicitly asks a **voice-based context question**, e.g.:

  > “What did I say earlier about…”, “Remind me what I mentioned regarding…”

### Pipeline Behavior

1. Concatenate ASR chunks from last 30 seconds → extract the user’s question.
2. Treat this as a RAG query.
3. Run `search_similar()` over all session events.
4. Insert top match(es) into LLM prompt.
5. Generate natural language answer.
6. Return in-app response (not a journal entry).

---

## 3. Backend Implementation

### Event Capture API

Re-use existing endpoints:

```python
@app.post("/api/events")
@app.post("/api/asr")        # from audio chunks
@app.post("/api/capture")    # from screenshots (OCR)
```

All events are stored with:

```json
{
  "session_id": "2025-07-13",
  "source": "asr" | "ocr",
  "ts": "2025-07-13T15:03:12",
  "text": "User said or screenshot read..."
}
```

### Query Inference Route

```python
@app.post("/api/query")
async def handle_query(request: QueryRequest):
    context_doc = get_recent_asr_text(request.session_id)
    query = extract_question(context_doc)
    results = search_similar(query, k=5, session_id=request.session_id)

    prompt = build_prompt(query, results)
    answer = llm.generate_response(prompt)
    return {"answer": answer, "sources": results}
```

#### Sample Prompt Template

```
User just asked: "{query}"
Relevant notes from earlier:
<<< {context_1} >>>
<<< {context_2} >>>
Write a short, helpful answer to the user's question using the context above.
```

---

## 4. Frontend Integration (Nextron)

### Voice Trigger UI

- Use `MediaRecorder` to buffer 30s of audio.
- On trigger phrase, send to `/asr` endpoint.
- Then POST to `/api/query` with current session.

### Response Display

```jsx
function AssistantBubble({ answer }) {
  return (
    <div className="fixed bottom-6 right-6 bg-blue-800 text-white p-4 rounded-xl">
      <b>EdgeElite says:</b>
      <br />
      {answer}
    </div>
  );
}
```

---

## 5. Seeding Demo Data

```python
sid = "2025-07-13"
store_raw_event(sid, "ocr", ts1, "Calendar: Meetings 1pm–6pm")
store_raw_event(sid, "asr", ts1, "We’re delaying Project X by 2 weeks")
process_session(sid)
```

Later:

```python
handle_query(QueryRequest(session_id=sid, query="What did I say about Project X?"))
```

---

## 6. Summary & Deliverables

| Component                          | Done?                         |
| ---------------------------------- | ----------------------------- |
| ASR / OCR capture (reused)         | ✅                            |
| Screenshot + audio interface       | ✅                            |
| `/api/query` endpoint              | ✅                            |
| RAG pipeline reuse                 | ✅                            |
| Frontend assistant response bubble | ✅                            |
| Trigger phrase detection           | ✅ (via audio chunk analysis) |
| Demo script + seed events          | ✅                            |

✅ **Total code delta minimal — \~120 LOC** thanks to reusing Journaling + RAG pipeline.

---
