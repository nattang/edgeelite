EdgeElite Journaling – “Walk-Without-Phone” R-A-G Showcase
(Business narrative ➔ user journey ➔ full-stack implementation blueprint)

1 Business Value & Demo Story (Why it matters)
Pain-point (today) EdgeElite outcome Benefit to highlight
Knowledge-worker stress journals pile up but are never re-read. EdgeElite retrieves past, self-authored coping tactics at the exact moment they are relevant. • Proof of personalized memory → differentiates vs. one-shot chatbots.
• Higher stickiness → user feels “It really knows me.”
Wellness platforms give generic tips. Advice is contextual & empirical (“A 15-min phoneless walk worked for you on 5 May”). • Drives trust & conversion to paid tiers.
• Showcases Snapdragon on-device smarts → partner appeal.

Demo narrative (60-second pitch)
5 May 2025 – Session #27
User says: “I’m burnt out… going for a 15-min walk without my phone.”
EdgeElite logs it.

10 Jun 2025 – Session #42 (live demo)
User says: “Huge headache, calendar is insane.” – shows screenshot.
EdgeElite responds:

“You felt the same on 5 May and a short phoneless walk restored your calm.
You have a 30-min gap at 14:30 – take that same walk and breathe deeply.”

Take-away for judges / investors: EdgeElite resurfaces a forgotten, self-proven remedy in real time → tangible well-being lift, powered by our R-A-G pipeline on-device.

2 Functional Requirements
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

Action referencing past success (“as on 5 May”).

Expandable “Related Memory” chip linking to Session #27.

3 Data & Control Flow (end-to-end)
text
Copy
Edit
Renderer (Nextron) Electron main.ts FastAPI backend
┌──────────────┐ ┌────────────┐ ┌─────────────────────┐
│ Start/Stop │─IPC──────────►│ edge-event │─HTTP──►│ /api/events │
│ Screenshot │ └────────────┘ │ (store_raw_event) │
└──────────────┘ │ │
▲ │ │
│ poll /api/journal │ async task: │
└──────────────────────────────────────────────┤ run_journal_pipeline│
│ ↳ process_session │
│ ↳ FAISS search │
│ ↳ LLM inference │
└─────────┬───────────┘
│ ready JSON
Renderer fetch ‹entry› ◄───────────────────────────────────────┘
4 Technical Implementation (step-by-step)
4.1 Frontend (renderer) – Journaling tab
Step What to code Key details
1 pages/journal.jsx – session UI Buttons: Start/Stop, 📸 Screenshot. Generate sessionId = crypto.randomUUID().
2 window.edgeElite.send(evt) → IPC Expose in preload.ts; relay via ipcMain to FastAPI (/api/events).
3 Screenshot helper desktopCapturer.getSources, convert to PNG, save temp file, POST /capture.
4 Poll endpoint After Stop, poll /api/journal every 1.5 s until status == "done".
5 Render card Show summary, actionable tip, and Related Memory chip (click → modal with old transcript + screenshot).

ETA: 150 LOC React/TS + Tailwind classes.

4.2 Backend – new/modified functions
➊ Event ingestion tweak (10 LOC)
python
Copy
Edit
@app.post("/api/events")
async def store_event(req: EventRequest):
event_id = store_raw_event(req.session_id, req.source, time.time(), req.text, req.metadata)
if req.source == "control:end":
asyncio.create_task(run_journal_pipeline(req.session_id))
return {"event_id": event_id, "status": "ok"}
➋ run_journal_pipeline(session_id) (≈80 LOC)
python
Copy
Edit
async def run_journal_pipeline(sid: str): # Gather & embed current session
node_ids = process_session(sid)
full_doc = "\n".join(ev["text"] for ev in StorageDB().get_raw_events_by_session(sid)
if ev["source"] in ("asr", "ocr"))

    # === Retrieval step – SHOWCASE RAG ===
    ctx = search_similar(full_doc, k=3, filter={"session_id": {"$ne": sid}})

    # Detect remedy pattern (very hackable heuristic)
    remedy = next((n for n in ctx if "walk without" in n[1].lower()), None)
    remedy_txt = remedy[0] if remedy else ""

    # Build LLM prompt
    prompt = f"""
    User journal (current):
    ```{full_doc}```

    Past similar entry:
    ```{remedy_txt}```

    Task: 1) Summarize emotions in 2 sentences.
          2) Recommend ONE concrete action referencing the past success explicitly.
          Limit to 120 words total.
    """
    answer = llm_service.generate_response(prompt, context=[])

    # Persist guidance as a node (future retrieval)
    StorageDB().store_session_node(sid, "Journal guidance", answer, embedder(answer))

    # Cache for frontend
    journal_cache[sid] = {
        "summary_action": answer,
        "related": {
            "session_id": remedy and remedy.metadata["session_id"],
            "snippet": remedy_txt
        }
    }

➌ Polling endpoint (15 LOC)
python
Copy
Edit
class JournalReq(BaseModel):
session_id: str = Field(alias="sessionId")

@app.post("/api/journal")
async def get_journal(req: JournalReq):
entry = journal_cache.get(req.session_id)
return {"status": "done", \*\*entry} if entry else {"status": "processing"}
4.3 Pre-loading historical memory for demo
python
Copy
Edit

# One-time script (run before hackathon)

sid_demo = "2025-05-05-demo"
store_raw_event(sid_demo, "asr", ts_may5,
"I’m extremely stressed … going for a 15-minute walk without my phone.")
process_session(sid_demo) # embeds & indexes
(You now have the May 5 node ready for FAISS retrieval.)

5 Prompt-to-Screen Trace (explain to judges)
Embedding → Similarity hit
FAISS returns node #8 (5 May 2025) cosine = 0.82.

Prompt (visible in demo slides)

Past similar entry: “I’m extremely stressed … going for a 15-minute walk without my phone.”

LLM output (rendered)

vbnet
Copy
Edit
You’re overwhelmed and sleep-deprived.  
Try the same 15-minute phoneless walk that calmed you on 5 May; you have a slot at 14:30.
Users & investors see the chain: memory ➔ retrieval ➔ grounded advice.

6 Risks & Mitigations
Risk Quick mitigation (hackathon scope)
False retrieval (irrelevant past entry) Use simple keyword filter (walk AND phone) before accepting remedy match.
Long LLM latency on-device Quantize model (ggml INT4) and limit max_new_tokens = 120.
Privacy concerns Point out everything stays local (SQLite + FAISS on disk), no cloud calls.

7 Deliverables Checklist
Markdown doc (this file) committed to docs/journal_rag_demo.md.

Frontend journal.jsx + IPC boilerplate.

Backend run_journal_pipeline, endpoint tweaks.

Script to seed May 5 session.

Demo script (slide + live run).

Total new code ≈ 300 LOC. The story crystal-clearly showcases EdgeElite’s unique selling point: personal memory-based guidance powered by real-time R-A-G on device.
