# EdgeElite Journal Feature - Execution Plan

## 📋 Overview

This document outlines the complete implementation plan for the EdgeElite Journal feature as described in `journal_instructions.md`. The journal feature will showcase RAG (Retrieval-Augmented Generation) capabilities by providing personalized guidance based on past session memories.

## 🎯 Goal

Implement a journaling system that:

1. Captures user sessions (OCR + Audio)
2. Processes sessions when they end (clean, chunk, embed)
3. Uses RAG to find similar past experiences
4. Generates personalized guidance with LLM
5. Displays journal entries with "Related Memory" links

## 📁 Current Architecture

- **Frontend**: Nextron (Electron + Next.js) with session management
- **Backend**: FastAPI with OCR, ASR, and LLM services
- **Storage**: SQLite + FAISS with semantic search capabilities
- **Functions**: `store_raw_ocr_event()`, `store_raw_audio_event()`, `process_session()`, `search_similar()`

---

## 🚀 Phase 1: Fix Current Endpoints (Priority: HIGH)

### **Issue**: Current OCR/ASR endpoints don't store data in the database

### **Task 1.1: Update CaptureRequest Model**

- **File**: `backend/ocr.py`
- **Changes**:
  ```python
  class CaptureRequest(BaseModel):
      filename: str
      session_id: str = Field(alias="sessionId")

      class Config:
          allow_population_by_field_name = True
  ```

### **Task 1.2: Fix /capture Endpoint**

- **File**: `backend/main.py`
- **Changes**:
  ```python
  @app.post("/capture")
  async def capture(data: CaptureRequest):
      import time
      from backend.storage.interface import store_raw_ocr_event

      print(f"Received capture request for: {data.filename}")
      result = process_image(data.filename)

      # Store in database using correct function
      store_raw_ocr_event(
          session_id=data.session_id,
          source="ocr",
          ts=time.time(),
          text=result,
          metadata={"image_file": data.filename}
      )

      return {"message": f"Processed {data.filename}"}
  ```

### **Task 1.3: Create ASRRequest Model**

- **File**: `backend/main.py`
- **Changes**:
  ```python
  class ASRRequest(BaseModel):
      session_id: str = Field(alias="sessionId")

      class Config:
          allow_population_by_field_name = True
  ```

### **Task 1.4: Fix /asr Endpoint**

- **File**: `backend/main.py`
- **Changes**:
  ```python
  @app.post("/asr")
  async def asr(request: ASRRequest):
      import time
      from backend.storage.interface import store_raw_audio_event

      print("🎤 Received ASR trigger request")

      # Process audio as before
      recordings_dir = os.path.join(os.path.expanduser("~"), "EdgeElite", "recordings")
      if os.path.exists(recordings_dir):
          wav_files = [f for f in os.listdir(recordings_dir) if f.endswith('.wav')]
          if wav_files:
              wav_files.sort(key=lambda x: os.path.getmtime(os.path.join(recordings_dir, x)), reverse=True)
              latest_audio_file = os.path.join(recordings_dir, wav_files[0])
              result = process_audio(latest_audio_file)

              # Store using correct function (audio expects list format)
              audio_data = [{
                  "timestamp": time.time(),
                  "text": result,
                  "audio_file": latest_audio_file
              }]
              store_raw_audio_event(
                  session_id=request.session_id,
                  source="audio",
                  audio_data=audio_data
              )

              return {"message": result}

      return {"message": "No audio file found"}
  ```

### **Task 1.5: Update Frontend API Calls**

- **File**: `renderer/lib/capture.js`
- **Changes**:

  ```javascript
  export const sendCaptureRequest = async (filename, sessionId) => {
    const res = await fetch("http://localhost:8000/capture", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename, sessionId }),
    });
    // ... rest of function
  };
  ```

- **File**: `renderer/lib/audio.js`
- **Changes**:
  ```javascript
  export async function sendListenRequest(filename, sessionId) {
    const res = await fetch("http://localhost:8000/asr", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId }),
    });
    // ... rest of function
  }
  ```

---

## 🔄 Phase 2: Add Journal Pipeline (Priority: HIGH)

### **Task 2.1: Add SessionEndRequest Model**

- **File**: `backend/main.py`
- **Changes**:

  ```python
  class SessionEndRequest(BaseModel):
      session_id: str = Field(alias="sessionId")

      class Config:
          allow_population_by_field_name = True

  class JournalRequest(BaseModel):
      session_id: str = Field(alias="sessionId")

      class Config:
          allow_population_by_field_name = True
  ```

### **Task 2.2: Add Journal Cache**

- **File**: `backend/main.py`
- **Changes**: Add global variable at top of file:
  ```python
  # Journal processing cache
  journal_cache = {}
  ```

### **Task 2.3: Implement Journal Pipeline Function**

- **File**: `backend/main.py`
- **Changes**:
  ````python
  async def run_journal_pipeline(session_id: str):
      """
      Process a session and generate journal entry with RAG.

      Steps:
      1. Process session (clean, chunk, embed)
      2. Get current session text
      3. Search for similar past sessions
      4. Detect remedy patterns
      5. Generate LLM response with context
      6. Cache result for frontend
      """
      try:
          print(f"🔄 Starting journal pipeline for session: {session_id}")

          # 1. Process session using storage system
          from backend.storage.interface import process_session
          node_ids = process_session(session_id)
          print(f"📊 Session processed: {len(node_ids)} nodes created")

          # 2. Get current session text for RAG
          from backend.storage.db import StorageDB
          db = StorageDB()
          raw_events = db.get_raw_events_by_session(session_id)
          full_doc = "\n".join(event["text"] for event in raw_events
                              if event["source"] in ("asr", "ocr"))

          # 3. Search for similar past sessions (exclude current session)
          from backend.storage.interface import search_similar
          similar_results = search_similar(full_doc, k=3)

          # 4. Detect remedy pattern (simple keyword matching)
          remedy_context = ""
          remedy_session_id = None
          for summary, content in similar_results:
              if "walk without" in content.lower():
                  remedy_context = content
                  # Extract session ID from content if possible
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

          print(f"✅ Journal pipeline completed for session: {session_id}")

      except Exception as e:
          print(f"❌ Journal pipeline error for session {session_id}: {e}")
          journal_cache[session_id] = {"error": str(e)}
  ````

### **Task 2.4: Add Session End Endpoint**

- **File**: `backend/main.py`
- **Changes**:
  ```python
  @app.post("/api/session/end")
  async def end_session(request: SessionEndRequest):
      """
      End a session and trigger journal processing.
      """
      session_id = request.session_id
      print(f"🔚 Session ending: {session_id}")

      # Trigger journal pipeline asynchronously
      import asyncio
      asyncio.create_task(run_journal_pipeline(session_id))

      return {"status": "processing", "session_id": session_id}
  ```

### **Task 2.5: Add Journal Polling Endpoint**

- **File**: `backend/main.py`
- **Changes**:
  ```python
  @app.post("/api/journal")
  async def get_journal(request: JournalRequest):
      """
      Poll for journal processing status and results.
      """
      session_id = request.session_id
      entry = journal_cache.get(session_id)

      if entry:
          return {"status": "done", "session_id": session_id, **entry}
      else:
          return {"status": "processing", "session_id": session_id}
  ```

---

## 🎨 Phase 3: Frontend Journal Interface (Priority: MEDIUM)

### **Task 3.1: Create Journal Page Component**

- **File**: `renderer/pages/journal.jsx`
- **Changes**: Create complete journal interface with:

  ```javascript
  import React from "react";
  import Head from "next/head";
  import { api } from "../lib/api";

  export default function JournalPage() {
    const [sessionId, setSessionId] = React.useState(null);
    const [isSessionActive, setIsSessionActive] = React.useState(false);
    const [journalEntry, setJournalEntry] = React.useState(null);
    const [isProcessing, setIsProcessing] = React.useState(false);
    const [showRelatedModal, setShowRelatedModal] = React.useState(false);

    // Session management functions
    const generateSessionId = () => {
      return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    };

    const startSession = () => {
      const newSessionId = generateSessionId();
      setSessionId(newSessionId);
      setIsSessionActive(true);
      setJournalEntry(null);
    };

    const endSession = async () => {
      if (!sessionId) return;

      setIsSessionActive(false);
      setIsProcessing(true);

      try {
        // End session and start journal processing
        await api.endSession(sessionId);

        // Poll for journal results
        const result = await api.pollJournal(sessionId);
        setJournalEntry(result);
      } catch (error) {
        console.error("Journal processing failed:", error);
      } finally {
        setIsProcessing(false);
      }
    };

    // Screenshot and audio functions
    const handleCapture = async () => {
      // ... screenshot logic with sessionId
    };

    const handleListen = async () => {
      // ... audio recording logic with sessionId
    };

    return (
      <>
        <Head>
          <title>Journal - EdgeElite</title>
        </Head>

        <div className="p-6 max-w-2xl mx-auto">
          {/* Session Controls */}
          <div className="mb-6">
            <h1 className="text-2xl font-bold mb-4">EdgeElite Journal</h1>

            {/* Session Status */}
            <div className="bg-gray-100 rounded-lg p-4 mb-4">
              <div className="flex items-center justify-between">
                <span className="font-medium">Session Status:</span>
                <span
                  className={`px-3 py-1 rounded-full text-sm ${
                    isSessionActive
                      ? "bg-green-200 text-green-800"
                      : "bg-gray-200"
                  }`}
                >
                  {isSessionActive ? "Active" : "Inactive"}
                </span>
              </div>

              {sessionId && (
                <div className="text-sm text-gray-600 mt-2">
                  Session ID: {sessionId}
                </div>
              )}
            </div>

            {/* Control Buttons */}
            <div className="flex gap-3 mb-4">
              <button
                onClick={isSessionActive ? endSession : startSession}
                disabled={isProcessing}
                className={`px-4 py-2 rounded-lg font-medium ${
                  isSessionActive
                    ? "bg-red-600 text-white hover:bg-red-700"
                    : "bg-blue-600 text-white hover:bg-blue-700"
                } ${isProcessing ? "opacity-50 cursor-not-allowed" : ""}`}
              >
                {isProcessing
                  ? "Processing..."
                  : isSessionActive
                  ? "End Session"
                  : "Start Session"}
              </button>

              <button
                onClick={handleCapture}
                disabled={!isSessionActive}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400"
              >
                📸 Screenshot
              </button>

              <button
                onClick={handleListen}
                disabled={!isSessionActive}
                className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-400"
              >
                🎤 Listen
              </button>
            </div>
          </div>

          {/* Journal Entry Display */}
          {journalEntry && (
            <div className="bg-white border rounded-lg p-6 shadow-sm">
              <h2 className="text-xl font-semibold mb-4">Journal Entry</h2>

              <div className="prose max-w-none">
                <p className="whitespace-pre-wrap">
                  {journalEntry.summary_action}
                </p>
              </div>

              {/* Related Memory Chip */}
              {journalEntry.related?.snippet && (
                <div className="mt-4">
                  <button
                    onClick={() => setShowRelatedModal(true)}
                    className="inline-flex items-center px-3 py-1 rounded-full text-sm bg-blue-100 text-blue-800 hover:bg-blue-200"
                  >
                    🔗 Related Memory
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Related Memory Modal */}
          {showRelatedModal && journalEntry?.related && (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
              <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
                <h3 className="text-lg font-semibold mb-3">Related Memory</h3>
                <p className="text-gray-700 mb-4">
                  {journalEntry.related.snippet}
                </p>
                <div className="text-sm text-gray-500 mb-4">
                  Session: {journalEntry.related.session_id}
                </div>
                <button
                  onClick={() => setShowRelatedModal(false)}
                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                >
                  Close
                </button>
              </div>
            </div>
          )}
        </div>
      </>
    );
  }
  ```

### **Task 3.2: Add Journal API Functions**

- **File**: `renderer/lib/api.js`
- **Changes**:
  ```javascript
  export const api = {
    // ... existing functions

    // End session and trigger journal processing
    endSession: async (sessionId) => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/session/end`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sessionId }),
        });

        if (!response.ok) {
          throw new Error(`Session end failed: ${response.status}`);
        }

        return await response.json();
      } catch (error) {
        console.error("Session end failed:", error);
        throw error;
      }
    },

    // Poll for journal processing results
    pollJournal: async (sessionId) => {
      let status = "processing";
      let attempts = 0;
      const maxAttempts = 40; // 60 seconds max

      while (status === "processing" && attempts < maxAttempts) {
        try {
          const response = await fetch(`${API_BASE_URL}/api/journal`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sessionId }),
          });

          if (!response.ok) {
            throw new Error(`Journal poll failed: ${response.status}`);
          }

          const data = await response.json();
          status = data.status;

          if (status === "done") {
            return data;
          }

          // Wait 1.5 seconds before next poll
          await new Promise((resolve) => setTimeout(resolve, 1500));
          attempts++;
        } catch (error) {
          console.error("Journal polling error:", error);
          throw error;
        }
      }

      throw new Error("Journal processing timeout");
    },
  };
  ```

### **Task 3.3: Add Navigation Link**

- **File**: `renderer/pages/home.jsx`
- **Changes**: Add link to journal page in the UI

---

## 🗂️ Phase 4: Demo Data Setup (Priority: LOW)

### **Task 4.1: Create Demo Data Seeder**

- **File**: `backend/seed_demo_data.py`
- **Changes**:

  ```python
  """
  Seed demo data for journal feature demonstration.
  Creates a historical session from "May 5th" with the walk-without-phone remedy.
  """

  from backend.storage.interface import store_raw_audio_event, process_session
  import time

  def seed_demo_data():
      """
      Create demo session with walk-without-phone remedy.
      """
      session_id = "2025-05-05-demo"
      may5_timestamp = 1715000000  # Approximate May 5, 2025

      # Create audio event with the remedy
      audio_data = [{
          "timestamp": may5_timestamp,
          "text": "I'm extremely stressed... going for a 15-minute walk without my phone.",
          "demo": True,
          "remedy_type": "walk_without_phone"
      }]

      # Store using correct storage function
      event_ids = store_raw_audio_event(session_id, "audio", audio_data)
      print(f"Stored demo audio events: {event_ids}")

      # Process session to make it searchable
      node_ids = process_session(session_id)
      print(f"Processed demo session: {len(node_ids)} nodes created")

      print(f"✅ Demo data seeded successfully for session: {session_id}")

  if __name__ == "__main__":
      seed_demo_data()
  ```

### **Task 4.2: Create Demo Runner Script**

- **File**: `backend/run_demo_setup.py`
- **Changes**:

  ```python
  """
  Demo setup script for journal feature.
  Run this before demonstrating the journal feature.
  """

  import sys
  import os

  # Add backend to path
  sys.path.append(os.path.dirname(__file__))

  from seed_demo_data import seed_demo_data

  if __name__ == "__main__":
      print("🚀 Setting up EdgeElite Journal Demo...")
      seed_demo_data()
      print("✅ Demo setup complete!")
      print("\nDemo scenario:")
      print("1. Start a new session")
      print("2. Say: 'Huge headache, calendar is insane'")
      print("3. Take a screenshot of a busy calendar")
      print("4. End session")
      print("5. Journal should reference the May 5th walk remedy")
  ```

---

## 🧪 Phase 5: Testing & Validation (Priority: LOW)

### **Task 5.1: Test Individual Components**

- Verify OCR/ASR endpoints store data correctly
- Test session processing pipeline
- Validate journal generation with demo data

### **Task 5.2: End-to-End Testing**

- Complete demo scenario walkthrough
- Test error handling and edge cases
- Verify UI responsiveness and polling

### **Task 5.3: Demo Preparation**

- Prepare demo script and talking points
- Test on clean database with demo data
- Verify journal retrieval and display

---

## 📊 Implementation Timeline

### **Week 1: Core Backend (8-10 hours)**

- Phase 1: Fix endpoints (2-3 hours)
- Phase 2: Journal pipeline (4-5 hours)
- Phase 4: Demo data (1-2 hours)

### **Week 2: Frontend & Polish (6-8 hours)**

- Phase 3: Journal interface (4-5 hours)
- Phase 5: Testing & validation (2-3 hours)

### **Total Estimated Effort: 14-18 hours**

---

## 🚨 Risk Assessment

### **High Risk**

- LLM prompt engineering for consistent journal format
- FAISS similarity search quality for RAG

### **Medium Risk**

- Frontend polling reliability
- Session end detection timing

### **Low Risk**

- Storage system integration (well-documented)
- Basic UI components (standard React)

---

## 🎯 Success Criteria

### **Functional Requirements**

- ✅ Sessions capture OCR + Audio data
- ✅ Session end triggers journal processing
- ✅ RAG retrieval finds relevant past experiences
- ✅ Journal displays summary + action + related memory
- ✅ Demo scenario works end-to-end

### **Technical Requirements**

- ✅ All data stored in SQLite + FAISS
- ✅ Processing pipeline uses existing storage functions
- ✅ Frontend polls for results reliably
- ✅ Related memory modal shows historical context

### **Demo Requirements**

- ✅ Pre-seeded May 5th demo data
- ✅ Live demo scenario executes smoothly
- ✅ Clear demonstration of RAG capabilities
- ✅ Compelling business narrative delivery

---

## 📝 Notes

- Keep journal prompts simple and template-based initially
- Use keyword matching for remedy detection (expandable later)
- Focus on core functionality first, polish UI later
- Ensure all changes are backward compatible
- Test thoroughly with demo data before presentation

---

**Next Steps**: Begin with Phase 1 - fixing the current endpoints to store data properly using the correct storage functions from the DEVELOPER_GUIDE.
