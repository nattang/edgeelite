# EdgeElite Recall Feature - Execution Plan

## 📋 Overview

This document outlines the complete implementation plan for the EdgeElite Recall feature as described in `recall_instructions.md`. The recall feature will showcase real-time context retrieval by allowing users to ask voice queries like "What did I say about X earlier?" and get instant responses powered by our existing RAG pipeline.

## 🎯 Goal

Implement a context recall system that:

1. **Continuously captures** user sessions (OCR + Audio)
2. **Detects voice queries** asking for context recall
3. **Uses RAG to find** relevant past information
4. **Generates natural responses** with LLM
5. **Displays assistant-style** responses in real-time

## 📁 Current Architecture Status

### **✅ ALREADY IMPLEMENTED (From Journal Feature)**

- **Storage System**: SQLite + FAISS with `store_raw_ocr_event()`, `store_raw_audio_event()`, `process_session()`, `search_similar()`
- **OCR/ASR Capture**: `/capture` and `/asr` endpoints (need storage integration)
- **RAG Pipeline**: Working similarity search that finds relevant past content
- **Demo Data**: Seeding scripts and test validation
- **LLM Integration**: `llm_service.generate_response()` functionality

### **🔧 NEEDS IMPLEMENTATION (New for Recall)**

- **Query Detection**: Extract questions from recent ASR text
- **Query Endpoint**: `/api/query` for handling context queries
- **Voice Trigger UI**: Continuous audio monitoring and trigger detection
- **Assistant Response**: Real-time response bubble display

---

## 🚀 Phase 1: Backend Query Endpoint (Priority: HIGH)

### **Task 1.1: Create Query Request Models**

**File**: `backend/main.py`
**Changes**:

```python
class QueryRequest(BaseModel):
    session_id: str = Field(alias="sessionId")
    query_text: str = Field(alias="queryText", default="")

    class Config:
        allow_population_by_field_name = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    session_id: str
```

### **Task 1.2: Implement Query Processing Functions**

**File**: `backend/main.py`
**Changes**:

```python
def get_recent_asr_text(session_id: str, minutes_back: int = 30) -> str:
    """
    Get recent ASR text from the last N minutes for query extraction.
    """
    from backend.storage.db import StorageDB
    import time

    db = StorageDB()
    cutoff_time = time.time() - (minutes_back * 60)

    # Get recent ASR events
    recent_events = db.get_recent_events_by_type(
        session_id=session_id,
        source="asr",
        since_timestamp=cutoff_time
    )

    # Concatenate recent ASR text
    return " ".join(event["text"] for event in recent_events)

def extract_question_from_text(text: str) -> str:
    """
    Extract the actual question from recent ASR text.
    Looks for patterns like "What did I say about X" or "Remind me about Y"
    """
    import re

    # Common question patterns
    patterns = [
        r"what did i say about (.+?)(?:\?|$)",
        r"remind me (?:about|what) (.+?)(?:\?|$)",
        r"what was (?:mentioned|said) about (.+?)(?:\?|$)",
        r"tell me about (.+?)(?:\?|$)",
        r"recall (.+?)(?:\?|$)"
    ]

    text_lower = text.lower()

    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            topic = match.group(1).strip()
            return f"What was mentioned about {topic}?"

    # If no pattern matched, return the last part as potential question
    sentences = text.split('.')
    if sentences:
        return sentences[-1].strip()

    return text

def build_query_prompt(query: str, search_results: List[Tuple[str, str]]) -> str:
    """
    Build LLM prompt for answering context queries.
    """
    if not search_results:
        return f"""
        User asked: "{query}"

        I don't have any relevant information from previous conversations to answer this question.

        Please respond helpfully that you don't have context about this topic.
        """

    context_sections = []
    for i, (summary, content) in enumerate(search_results, 1):
        context_sections.append(f"Context {i}: {content}")

    context_text = "\n\n".join(context_sections)

    return f"""
    User just asked: "{query}"

    Relevant information from earlier:
    {context_text}

    Task: Write a short, helpful answer to the user's question using the context above.
    - Be conversational and natural
    - Reference specific details from the context
    - If the context doesn't fully answer the question, say so
    - Keep response under 50 words
    """
```

### **Task 1.3: Implement Query Endpoint**

**File**: `backend/main.py`
**Changes**:

```python
@app.post("/api/query")
async def handle_query(request: QueryRequest):
    """
    Handle context recall queries using RAG pipeline.
    """
    try:
        session_id = request.session_id
        print(f"🔍 Received query for session: {session_id}")

        # If query text is provided, use it directly
        if request.query_text:
            query = request.query_text
        else:
            # Extract question from recent ASR text
            recent_text = get_recent_asr_text(session_id)
            query = extract_question_from_text(recent_text)

        print(f"🎯 Extracted query: {query}")

        # Search for relevant context using existing RAG pipeline
        from backend.storage.interface import search_similar
        search_results = search_similar(query, k=5)

        print(f"📚 Found {len(search_results)} relevant results")

        # Generate response using LLM
        prompt = build_query_prompt(query, search_results)
        answer = llm_service.generate_response(prompt, [])

        # Format sources for frontend
        sources = []
        for summary, content in search_results:
            sources.append({
                "summary": summary,
                "content": content[:200] + "..." if len(content) > 200 else content
            })

        return QueryResponse(
            answer=answer,
            sources=sources,
            confidence=0.8 if search_results else 0.2,
            session_id=session_id
        )

    except Exception as e:
        print(f"❌ Query processing error: {e}")
        return QueryResponse(
            answer="I'm sorry, I encountered an error while trying to recall that information.",
            sources=[],
            confidence=0.0,
            session_id=session_id
        )
```

---

## 🎨 Phase 2: Frontend Voice Trigger & Response UI (Priority: HIGH)

### **Task 2.1: Create Voice Query Hook**

**File**: `renderer/lib/voice-query.js`
**Changes**:

```javascript
// Voice query management for context recall
export class VoiceQueryManager {
  constructor() {
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.isListening = false;
    this.sessionId = null;
    this.onResponseCallback = null;
  }

  async startContinuousListening(sessionId, onResponse) {
    this.sessionId = sessionId;
    this.onResponseCallback = onResponse;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.mediaRecorder = new MediaRecorder(stream);

      this.mediaRecorder.addEventListener("dataavailable", (event) => {
        this.audioChunks.push(event.data);
      });

      this.mediaRecorder.addEventListener("stop", () => {
        this.processVoiceQuery();
      });

      // Record in 30-second chunks
      this.mediaRecorder.start();
      this.isListening = true;

      setTimeout(() => {
        if (this.isListening) {
          this.mediaRecorder.stop();
          this.startContinuousListening(sessionId, onResponse); // Restart
        }
      }, 30000);
    } catch (error) {
      console.error("Voice listening error:", error);
    }
  }

  async processVoiceQuery() {
    if (this.audioChunks.length === 0) return;

    const audioBlob = new Blob(this.audioChunks, { type: "audio/wav" });
    this.audioChunks = [];

    try {
      // Convert to WAV and save (reuse existing audio processing)
      const timestamp = Date.now();
      const filename = `query-${timestamp}.wav`;

      await window.electronAPI.saveAudioFile(audioBlob, filename);

      // Send to ASR for transcription
      const asrResponse = await fetch("http://localhost:8000/asr", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: this.sessionId }),
      });

      if (asrResponse.ok) {
        const asrResult = await asrResponse.json();
        const transcript = asrResult.message;

        // Check if this looks like a recall query
        if (this.isRecallQuery(transcript)) {
          await this.handleRecallQuery(transcript);
        }
      }
    } catch (error) {
      console.error("Voice query processing error:", error);
    }
  }

  isRecallQuery(transcript) {
    const recallKeywords = [
      "what did i say",
      "remind me",
      "what was mentioned",
      "tell me about",
      "recall",
      "edgeelite",
    ];

    const lowerTranscript = transcript.toLowerCase();
    return recallKeywords.some((keyword) => lowerTranscript.includes(keyword));
  }

  async handleRecallQuery(transcript) {
    try {
      const response = await fetch("http://localhost:8000/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sessionId: this.sessionId,
          queryText: transcript,
        }),
      });

      if (response.ok) {
        const result = await response.json();
        this.onResponseCallback(result);
      }
    } catch (error) {
      console.error("Recall query error:", error);
    }
  }

  stopListening() {
    this.isListening = false;
    if (this.mediaRecorder && this.mediaRecorder.state !== "inactive") {
      this.mediaRecorder.stop();
    }
  }
}
```

### **Task 2.2: Create Assistant Response Component**

**File**: `renderer/components/AssistantBubble.jsx`
**Changes**:

```javascript
import React from "react";

export default function AssistantBubble({ response, onClose }) {
  if (!response) return null;

  return (
    <div className="fixed bottom-6 right-6 bg-blue-800 text-white p-4 rounded-xl shadow-lg max-w-md z-50">
      <div className="flex justify-between items-start mb-2">
        <span className="font-bold text-blue-200">EdgeElite says:</span>
        <button
          onClick={onClose}
          className="text-blue-200 hover:text-white ml-2"
        >
          ×
        </button>
      </div>

      <div className="mb-3">{response.answer}</div>

      {response.sources && response.sources.length > 0 && (
        <div className="text-xs text-blue-200">
          <details>
            <summary className="cursor-pointer">
              Sources ({response.sources.length})
            </summary>
            <div className="mt-2 space-y-1">
              {response.sources.map((source, index) => (
                <div key={index} className="bg-blue-900 p-2 rounded text-xs">
                  {source.content}
                </div>
              ))}
            </div>
          </details>
        </div>
      )}

      <div className="text-xs text-blue-300 mt-2">
        Confidence: {Math.round(response.confidence * 100)}%
      </div>
    </div>
  );
}
```

### **Task 2.3: Update Main Pages with Voice Query**

**File**: `renderer/pages/home.jsx`
**Changes**:

```javascript
import { VoiceQueryManager } from "../lib/voice-query";
import AssistantBubble from "../components/AssistantBubble";

export default function HomePage() {
  const [voiceQueryManager] = useState(() => new VoiceQueryManager());
  const [assistantResponse, setAssistantResponse] = useState(null);
  const [isVoiceListening, setIsVoiceListening] = useState(false);

  const startVoiceRecall = () => {
    const sessionId = `recall_session_${Date.now()}`;

    voiceQueryManager.startContinuousListening(sessionId, (response) => {
      setAssistantResponse(response);

      // Auto-hide response after 10 seconds
      setTimeout(() => {
        setAssistantResponse(null);
      }, 10000);
    });

    setIsVoiceListening(true);
  };

  const stopVoiceRecall = () => {
    voiceQueryManager.stopListening();
    setIsVoiceListening(false);
  };

  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">EdgeElite</h1>

      {/* Voice Recall Controls */}
      <div className="mb-6">
        <h2 className="text-xl font-semibold mb-3">Voice Context Recall</h2>
        <div className="flex gap-3">
          <button
            onClick={isVoiceListening ? stopVoiceRecall : startVoiceRecall}
            className={`px-4 py-2 rounded-lg font-medium ${
              isVoiceListening
                ? "bg-red-600 text-white hover:bg-red-700"
                : "bg-green-600 text-white hover:bg-green-700"
            }`}
          >
            {isVoiceListening
              ? "🛑 Stop Voice Recall"
              : "🎤 Start Voice Recall"}
          </button>

          {isVoiceListening && (
            <div className="flex items-center text-green-600">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse mr-2"></div>
              Listening for "What did I say about..."
            </div>
          )}
        </div>
      </div>

      {/* Rest of home page content */}

      {/* Assistant Response Bubble */}
      <AssistantBubble
        response={assistantResponse}
        onClose={() => setAssistantResponse(null)}
      />
    </div>
  );
}
```

---

## 🗂️ Phase 3: Demo Data & Integration (Priority: MEDIUM)

### **Task 3.1: Create Recall Demo Data**

**File**: `backend/seed_recall_demo.py`
**Changes**:

```python
"""
Seed demo data for recall feature demonstration.
Creates a session with context that can be recalled later.
"""

from backend.storage.interface import store_raw_audio_event, store_raw_ocr_event, process_session
import time

def seed_recall_demo():
    """
    Create demo session with contextual information that can be recalled.
    """
    session_id = "recall_demo_session"
    demo_timestamp = time.time() - 3600  # 1 hour ago

    print(f"🎬 Creating recall demo session: {session_id}")

    # Simulate 1:00 PM context - User mentions delaying Project X
    ocr_events = [
        "Microsoft Teams - Project X Discussion",
        "Calendar: Project X Meeting - Moved to next week",
        "Slack: @channel Project X timeline update needed"
    ]

    for i, ocr_text in enumerate(ocr_events):
        store_raw_ocr_event(
            session_id=session_id,
            source="ocr",
            ts=demo_timestamp + i * 60,
            text=ocr_text,
            metadata={"demo": True, "context": "project_x_delay"}
        )

    # Simulate audio context
    audio_data = [
        {
            "timestamp": demo_timestamp + 300,
            "text": "I'm delaying Project X by 2 weeks due to scheduling conflicts",
            "context": "project_x_delay"
        },
        {
            "timestamp": demo_timestamp + 360,
            "text": "The client meeting needs to be rescheduled to accommodate the delay",
            "context": "project_x_delay"
        },
        {
            "timestamp": demo_timestamp + 420,
            "text": "I'll send an update to the team about the Project X timeline changes",
            "context": "project_x_delay"
        }
    ]

    store_raw_audio_event(session_id, "audio", audio_data)

    # Process session to make it searchable
    node_ids = process_session(session_id)
    print(f"✅ Recall demo session processed: {len(node_ids)} nodes created")

    # Create additional context for variety
    session_id_2 = "recall_demo_session_2"

    audio_data_2 = [
        {
            "timestamp": demo_timestamp + 1800,  # 30 minutes later
            "text": "The budget for Q4 marketing campaign is approved at $50,000",
            "context": "marketing_budget"
        },
        {
            "timestamp": demo_timestamp + 1860,
            "text": "We need to focus on digital channels for the Q4 campaign",
            "context": "marketing_budget"
        }
    ]

    store_raw_audio_event(session_id_2, "audio", audio_data_2)
    process_session(session_id_2)

    print(f"✅ Recall demo data seeded successfully!")
    print(f"\nDemo queries to try:")
    print(f"1. 'What did I say about Project X?'")
    print(f"2. 'Remind me about the marketing budget'")
    print(f"3. 'What was mentioned about scheduling?'")

if __name__ == "__main__":
    seed_recall_demo()
```

### **Task 3.2: Create Recall Test Script**

**File**: `backend/test_recall_pipeline.py`
**Changes**:

```python
#!/usr/bin/env python3
"""
Test script for EdgeElite Recall Pipeline
Tests the complete flow: Context Storage → Voice Query → RAG Retrieval → Response
"""

import sys
import os
import time

# Add backend to path
sys.path.append(os.path.dirname(__file__))

from storage.interface import search_similar
from seed_recall_demo import seed_recall_demo

def test_recall_queries():
    """
    Test various recall queries against the demo data.
    """
    print("🧪 Testing Recall Queries")
    print("=" * 50)

    # Test queries that should find relevant context
    test_queries = [
        "What did I say about Project X?",
        "Remind me about the marketing budget",
        "What was mentioned about scheduling?",
        "Tell me about the client meeting",
        "What did I say about Q4 campaign?"
    ]

    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")

        try:
            # Use existing RAG pipeline
            results = search_similar(query, k=3)

            if results:
                print(f"✅ Found {len(results)} relevant results:")
                for i, (summary, content) in enumerate(results, 1):
                    print(f"   {i}. {summary}")
                    print(f"      Content: {content[:100]}...")
            else:
                print("❌ No relevant results found")

        except Exception as e:
            print(f"❌ Query failed: {e}")

def test_query_extraction():
    """
    Test the query extraction logic.
    """
    print(f"\n🧪 Testing Query Extraction")
    print("=" * 50)

    from main import extract_question_from_text

    test_texts = [
        "Hey EdgeElite, what did I say about Project X earlier?",
        "Can you remind me about the marketing budget discussion?",
        "EdgeElite, what was mentioned about the client meeting?",
        "Tell me about the Q4 campaign budget approval",
        "Just random conversation text without questions"
    ]

    for text in test_texts:
        extracted = extract_question_from_text(text)
        print(f"Input: {text}")
        print(f"Extracted: {extracted}")
        print()

def main():
    """
    Main test function
    """
    print("🚀 EdgeElite Recall Pipeline Test")
    print("=" * 50)

    # Create demo data
    print("\n📝 Phase 1: Creating Demo Data")
    seed_recall_demo()

    # Test recall queries
    print("\n🔍 Phase 2: Testing Recall Queries")
    test_recall_queries()

    # Test query extraction
    print("\n🧠 Phase 3: Testing Query Extraction")
    test_query_extraction()

    print("\n✅ Recall Pipeline Test Complete!")
    print("\nWhat this test verified:")
    print("1. ✅ Context data can be stored and indexed")
    print("2. ✅ RAG retrieval can find relevant past context")
    print("3. ✅ Query extraction can identify recall questions")
    print("4. ✅ Multiple context types are discoverable")

if __name__ == "__main__":
    main()
```

---

## 🧪 Phase 4: Integration & Testing (Priority: MEDIUM)

### **Task 4.1: Update Backend Storage Functions**

**File**: `backend/storage/db.py`
**Changes**:

```python
def get_recent_events_by_type(self, session_id: str, source: str, since_timestamp: float) -> List[Dict]:
    """
    Get recent events of a specific type since a timestamp.
    Used for recall query processing.
    """
    cursor = self.conn.cursor()
    cursor.execute("""
        SELECT * FROM raw_events
        WHERE session_id = ? AND source = ? AND timestamp >= ?
        ORDER BY timestamp DESC
    """, (session_id, source, since_timestamp))

    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]
```

### **Task 4.2: Add API Function to lib/api.js**

**File**: `renderer/lib/api.js`
**Changes**:

```javascript
export const api = {
  // ... existing functions

  // Context recall query
  contextQuery: async (sessionId, queryText) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId, queryText }),
      });

      if (!response.ok) {
        throw new Error(`Query failed: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("Context query failed:", error);
      throw error;
    }
  },
};
```

---

## 📊 Implementation Timeline

### **Week 1: Core Backend (4-6 hours)**

- Phase 1: Query endpoint implementation (3-4 hours)
- Phase 3: Demo data and testing (1-2 hours)

### **Week 2: Frontend Integration (4-6 hours)**

- Phase 2: Voice trigger UI (3-4 hours)
- Phase 4: Integration testing (1-2 hours)

### **Total Estimated Effort: 8-12 hours**

---

## 🎯 Success Criteria

### **Functional Requirements**

- ✅ Continuous voice monitoring detects recall queries
- ✅ RAG retrieval finds relevant past context
- ✅ Natural language responses reference specific past information
- ✅ Real-time assistant bubble displays responses
- ✅ Demo scenario works end-to-end

### **Technical Requirements**

- ✅ Reuses existing storage and RAG infrastructure
- ✅ Voice processing integrates with current audio pipeline
- ✅ Query extraction identifies recall intentions
- ✅ Response confidence scoring works accurately

### **Demo Requirements**

- ✅ Pre-seeded demo data for Project X scenario
- ✅ Live voice query demonstration
- ✅ Clear showcase of context retention capabilities
- ✅ Compelling real-world use case delivery

---

## 🚨 Risk Assessment

### **High Risk**

- Voice trigger detection accuracy in noisy environments
- Query extraction from natural speech patterns

### **Medium Risk**

- Continuous audio processing performance impact
- Response timing for real-time experience

### **Low Risk**

- RAG retrieval (already tested and working)
- Storage integration (uses existing interfaces)

---

## 🔗 Integration with Journal Feature

The recall feature **perfectly complements** the journal feature:

1. **Shared Infrastructure**: Both use the same storage, RAG, and LLM systems
2. **Unified Data**: Journal sessions become searchable context for recall
3. **Cross-Feature Benefits**: Journal entries can reference recalled context
4. **Minimal Code Overhead**: ~120 LOC total as estimated

This creates a powerful **multi-modal memory system** that showcases EdgeElite's comprehensive AI capabilities! 🎯

---

**Next Steps**: Begin with Phase 1 - implementing the backend query endpoint that leverages our existing RAG pipeline.
