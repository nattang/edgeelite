from fastapi import FastAPI
from asr import process_audio
from llm import LLMService
from ocr.ocr import process_image
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os
import json
import datetime

import time
from backend.storage.interface import (
    store_raw_event,
    process_session,
    search_similar,
    get_session_stats,
    get_system_stats,
    clear_all_data
)

# Create LLM service instance
llm_service = LLMService()

class QueryRequest(BaseModel):
    session_id: str = Field(alias="sessionId")
    user_input: str = Field(alias="userInput")
    context: List[Dict[str, Any]] = []

    class Config:
        allow_population_by_field_name = True

class EventRequest(BaseModel):
    session_id: str = Field(alias="sessionId")
    source: str
    text: str
    metadata: Dict[str, Any] = {}

    class Config:
        allow_population_by_field_name = True

class ContextRequest(BaseModel):
    session_id: str = Field(alias="sessionId")
    count: int = 10

    class Config:
        allow_population_by_field_name = True

class ASRRequest(BaseModel):
    session_id: str = Field(alias="sessionId")
    
    class Config:
        allow_population_by_field_name = True

class CaptureRequest(BaseModel):
    filename: str
    session_id: str = Field(alias="sessionId")
    timestamp: float

    class Config:
        allow_population_by_field_name = True

class SessionEndRequest(BaseModel):
    session_id: str = Field(alias="sessionId")
    
    class Config:
        allow_population_by_field_name = True

class JournalRequest(BaseModel):
    session_id: str = Field(alias="sessionId")
    
    class Config:
        allow_population_by_field_name = True

class RecallRequest(BaseModel):
    session_id: str = Field(alias="sessionId")
    query_text: str = Field(alias="queryText")
    
    class Config:
        allow_population_by_field_name = True

class RecallResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    session_id: str

app = FastAPI()

origins = [
    "http://localhost:8888",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# This is run when the user ends the session from frontend
# Journal processing cache
journal_cache = {}

# Journal entries file path
JOURNAL_ENTRIES_FILE = "journal_entries.json"

def save_journal_entry(session_id: str, entry: Dict[str, Any]):
    """
    Save a journal entry to the JSON file.
    """
    try:
        # Read existing entries
        if os.path.exists(JOURNAL_ENTRIES_FILE):
            with open(JOURNAL_ENTRIES_FILE, 'r') as f:
                entries = json.load(f)
        else:
            entries = []
        
        # Create new entry
        new_entry = {
            "session_id": session_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "summary_action": entry.get("summary_action", ""),
            "related_memory": entry.get("related_memory", None),
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.datetime.now().strftime("%H:%M:%S")
        }
        
        # Add to entries list (newest first)
        entries.insert(0, new_entry)
        
        # Save back to file
        with open(JOURNAL_ENTRIES_FILE, 'w') as f:
            json.dump(entries, f, indent=2)
        
        print(f"✅ Journal entry saved for session: {session_id}")
        
    except Exception as e:
        print(f"❌ Error saving journal entry: {e}")

def load_journal_entries():
    """
    Load all journal entries from the JSON file.
    """
    try:
        if os.path.exists(JOURNAL_ENTRIES_FILE):
            with open(JOURNAL_ENTRIES_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"❌ Error loading journal entries: {e}")
        return []

async def run_journal_pipeline(session_id: str):
    """
    Process a session and generate journal entry with RAG.
    
    Steps:
    1. Process session (clean, chunk, embed)
    2. Get current session text
    3. Search for similar past sessions
    4. Use most relevant past experience as context
    5. Generate LLM response with personalized guidance
    6. Cache result for frontend
    """
    try:
        print(f"🔄 Starting journal pipeline for session: {session_id}")
        
        # 1. Process session using storage system -> This triggers the backend processing process
        from storage.interface import process_session
        node_ids = process_session(session_id)
        print(f"📊 Session processed: {len(node_ids)} nodes created")
        
        # 2. Get current session text for RAG
        from storage.db import StorageDB
        db = StorageDB()
        raw_events = db.get_raw_events_by_session(session_id)
        full_doc = "\n".join(event["text"] for event in raw_events 
                            if event["source"] in ("asr", "ocr"))
        
        # 3. Search for similar past sessions (exclude current session)
        from storage.interface import search_similar
        
        ##TODO: Doing RAG on the entire session document. Check if this gives good results.
        similar_results = search_similar(full_doc, k=3)
        
        # 4. Use most relevant similar experience as context
        remedy_context = ""
        if similar_results:
            # Take the most semantically similar result as context
            top_result = similar_results[0]  # Most similar by FAISS ranking
            remedy_context = top_result[1]   # Full content - this is all we need
        
        # 5. Generate journal entry with LLM
        if remedy_context:
            prompt = f"""
            Current session:
            ```{full_doc}```
            
            This is the most relevant past experience that could be helpful for this results:
            ```{remedy_context}```
            
            Task: Analyze the current session and provide:
            1) A brief summary of the current situation and emotions (1-2 sentences)
            2) Actionable guidance that draws insight from the related past experience
            
            If the past experience contains a successful approach or solution, reference it specifically.
            Keep response under 120 words and make it personal and actionable.
            """
        else:
            prompt = f"""
            Current session:
            ```{full_doc}```
            
            Task: Analyze this session and provide:
            1) A brief summary of the current situation and emotions (1-2 sentences)  
            2) Thoughtful, actionable guidance based on the content
            
            Keep response under 120 words and make it personal and actionable.
            """
        
        response = llm_service.generate_response(prompt, [])
        
        # 6. Cache for frontend polling
        journal_entry = {
            "summary_action": response,
            "related_memory": remedy_context[:200] if remedy_context else None
        }
        journal_cache[session_id] = journal_entry
        
        # 7. Save to JSON file for persistent storage
        save_journal_entry(session_id, journal_entry)
        
        print(f"✅ Journal pipeline completed for session: {session_id}")
        
    except Exception as e:
        print(f"❌ Journal pipeline error for session {session_id}: {e}")
        journal_cache[session_id] = {"error": str(e)}

def extract_question_from_text(text: str) -> str:
    """
    Extract the actual question from voice input.
    Looks for patterns like "What did I say about X" or "Remind me about Y"
    """
    import re
    
    # Common recall question patterns
    patterns = [
        r"what did i say about (.+?)(?:\?|$)",
        r"remind me (?:about|what) (.+?)(?:\?|$)", 
        r"what was (?:mentioned|said) about (.+?)(?:\?|$)",
        r"tell me about (.+?)(?:\?|$)",
        r"recall (.+?)(?:\?|$)",
        r"edgeelite.*?about (.+?)(?:\?|$)"
    ]
    
    text_lower = text.lower()
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            topic = match.group(1).strip()
            return f"What was mentioned about {topic}?"
    
    # If no pattern matched, return the original text
    return text

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI!"}

@app.get("/health")
def health_check():
    # Check OCR models
    import os
    ocr_models_dir = os.path.join(os.path.dirname(__file__), "models", "ocr")
    detector_exists = os.path.exists(os.path.join(ocr_models_dir, "easyocr-easyocrdetector.onnx"))
    recognizer_exists = os.path.exists(os.path.join(ocr_models_dir, "easyocr-easyocrrecognizer.onnx"))
    
    return {
        "status": "healthy",
        "backend": "running",
        "llm": "Flan-T5 loaded" if llm_service.model_loaded else "mock mode",
        "asr": "QNN NPU optimized",
        "ocr": "ONNX models ready" if (detector_exists and recognizer_exists) else "EasyOCR fallback"
    }

import asyncio
import concurrent.futures

@app.post("/asr")
async def asr(request: ASRRequest):
    import time
    from storage.interface import store_raw_audio_event
    
    print("🎤 Received ASR trigger request")
    
    try:
        recordings_dir = os.path.join(os.path.expanduser("~"), "EdgeElite", "recordings")
        msg = ""

        if os.path.exists(recordings_dir):
            wav_files = [f for f in os.listdir(recordings_dir) if f.endswith('.wav')]
            if wav_files:
                wav_files.sort(key=lambda x: os.path.getmtime(os.path.join(recordings_dir, x)), reverse=True)
                latest_audio_file = os.path.join(recordings_dir, wav_files[0])
                print(f"🎤 Processing latest audio file: {latest_audio_file}")
                
                try:
                    # Run ASR processing with timeout to prevent hanging
                    loop = asyncio.get_event_loop()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = loop.run_in_executor(executor, process_audio, latest_audio_file)
                        result = await asyncio.wait_for(future, timeout=30.0)  # 30 second timeout
                    
                    print(f"ASR result: {result}")
                    msg = " ".join([r["text"] for r in result]).strip()
                    print(f"🎤 Transcription result: {msg}")
                    
                    # Store using correct function (audio expects list format)
                    audio_data = [{
                        "timestamp": time.time(),
                        "text": msg,
                        "audio_file": latest_audio_file
                    }]
                    store_raw_audio_event(
                        session_id=request.session_id,
                        source="audio",
                        audio_data=audio_data
                    )
                    
                except asyncio.TimeoutError:
                    print("❌ ASR processing timed out after 30 seconds")
                    msg = "ASR processing timed out - please try again"
                except Exception as asr_error:
                    print(f"❌ ASR processing error: {asr_error}")
                    import traceback
                    traceback.print_exc()
                    msg = f"ASR Error: {str(asr_error)}"
            else:
                print("🎤 No audio files found in recordings directory")
                msg = "No audio file found"
        else:
            print("🎤 Recordings directory not found")
            msg = "Recordings directory not found"
        
        return {"message": msg}
        
    except Exception as e:
        print(f"❌ ASR endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return {"message": f"ASR Error: {str(e)}"}

@app.post("/capture")
async def capture(data: CaptureRequest):
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

@app.post("/api/query")
async def query_llm(request: QueryRequest):
    print(f"🤖 Received LLM query for session: {request.session_id}")
    
    try:
        response = llm_service.generate_response(
            request.user_input, 
            request.context
        )
        return {"response": response, "session_id": request.session_id}
    except Exception as e:
        print(f"LLM query error: {e}")
        return {"error": str(e), "session_id": request.session_id}

@app.post("/api/events")
async def store_event(request: EventRequest):
    print(f"📝 Received event storage request for session: {request.session_id}")
    
    try:
        # TODO: Person 3 will implement actual storage
        # For now, just log the event
        print(f"Event: {request.source} - {request.text[:50]}...")
        
        return {
            "event_id": f"event_{request.session_id}_{len(request.text)}",
            "status": "stored",
            "message": "Event stored (mock mode)"
        }
    except Exception as e:
        print(f"Event storage error: {e}")
        return {"error": str(e), "session_id": request.session_id}

@app.post("/api/context")
async def get_context(request: ContextRequest):
    print(f"🔍 Received context request for session: {request.session_id}")
    
    try:
        # TODO: Person 3 will implement actual context retrieval
        # For now, return mock context
        mock_context = [
            {
                "id": "mock_event_1",
                "session_id": request.session_id,
                "source": "ocr",
                "text": "Mock screenshot content",
                "metadata": {"timestamp": "2025-07-12T22:00:00Z"}
            },
            {
                "id": "mock_event_2", 
                "session_id": request.session_id,
                "source": "asr",
                "text": "Mock audio transcription",
                "metadata": {"timestamp": "2025-07-12T22:01:00Z"}
            }
        ]
        
        return {
            "session_id": request.session_id,
            "context": mock_context,
            "count": len(mock_context),
            "message": "Context retrieved (mock mode)"
        }
    except Exception as e:
        print(f"Context retrieval error: {e}")
        return {"error": str(e), "session_id": request.session_id}

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

@app.post("/api/recall")
async def handle_recall(request: RecallRequest):
    """
    Handle context recall queries using RAG pipeline.
    Returns immediate response without storing anything.
    """
    try:
        session_id = request.session_id
        query_text = request.query_text
        
        print(f"🔍 Received recall query for session: {session_id}")
        print(f"🎯 Query text: {query_text}")
        
        # Extract question from query text
        extracted_query = extract_question_from_text(query_text)
        print(f"🧠 Extracted query: {extracted_query}")
        
        # Search for relevant context using existing RAG pipeline
        from storage.interface import search_similar
        search_results = search_similar(extracted_query, k=5)
        
        print(f"📚 Found {len(search_results)} relevant results")
        
        # Generate response using LLM
        if search_results:
            # Build context from search results
            context_sections = []
            for i, (summary, content) in enumerate(search_results, 1):
                context_sections.append(f"Context {i}: {content}")
            
            context_text = "\n\n".join(context_sections)
            
            prompt = f"""
            User just asked: "{extracted_query}"
            
            Relevant information from earlier:
            {context_text}
            
            Task: Write a short, helpful answer to the user's question using the context above.
            - Be conversational and natural
            - Reference specific details from the context
            - If the context doesn't fully answer the question, say so
            - Keep response under 50 words
            """
            
            answer = llm_service.generate_response(prompt, [])
        else:
            answer = "I don't have any relevant information about that topic from your previous conversations."
        
        # Format sources for frontend
        sources = []
        for summary, content in search_results:
            sources.append({
                "summary": summary,
                "content": content[:200] + "..." if len(content) > 200 else content
            })
        
        return RecallResponse(
            answer=answer,
            sources=sources,
            confidence=0.8 if search_results else 0.2,
            session_id=session_id
        )
        
    except Exception as e:
        print(f"❌ Recall processing error: {e}")
        return RecallResponse(
            answer="I'm sorry, I encountered an error while trying to recall that information.",
            sources=[],
            confidence=0.0,
            session_id=request.session_id
        )

@app.get("/api/journal/entries")
async def get_all_journal_entries():
    """
    Get all journal entries from the JSON file.
    """
    try:
        entries = load_journal_entries()
        return {"entries": entries, "count": len(entries)}
    except Exception as e:
        print(f"❌ Error fetching journal entries: {e}")
        return {"entries": [], "count": 0, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting EdgeElite Backend Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("🎤 ASR: QNN NPU optimized Whisper")
    
    # Pre-load LLM models
    print("🤖 Loading LLM models...")
    try:
        llm_service.load_model()
        if hasattr(llm_service, 'flan_t5_service') and llm_service.flan_t5_service and llm_service.flan_t5_service.model_loaded:
            print("🤖 LLM: Flan-T5 Small (80M parameters, INT8 quantized)")
        elif llm_service.model_loaded:
            print("🤖 LLM: Local models loaded successfully")
        else:
            print("🤖 LLM: Using enhanced mock responses")
    except Exception as e:
        print(f"🤖 LLM: Error loading models: {e}")
        print("🤖 LLM: Using enhanced mock responses")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
