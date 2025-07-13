from fastapi import FastAPI
from backend.ocr.ocr import process_image
from backend.asr import process_audio
from backend.llm import llm_service
from backend.asr_final import process_audio
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os

import time
from backend.storage.interface import (
    store_raw_event,
    process_session,
    search_similar,
    get_session_stats,
    get_system_stats,
    clear_all_data
)

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

class SessionEndRequest(BaseModel):
    session_id: str = Field(alias="sessionId")
    
    class Config:
        allow_population_by_field_name = True

class JournalRequest(BaseModel):
    session_id: str = Field(alias="sessionId")
    
    class Config:
        allow_population_by_field_name = True

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
        journal_cache[session_id] = {
            "summary_action": response,
            "related_memory": remedy_context[:200] if remedy_context else None
        }
        
        print(f"✅ Journal pipeline completed for session: {session_id}")
        
    except Exception as e:
        print(f"❌ Journal pipeline error for session {session_id}: {e}")
        journal_cache[session_id] = {"error": str(e)}

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI!"}

@app.post("/asr") #FOR ASR CAPTURE
async def asr(request: ASRRequest):
    import time
    from storage.interface import store_raw_audio_event
    
    print("🎤 Received ASR trigger request")
    
    recordings_dir = os.path.join(os.path.expanduser("~"), "EdgeElite", "recordings")
    msg = ""

    if os.path.exists(recordings_dir):
        wav_files = [f for f in os.listdir(recordings_dir) if f.endswith('.wav')]
        if wav_files:
            wav_files.sort(key=lambda x: os.path.getmtime(os.path.join(recordings_dir, x)), reverse=True)
            latest_audio_file = os.path.join(recordings_dir, wav_files[0])
            print(f"🎤 Processing latest audio file: {latest_audio_file}")
            result = process_audio(latest_audio_file)
            print(result)
            msg = " ".join([r["text"] for r in result]).strip()
            print(f"🎤 Transcription result: {msg}")
        else:
            print("🎤 No audio files found in recordings directory")
            msg = "No audio file found"
    else:
        print("🎤 Recordings directory not found")
        msg = "Recordings directory not found"
    
    return {"message": msg}

@app.post("/capture") #FOR OCR CAPTURE
async def capture(data: CaptureRequest):
    import time
    from storage.interface import store_raw_ocr_event
    
    print(f"Received capture request for: {data.filename}")
    # TODO: add processed image to database w session id
    message = process_image(data.filename)
    event_id = store_raw_event(
        session_id=data.session_id,
        source="ocr",
        ts=data.timestamp,
        text=message,
        metadata={"screen_region": "main_editor"}
    )
    print(f"   Stored OCR event: {event_id[:8]}... -> '{message[:30]}...'")
    return {"message": f"Text: {message}"}

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
