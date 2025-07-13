from fastapi import FastAPI
from ocr import CaptureRequest, process_image
from asr import process_audio
#from asr_simple import process_audio
#from asr_final import process_audio
from llm import llm_service
from backend.ocr.ocr import process_image
# from backend.asr import process_audio
# from backend.llm import llm_service
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
    filename: str = ""
class CaptureRequest(BaseModel):
    filename: str
    session_id: str = Field(alias="sessionId")
    timestamp: float

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

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI!"}

import asyncio
import concurrent.futures

@app.post("/asr")
async def asr(request: ASRRequest = ASRRequest()):
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
