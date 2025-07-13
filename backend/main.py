from fastapi import FastAPI
from backend.ocr import CaptureRequest, process_image
from backend.asr import process_audio
from backend.llm import llm_service
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os

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

@app.post("/asr")
async def asr():
    print("🎤 Received ASR trigger request")
    
    # Look for the most recent audio file in the recordings folder
    recordings_dir = os.path.join(os.path.expanduser("~"), "EdgeElite", "recordings")
    
    if os.path.exists(recordings_dir):
        # Get the most recent .wav file
        wav_files = [f for f in os.listdir(recordings_dir) if f.endswith('.wav')]
        if wav_files:
            # Sort by modification time (newest first)
            wav_files.sort(key=lambda x: os.path.getmtime(os.path.join(recordings_dir, x)), reverse=True)
            latest_audio_file = os.path.join(recordings_dir, wav_files[0])
            print(f"🎤 Processing latest audio file: {latest_audio_file}")
            result = process_audio(latest_audio_file)
        else:
            print("🎤 No audio files found in recordings directory")
            result = "No audio file found"
    else:
        print("🎤 Recordings directory not found")
        result = "Recordings directory not found"
    
    return {"message": result}

@app.post("/capture")
async def capture(data: CaptureRequest):
    print(f"Received capture request for: {data.filename}")
    process_image(data.filename)
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
