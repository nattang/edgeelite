from fastapi import FastAPI
from backend.ocr import CaptureRequest, process_image
from backend.asr import process_audio
from fastapi.middleware.cors import CORSMiddleware
import os


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
