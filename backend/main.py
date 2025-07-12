from fastapi import FastAPI
from backend.ocr import CaptureRequest, process_image
from backend.asr_macos import process_audio
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

@app.post("/capture")
async def capture(data: CaptureRequest):
    print(f"Received capture request for: {data.filename}")
    process_image(data.filename)
    return {"message": f"Processed {data.filename}"}
