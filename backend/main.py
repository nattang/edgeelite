from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.asr import process_audio

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
    result = process_audio("temp_audio.wav") 
    return {"message": result}
