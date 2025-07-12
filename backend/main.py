from fastapi import FastAPI
from backend.ocr import CaptureRequest, process_image
from backend.asr import process_audio
from fastapi.middleware.cors import CORSMiddleware


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
@app.post("/capture")
async def capture(data: CaptureRequest):
    print(f"Received capture request for: {data.filename}")
    process_image(data.filename)
    return {"message": f"Processed {data.filename}"}
