from fastapi import FastAPI
from backend.ocr.ocr import process_image
from fastapi.middleware.cors import CORSMiddleware
import base64
import io
from PIL import Image
import os
from pathlib import Path
from pydantic import BaseModel  

app = FastAPI()

origins = [
    "http://localhost:8888",  # frontend origin (where fetch is called)
    # you can add more origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # or ["*"] to allow all origins (for dev only)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI!"}

class CaptureRequest(BaseModel):
    filename: str

@app.post("/capture")
async def capture(data: CaptureRequest):
    print(f"Received capture request for: {data.filename}")
    process_image(data.filename)
    return {"message": f"Processed {data.filename}"}