from fastapi import FastAPI, Request
from pydantic import BaseModel
import onnxruntime as ort 
app = FastAPI()

class CaptureRequest(BaseModel):
    filename: str

OUTPUT_TEXT_PATH = "temp_data/ocr_output.txt"

@app.post("/capture")
async def capture(data: CaptureRequest):
    print(f"Received capture request for: {data.filename}")
    return {"message": f"Processed {data.filename}"}
    