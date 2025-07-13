from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
import onnxruntime as ort 

class CaptureRequest(BaseModel):
    filename: str
    session_id: str = Field(alias="sessionId")
    
    class Config:
        allow_population_by_field_name = True

def process_image(filename):
    OUTPUT_TEXT_PATH = "temp_data/ocr_output.txt"
    print(filename)