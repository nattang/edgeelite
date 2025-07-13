from fastapi import FastAPI, Request
from pydantic import BaseModel
import onnxruntime as ort 

class CaptureRequest(BaseModel):
    filename: str

def process_image(filename):
    OUTPUT_TEXT_PATH = "temp_data/ocr_output.txt"
    print(filename)