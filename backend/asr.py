from pydantic import BaseModel
import time

class ASRRequest(BaseModel):
    filename: str

def process_audio(filename: str) -> str:
    print(f"Processing audio file: {filename}")

    time.sleep(2)

    # call model here later
    # for now, we simulate a transcription result
    transcript = f"Transcription result for {filename}"

    print(f"Done processing: {transcript}")
    return transcript