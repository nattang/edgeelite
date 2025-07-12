# import numpy as np
# import onnxruntime as ort
# import soundfile as sf
# import librosa
# import os

# CHUNK_DURATION = 2.0  # seconds
# MODEL_PATH = "models/asr_model.onnx"  # replace with model path
# SAMPLING_RATE = 16000

# session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# def read_audio(filename, sr=SAMPLING_RATE):
#     audio, file_sr = sf.read(filename)
#     if file_sr != sr:
#         audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
#     return audio

# def chunk_audio(audio, sr=SAMPLING_RATE, chunk_duration=CHUNK_DURATION):
#     chunk_size = int(sr * chunk_duration)
#     return [(i / sr, (i + chunk_size) / sr, audio[i:i + chunk_size])
#             for i in range(0, len(audio), chunk_size)]

# def run_inference(chunk):
#     # replace with actual preprocessing if needed
#     input_data = np.expand_dims(chunk, axis=0).astype(np.float32)
#     inputs = {session.get_inputs()[0].name: input_data}
#     outputs = session.run(None, inputs)
#     return outputs

# def process_audio(filename):
#     audio = read_audio(filename)
#     chunks = chunk_audio(audio)
#     print(f"Processing {len(chunks)} audio chunks")

#     results = []
#     for start_ts, end_ts, chunk in chunks:
#         output = run_inference(chunk)
#         text = decode_output(output)  # implement this based on model's output format
#         results.append({
#             "start": round(start_ts, 2),
#             "end": round(end_ts, 2),
#             "text": text
#         })
#     try:
#         os.remove(filename)
#         print(f"🗑️ Deleted temporary audio file: {filename}")
#     except Exception as e:
#         print(f"⚠️ Failed to delete file: {e}")
#     return results

# def decode_output(output):
#     # replace with actual decoding logic based on model's output
#     return "[decoded text]"

# if __name__ == "__main__":
#     # for testing
#     from pprint import pprint
#     result = process_audio("temp_audio.wav")
#     pprint(result)

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