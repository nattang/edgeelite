#!/usr/bin/env python3
"""
Download Whisper ONNX Models for EdgeElite ASR

This script downloads pre-converted Whisper ONNX models (encoder and decoder)
into backend/models/whisper-large-v3/ for fully offline, on-device ASR.
"""

import os
from pathlib import Path
import requests

ONNX_MODELS = {
    "encoder_model.onnx": "https://huggingface.co/optimum/whisper-tiny.en/resolve/main/encoder_model.onnx",
    "decoder_model_merged.onnx": "https://huggingface.co/optimum/whisper-tiny.en/resolve/main/decoder_model_merged.onnx"
}


def download_onnx_models():
    models_dir = Path(__file__).parent / "models" / "whisper-large-v3"
    models_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in ONNX_MODELS.items():
        file_path = models_dir / filename
        if file_path.exists():
            print(f"✅ {filename} already exists.")
            continue
        print(f"📥 Downloading {filename}...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"✅ {filename} downloaded.")
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")

if __name__ == "__main__":
    print("EdgeElite ASR - Whisper ONNX Model Downloader")
    download_onnx_models()
    print("\nDone. Please check backend/models/whisper-large-v3/ for ONNX files.") 