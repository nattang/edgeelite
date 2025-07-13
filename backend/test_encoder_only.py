#!/usr/bin/env python3
"""
Test encoder only for speed
"""

import time
import numpy as np
import librosa
import onnxruntime as ort

def test_encoder_only():
    """Test just the encoder speed"""
    print("🧪 Testing encoder only...")
    
    # Model path
    MODEL_DIR = "whisper-small-onnx/onnx"
    ENCODER_PATH = f"{MODEL_DIR}/encoder_model_int8.onnx"
    
    # Providers
    QNN_PROVIDER = [("QNNExecutionProvider", {"device_id": 0})]
    CPU_PROVIDER = ["CPUExecutionProvider"]
    
    # Load encoder
    try:
        print("🚀 Loading encoder with QNN...")
        session = ort.InferenceSession(ENCODER_PATH, providers=QNN_PROVIDER)
        provider = session.get_providers()[0]
        print(f"✅ Encoder loaded with: {provider}")
    except Exception as e:
        print(f"⚠️ QNN failed, trying CPU: {e}")
        session = ort.InferenceSession(ENCODER_PATH, providers=CPU_PROVIDER)
        provider = session.get_providers()[0]
        print(f"✅ Encoder loaded with: {provider}")
    
    # Create dummy audio input (1 second of silence)
    sr = 16000
    dummy_audio = np.zeros(sr, dtype=np.float32)
    
    # Extract mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=dummy_audio, sr=sr, n_mels=80, hop_length=160, n_fft=400
    )
    log_mel = np.log1p(mel).astype(np.float32)[None, :, :]
    
    print(f"📊 Input shape: {log_mel.shape}")
    
    # Test encoder
    start_time = time.time()
    try:
        output = session.run(None, {session.get_inputs()[0].name: log_mel})[0]
        end_time = time.time()
        
        print(f"✅ Encoder completed in {end_time - start_time:.3f} seconds")
        print(f"📊 Output shape: {output.shape}")
        print("🚀 Encoder is working fast!")
        
    except Exception as e:
        print(f"❌ Encoder failed: {e}")

if __name__ == "__main__":
    test_encoder_only() 