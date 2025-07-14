#!/usr/bin/env python3
"""
Simple test to check ONNX model loading
"""

import os
import onnxruntime as ort

print("🔍 Simple ASR Model Test")
print("=" * 30)

# Check file sizes
encoder_path = "models/whisper_large_v3_turbo-hfwhisperencoder-qualcomm_snapdragon_x_elite.onnx/model.onnx/model.onnx"
decoder_path = "models/whisper_large_v3_turbo-hfwhisperdecoder-qualcomm_snapdragon_x_elite.onnx/model.onnx/model.onnx"

print(f"Encoder file size: {os.path.getsize(encoder_path) if os.path.exists(encoder_path) else 'NOT FOUND'} bytes")
print(f"Decoder file size: {os.path.getsize(decoder_path) if os.path.exists(decoder_path) else 'NOT FOUND'} bytes")

print("\n🔧 Testing encoder...")
try:
    print("Creating encoder session...")
    encoder_sess = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
    print("✅ Encoder loaded!")
except Exception as e:
    print(f"❌ Encoder failed: {e}")
    import traceback
    traceback.print_exc()

print("\n🔧 Testing decoder...")
try:
    print("Creating decoder session...")
    decoder_sess = ort.InferenceSession(decoder_path, providers=["CPUExecutionProvider"])
    print("✅ Decoder loaded!")
except Exception as e:
    print(f"❌ Decoder failed: {e}")
    import traceback
    traceback.print_exc()

print("\n�� Test complete!") 