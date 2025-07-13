#!/usr/bin/env python3
"""
Test script to debug ASR module
"""

import os
import onnxruntime as ort

# Model paths
BASE_DIR = os.path.dirname(__file__)
ENCODER_PATH = os.path.join(BASE_DIR, "models/whisper-large-v3/encoder_model.onnx")
DECODER_PATH = os.path.join(BASE_DIR, "models/whisper_large_v3_turbo-hfwhisperdecoder-qualcomm_snapdragon_x_elite.onnx/model.onnx/model.onnx")

print("=== ASR Debug Test ===")
print(f"Base dir: {BASE_DIR}")
print(f"Encoder path: {ENCODER_PATH}")
print(f"Decoder path: {DECODER_PATH}")
print(f"Encoder exists: {os.path.exists(ENCODER_PATH)}")
print(f"Decoder exists: {os.path.exists(DECODER_PATH)}")

try:
    print("\n=== Testing Encoder ===")
    encoder_sess = ort.InferenceSession(ENCODER_PATH, providers=["CPUExecutionProvider"])
    print("✅ Encoder session created successfully")
    
    print("Encoder inputs:")
    for inp in encoder_sess.get_inputs():
        print("  ", inp.name, inp.shape, inp.type)
    print("Encoder outputs:")
    for out in encoder_sess.get_outputs():
        print("  ", out.name, out.shape, out.type)
        
except Exception as e:
    print(f"❌ Encoder error: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n=== Testing Decoder ===")
    decoder_sess = ort.InferenceSession(DECODER_PATH, providers=["CPUExecutionProvider"])
    print("✅ Decoder session created successfully")
    
    print("Decoder inputs:")
    for inp in decoder_sess.get_inputs():
        print("  ", inp.name, inp.shape, inp.type)
    print("Decoder outputs:")
    for out in decoder_sess.get_outputs():
        print("  ", out.name, out.shape, out.type)
        
except Exception as e:
    print(f"❌ Decoder error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test Complete ===") 