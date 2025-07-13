#!/usr/bin/env python3
"""
Test script to check ASR model loading and specifications
"""

import os
import onnxruntime as ort

BASE_DIR = os.path.dirname(__file__)

# Model paths
ENCODER_PATH = os.path.join(BASE_DIR, "models/whisper_large_v3_turbo-hfwhisperencoder-qualcomm_snapdragon_x_elite.onnx/model.onnx/model.onnx")
DECODER_PATH = os.path.join(BASE_DIR, "models/whisper_large_v3_turbo-hfwhisperdecoder-qualcomm_snapdragon_x_elite.onnx/model.onnx/model.onnx")

def test_models():
    print("🧪 Testing ASR Model Loading")
    print("=" * 50)
    
    # Check if files exist
    print(f"Encoder path: {ENCODER_PATH}")
    print(f"Encoder exists: {os.path.exists(ENCODER_PATH)}")
    print(f"Decoder path: {DECODER_PATH}")
    print(f"Decoder exists: {os.path.exists(DECODER_PATH)}")
    print()
    
    # Test encoder
    print("🔧 Testing Encoder Model...")
    try:
        encoder_sess = ort.InferenceSession(ENCODER_PATH, providers=["CPUExecutionProvider"])
        print("✅ Encoder loaded successfully!")
        print(f"Providers: {encoder_sess.get_providers()}")
        print(f"Active provider: {encoder_sess.get_provider_options()}")
        
        print("\nEncoder Inputs:")
        for inp in encoder_sess.get_inputs():
            print(f"  {inp.name}: {inp.shape} ({inp.type})")
        
        print("\nEncoder Outputs:")
        for out in encoder_sess.get_outputs():
            print(f"  {out.name}: {out.shape} ({out.type})")
            
    except Exception as e:
        print(f"❌ Encoder loading failed: {e}")
        return False
    
    print()
    
    # Test decoder
    print("🔧 Testing Decoder Model...")
    try:
        decoder_sess = ort.InferenceSession(DECODER_PATH, providers=["CPUExecutionProvider"])
        print("✅ Decoder loaded successfully!")
        print(f"Providers: {decoder_sess.get_providers()}")
        print(f"Active provider: {decoder_sess.get_provider_options()}")
        
        print("\nDecoder Inputs:")
        for inp in decoder_sess.get_inputs():
            print(f"  {inp.name}: {inp.shape} ({inp.type})")
        
        print("\nDecoder Outputs:")
        for out in decoder_sess.get_outputs():
            print(f"  {out.name}: {out.shape} ({out.type})")
            
    except Exception as e:
        print(f"❌ Decoder loading failed: {e}")
        return False
    
    print()
    print("🎉 All models loaded successfully!")
    return True

if __name__ == "__main__":
    test_models() 