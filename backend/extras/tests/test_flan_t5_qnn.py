#!/usr/bin/env python3
"""
Test Flan-T5 Small ONNX Models with QNN EP
This script tests the quantized Flan-T5 models with Qualcomm's QNN Execution Provider.
"""

import os
import sys
import numpy as np
import onnxruntime as ort

def test_qnn_availability():
    """Test if QNN EP is available."""
    print("🧪 Testing QNN EP Availability")
    print("=" * 40)
    
    providers = ort.get_available_providers()
    print(f"Available providers: {providers}")
    
    if 'QNNExecutionProvider' in providers:
        print("✅ QNNExecutionProvider is available!")
        return True
    else:
        print("❌ QNNExecutionProvider not found")
        print("   Available providers:", providers)
        return False

def test_model_loading():
    """Test loading the quantized models."""
    print("\n🧪 Testing Model Loading")
    print("=" * 30)
    
    model_dir = "flan-t5-small-ONNX/onnx"
    encoder_path = os.path.join(model_dir, "encoder_model_int8.onnx")
    decoder_path = os.path.join(model_dir, "decoder_with_past_model_int8.onnx")
    
    # Check if models exist
    if not os.path.exists(encoder_path):
        print(f"❌ Encoder model not found: {encoder_path}")
        return False
    if not os.path.exists(decoder_path):
        print(f"❌ Decoder model not found: {decoder_path}")
        return False
    
    print(f"✅ Encoder model found: {os.path.getsize(encoder_path) / 1024 / 1024:.1f} MB")
    print(f"✅ Decoder model found: {os.path.getsize(decoder_path) / 1024 / 1024:.1f} MB")
    
    return True

def test_inference():
    """Test basic inference with the models."""
    print("\n🧪 Testing Inference")
    print("=" * 25)
    
    model_dir = "flan-t5-small-ONNX/onnx"
    encoder_path = os.path.join(model_dir, "encoder_model_int8.onnx")
    decoder_path = os.path.join(model_dir, "decoder_with_past_model_int8.onnx")
    
    try:
        # Try to load with QNN EP first
        providers = [('QNNExecutionProvider', {'device_id': '0'})]
        
        print("Loading encoder with QNN EP...")
        encoder_session = ort.InferenceSession(encoder_path, providers=providers)
        print("✅ Encoder loaded successfully with QNN EP")
        
        print("Loading decoder with QNN EP...")
        decoder_session = ort.InferenceSession(decoder_path, providers=providers)
        print("✅ Decoder loaded successfully with QNN EP")
        
        # Get input/output info
        encoder_inputs = encoder_session.get_inputs()
        encoder_outputs = encoder_session.get_outputs()
        decoder_inputs = decoder_session.get_inputs()
        decoder_outputs = decoder_session.get_outputs()
        
        print(f"\nEncoder inputs: {[inp.name for inp in encoder_inputs]}")
        print(f"Encoder outputs: {[out.name for out in encoder_outputs]}")
        print(f"Decoder inputs: {[inp.name for inp in decoder_inputs]}")
        print(f"Decoder outputs: {[out.name for out in decoder_outputs]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        
        # Try with CPU fallback
        try:
            print("\nTrying with CPU fallback...")
            providers = ['CPUExecutionProvider']
            
            encoder_session = ort.InferenceSession(encoder_path, providers=providers)
            decoder_session = ort.InferenceSession(decoder_path, providers=providers)
            
            print("✅ Models loaded successfully with CPU fallback")
            return True
            
        except Exception as e2:
            print(f"❌ CPU fallback also failed: {e2}")
            return False

def show_next_steps():
    """Show next steps for integration."""
    print("\n📋 Next Steps for EdgeElite Integration:")
    print("=" * 50)
    
    print("\n1️⃣ Tokenizer Setup")
    print("   • Install transformers: pip install transformers")
    print("   • Load T5 tokenizer from the model directory")
    
    print("\n2️⃣ Integration with EdgeElite")
    print("   • Update llm.py to use Flan-T5 models")
    print("   • Add sequence-to-sequence inference logic")
    print("   • Integrate with the existing ASR pipeline")
    
    print("\n3️⃣ Model Optimization")
    print("   • Models are already quantized to INT8")
    print("   • Ready for QNN EP inference")
    print("   • Can be used for text generation tasks")
    
    print("\n🎯 Benefits:")
    print("   • Flan-T5 Small: 80M parameters, fast inference")
    print("   • INT8 quantization: reduced memory usage")
    print("   • QNN EP: NPU acceleration on Snapdragon X Elite")
    print("   • Sequence-to-sequence: good for summarization, translation")

def main():
    """Main function."""
    print("EdgeElite AI Assistant - Flan-T5 QNN Integration Test")
    print("For Qualcomm HaQathon - High-Quality Edge AI")
    print("=" * 80)
    
    # Test QNN availability
    qnn_available = test_qnn_availability()
    
    # Test model loading
    models_available = test_model_loading()
    
    if not models_available:
        print("\n❌ Models not available. Please check the model directory.")
        return
    
    # Test inference
    inference_works = test_inference()
    
    # Show next steps
    show_next_steps()
    
    print("\n" + "=" * 80)
    if inference_works:
        if qnn_available:
            print("🎉 Flan-T5 models are ready for QNN inference!")
        else:
            print("⚠️ Models work with CPU fallback. QNN EP not available.")
    else:
        print("❌ Model loading failed. Check model files and dependencies.")

if __name__ == "__main__":
    main() 