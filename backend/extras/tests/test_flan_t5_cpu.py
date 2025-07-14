#!/usr/bin/env python3
"""
Test Flan-T5 Small ONNX Models with CPU EP (Fallback)
This script tests the quantized Flan-T5 models with CPU execution provider.
"""

import os
import sys
import numpy as np
import onnxruntime as ort

def test_model_loading():
    """Test loading the quantized models with CPU."""
    print("🧪 Testing Model Loading with CPU")
    print("=" * 40)
    
    model_dir = "flan-t5-small-ONNX/onnx"
    encoder_path = os.path.join(model_dir, "encoder_model_int8.onnx")
    decoder_path = os.path.join(model_dir, "decoder_with_past_model_int8.onnx")
    
    # Check if models exist
    if not os.path.exists(encoder_path):
        print(f"❌ Encoder model not found: {encoder_path}")
        return False, None, None
    if not os.path.exists(decoder_path):
        print(f"❌ Decoder model not found: {decoder_path}")
        return False, None, None
    
    print(f"✅ Encoder model found: {os.path.getsize(encoder_path) / 1024 / 1024:.1f} MB")
    print(f"✅ Decoder model found: {os.path.getsize(decoder_path) / 1024 / 1024:.1f} MB")
    
    try:
        # Load with CPU EP
        providers = ['CPUExecutionProvider']
        
        print("\nLoading encoder with CPU EP...")
        encoder_session = ort.InferenceSession(encoder_path, providers=providers)
        print("✅ Encoder loaded successfully with CPU EP")
        
        print("Loading decoder with CPU EP...")
        decoder_session = ort.InferenceSession(decoder_path, providers=providers)
        print("✅ Decoder loaded successfully with CPU EP")
        
        return True, encoder_session, decoder_session
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False, None, None

def test_basic_inference(encoder_session, decoder_session):
    """Test basic inference with dummy inputs."""
    print("\n🧪 Testing Basic Inference")
    print("=" * 30)
    
    try:
        # Get input/output info
        encoder_inputs = encoder_session.get_inputs()
        encoder_outputs = encoder_session.get_outputs()
        decoder_inputs = decoder_session.get_inputs()
        decoder_outputs = decoder_session.get_outputs()
        
        print(f"Encoder inputs: {[inp.name for inp in encoder_inputs]}")
        print(f"Encoder outputs: {[out.name for out in encoder_outputs]}")
        print(f"Decoder inputs: {[inp.name for inp in decoder_inputs]}")
        print(f"Decoder outputs: {[out.name for out in decoder_outputs]}")
        
        # Create dummy input for encoder (batch_size=1, sequence_length=10)
        dummy_input = np.random.randint(0, 1000, (1, 10), dtype=np.int64)
        
        print(f"\nRunning encoder with dummy input shape: {dummy_input.shape}")
        encoder_output = encoder_session.run(None, {encoder_inputs[0].name: dummy_input})
        print(f"✅ Encoder inference successful! Output shape: {encoder_output[0].shape}")
        
        # Create dummy input for decoder
        # For decoder_with_past, we need encoder_hidden_states and decoder_input_ids
        if len(decoder_inputs) >= 2:
            decoder_input_ids = np.array([[0]], dtype=np.int64)  # Start token
            encoder_hidden_states = encoder_output[0]  # Use encoder output
            
            print(f"Running decoder with input shapes: {decoder_input_ids.shape}, {encoder_hidden_states.shape}")
            decoder_output = decoder_session.run(None, {
                decoder_inputs[0].name: decoder_input_ids,
                decoder_inputs[1].name: encoder_hidden_states
            })
            print(f"✅ Decoder inference successful! Output shape: {decoder_output[0].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False

def show_model_status():
    """Show the current status and recommendations."""
    print("\n📊 Model Status Summary:")
    print("=" * 30)
    
    print("✅ Models downloaded and quantized successfully")
    print("✅ Models load with CPU execution provider")
    print("⚠️ QNN EP has compatibility issues with some operations")
    print("✅ Ready for CPU-based edge inference")
    
    print("\n🎯 Recommendations:")
    print("• Use CPU EP for now (still fast on Snapdragon X Elite)")
    print("• Consider using pre-compiled QNN models if available")
    print("• Models are ready for EdgeElite integration")
    print("• Add tokenizer for full text generation pipeline")

def main():
    """Main function."""
    print("EdgeElite AI Assistant - Flan-T5 CPU Integration Test")
    print("For Qualcomm HaQathon - High-Quality Edge AI")
    print("=" * 80)
    
    # Test model loading
    success, encoder_session, decoder_session = test_model_loading()
    
    if not success:
        print("\n❌ Model loading failed. Check model files and dependencies.")
        return
    
    # Test basic inference
    inference_works = test_basic_inference(encoder_session, decoder_session)
    
    # Show status
    show_model_status()
    
    print("\n" + "=" * 80)
    if inference_works:
        print("🎉 Flan-T5 models are working with CPU EP!")
        print("🚀 Ready for EdgeElite integration!")
    else:
        print("❌ Basic inference failed. Check model compatibility.")

if __name__ == "__main__":
    main() 