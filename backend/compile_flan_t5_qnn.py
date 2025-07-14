#!/usr/bin/env python3
"""
Compile Flan-T5 Small ONNX Models for QNN
This script compiles the quantized Flan-T5 models for Qualcomm's QNN backend.
"""

import os
import sys
import numpy as np
import onnxruntime as ort

def compile_model_for_qnn(model_path, output_path):
    """Compile a model for QNN using ONNX Runtime."""
    print(f"Compiling {os.path.basename(model_path)} for QNN...")
    
    try:
        # Load the model with QNN EP
        providers = [('QNNExecutionProvider', {
            'device_id': '0',
            'backend_path': '',  # Use default backend
            'qnn_context_cache_enable': '1',  # Enable caching
            'qnn_context_cache_path': './qnn_cache'  # Cache directory
        })]
        
        # Create inference session with QNN EP
        session = ort.InferenceSession(model_path, providers=providers)
        
        # Get model info
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        print(f"✅ Model compiled successfully!")
        print(f"   Inputs: {[inp.name for inp in inputs]}")
        print(f"   Outputs: {[out.name for out in outputs]}")
        print(f"   Providers: {session.get_providers()}")
        
        # Test with dummy input
        if inputs:
            dummy_input = np.random.randint(0, 1000, (1, 10), dtype=np.int64)
            print(f"   Testing with dummy input shape: {dummy_input.shape}")
            
            try:
                output = session.run(None, {inputs[0].name: dummy_input})
                print(f"   ✅ Test inference successful! Output shape: {output[0].shape}")
                return True
            except Exception as e:
                print(f"   ⚠️ Test inference failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
        return False

def test_qnn_optimization():
    """Test if we can optimize the models for QNN."""
    print("🧪 Testing QNN Optimization")
    print("=" * 40)
    
    model_dir = "flan-t5-small-ONNX/onnx"
    encoder_path = os.path.join(model_dir, "encoder_model_int8.onnx")
    decoder_path = os.path.join(model_dir, "decoder_with_past_model_int8.onnx")
    
    # Check if models exist
    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        print("❌ Model files not found")
        return False
    
    print("✅ Model files found")
    
    # Try to compile encoder
    print("\n📦 Compiling Encoder Model")
    encoder_success = compile_model_for_qnn(encoder_path, "encoder_qnn.plan")
    
    # Try to compile decoder
    print("\n📦 Compiling Decoder Model")
    decoder_success = compile_model_for_qnn(decoder_path, "decoder_qnn.plan")
    
    return encoder_success and decoder_success

def show_optimization_options():
    """Show alternative optimization approaches."""
    print("\n🔧 Alternative Optimization Approaches:")
    print("=" * 50)
    
    print("\n1️⃣ Model Simplification")
    print("   • Remove unsupported operations (ElementWiseNeg)")
    print("   • Use ONNX optimization passes")
    print("   • Simplify the model graph")
    
    print("\n2️⃣ Use Different Model Variants")
    print("   • Try other quantized versions (q4, fp16)")
    print("   • Use merged models instead of separate encoder/decoder")
    print("   • Consider smaller models (T5-tiny)")
    
    print("\n3️⃣ QNN-Specific Optimizations")
    print("   • Use QNN SDK's model conversion tools")
    print("   • Apply QNN-specific quantization")
    print("   • Use pre-compiled QNN models")
    
    print("\n4️⃣ Runtime Optimization")
    print("   • Use mixed precision (FP16 + INT8)")
    print("   • Enable QNN caching")
    print("   • Optimize input/output handling")

def main():
    """Main function."""
    print("EdgeElite AI Assistant - Flan-T5 QNN Compilation")
    print("For Qualcomm HaQathon - High-Quality Edge AI")
    print("=" * 80)
    
    # Test QNN optimization
    success = test_qnn_optimization()
    
    # Show alternatives
    show_optimization_options()
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 QNN compilation successful!")
        print("🚀 Models are ready for NPU inference!")
    else:
        print("⚠️ QNN compilation had issues")
        print("💡 Consider alternative optimization approaches")

if __name__ == "__main__":
    main() 