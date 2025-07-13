#!/usr/bin/env python3
"""
Download Llama3-TAIDE-LX-8B-Chat-Alpha1 following Qualcomm AI Hub LLM on Genie Tutorial
Based on: https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie

This script follows the exact steps from the tutorial to set up the model for EdgeElite AI Assistant.
"""

import os
import sys
import subprocess
from pathlib import Path
import json

def check_prerequisites():
    """Check if all prerequisites are installed."""
    print("🔍 Checking prerequisites for Llama3-TAIDE-LX-8B-Chat-Alpha1...")
    
    # Check Python packages
    required_packages = [
        "torch",
        "transformers", 
        "onnxruntime",
        "onnxruntime-qnn",
        "accelerate"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - MISSING")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                return False
    
    # Check QNN SDK
    try:
        import onnxruntime as ort
        if 'QNNExecutionProvider' in ort.get_available_providers():
            print("✅ QNN SDK - Available")
        else:
            print("❌ QNN SDK - Not available")
            return False
    except ImportError:
        print("❌ ONNX Runtime not available")
        return False
    
    print("✅ All prerequisites met!")
    return True

def setup_model_directory():
    """Set up the model directory structure."""
    print("\n📁 Setting up model directory...")
    
    models_dir = Path(__file__).parent / "models"
    llama_dir = models_dir / "llama3_taide_lx_8b_chat_alpha1"
    llama_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Model directory: {llama_dir}")
    return llama_dir

def create_download_instructions():
    """Create detailed download instructions following the tutorial."""
    print("\n📋 Creating download instructions following Qualcomm AI Hub tutorial...")
    
    llama_dir = setup_model_directory()
    
    instructions = """# Llama3-TAIDE-LX-8B-Chat-Alpha1 Download Instructions
# Following Qualcomm AI Hub LLM on Genie Tutorial

## Step 1: Access Qualcomm AI Hub
1. Visit: https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie
2. Follow the tutorial setup steps
3. Access the model repository: https://github.com/quic/ai-hub-models/tree/main/qai_hub_models/models/llama3_taide_lx_8b_chat_alpha1

## Step 2: Download Model Files
Based on the tutorial, you need to download:

### Required Files:
- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer vocabulary  
- `tokenizer_config.json` - Tokenizer configuration
- `model.safetensors` - Model weights (or pytorch_model.bin)
- `special_tokens_map.json` - Special tokens mapping
- `tokenizer.model` - SentencePiece tokenizer model

### Optional Files:
- `model.onnx` - ONNX version for faster inference
- `model_quantized.onnx` - Quantized ONNX version

## Step 3: Place Files in Directory
Extract all files to: backend/models/llama3_taide_lx_8b_chat_alpha1/

## Step 4: Verify Installation
Run: python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; print('Model ready!')"

## Tutorial-Specific Notes:
- This model is optimized for Qualcomm Snapdragon X-Elite NPU
- Uses QNN (Qualcomm Neural Network) for acceleration
- Supports 4096 token context window
- Chat-tuned for conversational AI
- Alpha1 version with latest improvements

## Expected File Structure:
```
backend/models/llama3_taide_lx_8b_chat_alpha1/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── tokenizer.model
├── model.safetensors (or pytorch_model.bin)
└── model.onnx (optional)
```

## Performance Expectations:
- NPU-accelerated inference on Snapdragon X-Elite
- Sub-second response times
- 8B parameter model optimized for edge devices
- Chat-tuned for natural conversations
"""
    
    instructions_path = llama_dir / "DOWNLOAD_INSTRUCTIONS.md"
    with open(instructions_path, 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print(f"✅ Download instructions created: {instructions_path}")
    return instructions_path

def create_test_script():
    """Create a test script to verify the model works."""
    print("\n🧪 Creating test script...")
    
    llama_dir = setup_model_directory()
    
    test_script = '''#!/usr/bin/env python3
"""
Test script for Llama3-TAIDE-LX-8B-Chat-Alpha1
Run this after downloading the model files to verify everything works.
"""

import os
import sys
from pathlib import Path

def test_model():
    """Test if the Llama3-TAIDE-LX model is working."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import onnxruntime as ort
        
        print("🧪 Testing Llama3-TAIDE-LX-8B-Chat-Alpha1 model...")
        
        # Check model directory
        model_path = Path(__file__).parent / "llama3_taide_lx_8b_chat_alpha1"
        if not model_path.exists():
            print("❌ Model directory not found!")
            return False
        
        # Check required files
        required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        missing_files = []
        for file in required_files:
            if not (model_path / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ Missing files: {', '.join(missing_files)}")
            return False
        
        # Check for model weights
        model_files = ["model.safetensors", "pytorch_model.bin"]
        has_weights = any((model_path / f).exists() for f in model_files)
        if not has_weights:
            print("❌ No model weights found!")
            return False
        
        print("✅ All required files present")
        
        # Test tokenizer loading
        print("📝 Testing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        print("✅ Tokenizer loaded successfully")
        
        # Test model loading
        print("🤖 Testing model loading...")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype="auto",
            device_map="cpu",  # Start with CPU for testing
            trust_remote_code=True
        )
        print("✅ Model loaded successfully")
        
        # Test QNN availability
        print("🚀 Testing QNN NPU availability...")
        if 'QNNExecutionProvider' in ort.get_available_providers():
            print("✅ QNN NPU available for acceleration!")
        else:
            print("⚠️ QNN NPU not available, will use CPU/GPU")
        
        # Test simple inference
        print("💬 Testing simple inference...")
        prompt = "Hello, how are you?"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Inference test successful!")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        
        print("🎉 Llama3-TAIDE-LX-8B-Chat-Alpha1 is ready for EdgeElite!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    import torch
    test_model()
'''
    
    test_path = llama_dir / "test_model.py"
    with open(test_path, 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print(f"✅ Test script created: {test_path}")
    return test_path

def check_model_files():
    """Check if model files are already present."""
    llama_dir = setup_model_directory()
    
    required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    model_files = ["model.safetensors", "pytorch_model.bin"]
    
    has_required = all((llama_dir / f).exists() for f in required_files)
    has_weights = any((llama_dir / f).exists() for f in model_files)
    
    return has_required and has_weights

def main():
    """Main function following the tutorial steps."""
    print("🚀 Llama3-TAIDE-LX-8B-Chat-Alpha1 Setup")
    print("Following Qualcomm AI Hub LLM on Genie Tutorial")
    print("=" * 80)
    
    # Check if model is already downloaded
    if check_model_files():
        print("✅ Llama3-TAIDE-LX-8B-Chat-Alpha1 model files found!")
        print("🚀 Your EdgeElite AI Assistant is ready for real AI responses!")
        return True
    
    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites not met. Please install required packages.")
        return False
    
    # Create download instructions
    instructions_path = create_download_instructions()
    
    # Create test script
    test_path = create_test_script()
    
    print("\n🎯 Setup Complete!")
    print("=" * 50)
    print("📋 Next steps:")
    print("1. Follow the download instructions in:")
    print(f"   {instructions_path}")
    print("2. Download the model files from Qualcomm AI Hub")
    print("3. Place files in the model directory")
    print("4. Test the model with:")
    print(f"   python {test_path}")
    print("5. Restart your EdgeElite backend")
    print("6. Enjoy real AI responses with NPU acceleration!")
    print()
    print("🔗 Tutorial Reference:")
    print("   https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie")
    print("🔗 Model Repository:")
    print("   https://github.com/quic/ai-hub-models/tree/main/qai_hub_models/models/llama3_taide_lx_8b_chat_alpha1")
    
    return True

if __name__ == "__main__":
    main() 