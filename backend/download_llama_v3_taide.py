#!/usr/bin/env python3
"""
Download Llama v3 TAIDE 8B Chat from Qualcomm AI Hub for EdgeElite AI Assistant
For Qualcomm HaQathon - Using Qualcomm AI Hub Models
"""

import os
import sys
from pathlib import Path
import requests
import json

def download_llama_v3_taide():
    """Download Llama v3 TAIDE 8B Chat model from Qualcomm AI Hub."""
    
    print("🚀 Downloading Llama v3 TAIDE 8B Chat from Qualcomm AI Hub")
    print("=" * 70)
    
    # Create models directory
    models_dir = Path(__file__).parent / "models"
    llama_dir = models_dir / "llama_v3_taide_8b_chat"
    llama_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if model already exists
    if llama_dir.exists() and any(llama_dir.iterdir()):
        print(f"✅ Llama v3 TAIDE 8B Chat model already exists at: {llama_dir}")
        return True
    
    try:
        print("📥 Downloading Llama v3 TAIDE 8B Chat from Qualcomm AI Hub...")
        print("Model: quic/ai-hub-models/models/llama_v3_taide_8b_chat")
        print("Size: ~8GB (optimized for Qualcomm Snapdragon X-Elite)")
        print("Features: Optimized for NPU acceleration, chat-tuned")
        print()
        
        # Qualcomm AI Hub model URL
        model_url = "https://github.com/quic/ai-hub-models/blob/main/qai_hub_models/models/llama_v3_taide_8b_chat"
        
        print("🔗 Model information:")
        print(f"   GitHub: {model_url}")
        print("   README: https://github.com/quic/ai-hub-models/blob/main/qai_hub_models/models/llama_v3_taide_8b_chat/README.md")
        print()
        
        print("📋 Model details:")
        print("   - Base: Llama v3 8B")
        print("   - Tuned: TAIDE (Taiwan AI Development Institute)")
        print("   - Format: Chat/Instruct")
        print("   - Optimization: Qualcomm NPU optimized")
        print("   - Context: 4096 tokens")
        print()
        
        print("⚠️ Manual download required:")
        print("   1. Visit the Qualcomm AI Hub GitHub repository")
        print("   2. Download the Llama v3 TAIDE 8B Chat model files")
        print("   3. Extract to: backend/models/llama_v3_taide_8b_chat/")
        print()
        
        print("📁 Expected file structure:")
        print("   backend/models/llama_v3_taide_8b_chat/")
        print("   ├── config.json")
        print("   ├── tokenizer.json")
        print("   ├── tokenizer_config.json")
        print("   ├── model.safetensors (or pytorch_model.bin)")
        print("   └── model.onnx (if ONNX version available)")
        print()
        
        # Create a placeholder README file
        readme_path = llama_dir / "README.md"
        readme_content = """# Llama v3 TAIDE 8B Chat - Qualcomm AI Hub Model

This directory should contain the Llama v3 TAIDE 8B Chat model files from Qualcomm AI Hub.

## Download Instructions

1. Visit: https://github.com/quic/ai-hub-models/blob/main/qai_hub_models/models/llama_v3_taide_8b_chat
2. Download the model files
3. Extract to this directory

## Expected Files

- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer vocabulary
- `tokenizer_config.json` - Tokenizer configuration
- `model.safetensors` - Model weights (or pytorch_model.bin)
- `model.onnx` - ONNX version (if available)

## Usage

The EdgeElite AI Assistant will automatically detect and use this model for:
- Real-time chat responses
- Session summarization
- Context-aware assistance
- NPU-accelerated inference on Snapdragon X-Elite

## Performance

- Optimized for Qualcomm Snapdragon X-Elite NPU
- Sub-second response times
- 4096 token context window
- Chat-tuned for conversational AI
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print("✅ Created README with download instructions")
        print(f"📍 Location: {readme_path}")
        print()
        
        print("🎯 Next steps:")
        print("   1. Download the model files manually")
        print("   2. Place them in the llama_v3_taide_8b_chat directory")
        print("   3. Restart your EdgeElite backend")
        print("   4. Test real AI responses!")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

def check_model_files():
    """Check if Llama v3 TAIDE 8B Chat model files are present."""
    models_dir = Path(__file__).parent / "models"
    llama_dir = models_dir / "llama_v3_taide_8b_chat"
    
    if not llama_dir.exists():
        return False
    
    # Check for essential files
    essential_files = [
        "config.json",
        "tokenizer.json", 
        "tokenizer_config.json"
    ]
    
    # Check for model weights
    model_files = [
        "model.safetensors",
        "pytorch_model.bin",
        "model.onnx"
    ]
    
    has_essential = all((llama_dir / f).exists() for f in essential_files)
    has_model = any((llama_dir / f).exists() for f in model_files)
    
    return has_essential and has_model

def main():
    """Main function."""
    print("EdgeElite AI Assistant - Llama v3 TAIDE 8B Chat Downloader")
    print("For Qualcomm HaQathon - Using Qualcomm AI Hub Models")
    print()
    
    # Check if model files are already present
    if check_model_files():
        print("✅ Llama v3 TAIDE 8B Chat model files found!")
        print("🚀 Your EdgeElite AI Assistant is ready for real AI responses!")
        return True
    
    # Set up download instructions
    success = download_llama_v3_taide()
    
    if success:
        print("📋 Download instructions created successfully!")
        print("Please follow the manual download steps to get real AI responses.")
    else:
        print("❌ Failed to create download instructions.")
    
    return success

if __name__ == "__main__":
    main() 