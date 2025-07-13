#!/usr/bin/env python3
"""
Download Llama3-TAIDE-LX-8B-Chat-Alpha1 from Qualcomm AI Hub for EdgeElite AI Assistant
Based on Qualcomm AI Hub tutorials: https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie
"""

import os
import sys
from pathlib import Path
import requests
import json

def download_llama3_taide_lx():
    """Download Llama3-TAIDE-LX-8B-Chat-Alpha1 model from Qualcomm AI Hub."""
    
    print("🚀 Downloading Llama3-TAIDE-LX-8B-Chat-Alpha1 from Qualcomm AI Hub")
    print("=" * 80)
    
    # Create models directory
    models_dir = Path(__file__).parent / "models"
    llama_dir = models_dir / "llama3_taide_lx_8b_chat_alpha1"
    llama_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if model already exists
    if llama_dir.exists() and any(llama_dir.iterdir()):
        print(f"✅ Llama3-TAIDE-LX-8B-Chat-Alpha1 model already exists at: {llama_dir}")
        return True
    
    try:
        print("📥 Downloading Llama3-TAIDE-LX-8B-Chat-Alpha1 from Qualcomm AI Hub...")
        print("Model: Llama3-TAIDE-LX-8B-Chat-Alpha1")
        print("Size: ~8GB (optimized for Qualcomm Snapdragon X-Elite)")
        print("Features: Optimized for NPU acceleration, chat-tuned, Alpha1 version")
        print()
        
        # Qualcomm AI Hub model information
        model_url = "https://github.com/quic/ai-hub-models/tree/main/qai_hub_models/models/llama3_taide_lx_8b_chat_alpha1"
        tutorial_url = "https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie"
        
        print("🔗 Model information:")
        print(f"   GitHub: {model_url}")
        print(f"   Tutorial: {tutorial_url}")
        print()
        
        print("📋 Model details:")
        print("   - Base: Llama 3 8B")
        print("   - Tuned: TAIDE LX (Taiwan AI Development Institute)")
        print("   - Format: Chat/Instruct (Alpha1)")
        print("   - Optimization: Qualcomm NPU optimized")
        print("   - Context: 4096 tokens")
        print("   - Performance: Optimized for Genie platform")
        print()
        
        print("⚠️ Manual download required:")
        print("   1. Visit the Qualcomm AI Hub GitHub repository")
        print("   2. Download the Llama3-TAIDE-LX-8B-Chat-Alpha1 model files")
        print("   3. Extract to: backend/models/llama3_taide_lx_8b_chat_alpha1/")
        print()
        
        print("📁 Expected file structure:")
        print("   backend/models/llama3_taide_lx_8b_chat_alpha1/")
        print("   ├── config.json")
        print("   ├── tokenizer.json")
        print("   ├── tokenizer_config.json")
        print("   ├── model.safetensors (or pytorch_model.bin)")
        print("   └── model.onnx (if ONNX version available)")
        print()
        
        # Create a placeholder README file
        readme_path = llama_dir / "README.md"
        readme_content = """# Llama3-TAIDE-LX-8B-Chat-Alpha1 - Qualcomm AI Hub Model

This directory should contain the Llama3-TAIDE-LX-8B-Chat-Alpha1 model files from Qualcomm AI Hub.

## Model Information

- **Base Model**: Llama 3 8B
- **Fine-tuned**: TAIDE LX (Taiwan AI Development Institute)
- **Version**: Alpha1
- **Format**: Chat/Instruct
- **Optimization**: Qualcomm NPU optimized for Snapdragon X-Elite

## Download Instructions

1. Visit: https://github.com/quic/ai-hub-models/tree/main/qai_hub_models/models/llama3_taide_lx_8b_chat_alpha1
2. Download the model files
3. Extract to this directory

## Tutorial Reference

Based on Qualcomm AI Hub tutorial: https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie

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
- Alpha1 version with latest improvements
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print("✅ Created README with download instructions")
        print(f"📍 Location: {readme_path}")
        print()
        
        print("🎯 Next steps:")
        print("   1. Download the model files manually")
        print("   2. Place them in the llama3_taide_lx_8b_chat_alpha1 directory")
        print("   3. Restart your EdgeElite backend")
        print("   4. Test real AI responses!")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

def check_model_files():
    """Check if Llama3-TAIDE-LX-8B-Chat-Alpha1 model files are present."""
    models_dir = Path(__file__).parent / "models"
    llama_dir = models_dir / "llama3_taide_lx_8b_chat_alpha1"
    
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
    print("EdgeElite AI Assistant - Llama3-TAIDE-LX-8B-Chat-Alpha1 Downloader")
    print("For Qualcomm HaQathon - Using Qualcomm AI Hub Models")
    print("Based on: https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie")
    print()
    
    # Check if model files are already present
    if check_model_files():
        print("✅ Llama3-TAIDE-LX-8B-Chat-Alpha1 model files found!")
        print("🚀 Your EdgeElite AI Assistant is ready for real AI responses!")
        return True
    
    # Set up download instructions
    success = download_llama3_taide_lx()
    
    if success:
        print("📋 Download instructions created successfully!")
        print("Please follow the manual download steps to get real AI responses.")
    else:
        print("❌ Failed to create download instructions.")
    
    return success

if __name__ == "__main__":
    main() 