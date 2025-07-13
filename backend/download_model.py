#!/usr/bin/env python3
"""
Download Small Edge Model for EdgeElite AI Assistant

This script downloads a smaller, faster model perfect for edge devices.
Only 117M parameters - much faster than 7B models!
"""

import os
import sys
from pathlib import Path

def download_edge_model():
    """Download small edge model for fast inference."""
    
    print("🚀 Downloading Small Edge Model for EdgeElite AI Assistant")
    print("=" * 60)
    
    # Create models directory
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "dialogpt-small"
    
    if model_path.exists():
        print(f"✅ Model already exists at: {model_path}")
        return True
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("📥 Downloading small edge model...")
        print("Model: microsoft/DialoGPT-small")
        print("Size: ~117M parameters (much smaller than 7B!)")
        print("Time: ~2-5 minutes")
        print()
        
        # Download tokenizer
        print("1. Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/DialoGPT-small",
            trust_remote_code=True,
            cache_dir=models_dir
        )
        tokenizer.save_pretrained(model_path)
        print("✅ Tokenizer downloaded")
        
        # Download model
        print("2. Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-small",
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            cache_dir=models_dir
        )
        model.save_pretrained(model_path)
        print("✅ Model downloaded")
        
        print()
        print("🎉 Small edge model successfully downloaded!")
        print(f"📍 Location: {model_path}")
        print("⚡ Ready for fast edge AI inference on Snapdragon X-Elite")
        print()
        print("Next steps:")
        print("1. Restart your backend server")
        print("2. Test the AI assistant in your app")
        print("3. Enjoy real AI responses!")
        
        return True
        
    except ImportError:
        print("❌ Transformers not installed. Please install dependencies first:")
        print("   pip install transformers torch")
        return False
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print()
        print("Alternative: Use the enhanced mock system for now")
        print("The mock system provides realistic AI responses for development")
        return False

def main():
    """Main function."""
    print("EdgeElite AI Assistant - Small Edge Model Downloader")
    print("For Qualcomm HaQathon - Fast & Lightweight")
    print()
    
    success = download_edge_model()
    
    if success:
        print("\n✅ Setup complete! Your EdgeElite AI Assistant is ready.")
    else:
        print("\n⚠️ Setup incomplete. Using enhanced mock system.")
        print("Your app will still work with realistic AI responses.")

if __name__ == "__main__":
    main() 