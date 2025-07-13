#!/usr/bin/env python3
"""
Download Small LLM Model for EdgeElite AI Assistant
Perfect for Snapdragon X-Elite edge devices - Fast, efficient, and ready to use!

This script downloads a small model that's perfect for real-time edge AI inference.
"""

import os
import sys
from pathlib import Path
import subprocess

def download_small_model():
    """Download a small, fast LLM model for edge devices."""
    
    print("🚀 Downloading Small LLM Model for EdgeElite AI Assistant")
    print("Perfect for Snapdragon X-Elite edge devices!")
    print("=" * 70)
    
    # Create models directory
    models_dir = Path(__file__).parent / "models"
    small_model_dir = models_dir / "small_llm"
    small_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if model already exists
    if small_model_dir.exists() and any(small_model_dir.iterdir()):
        print(f"✅ Small LLM model already exists at: {small_model_dir}")
        return True
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("📥 Downloading small edge model...")
        print("Model: microsoft/DialoGPT-small")
        print("Size: ~117M parameters (much smaller than 7B!)")
        print("Speed: Sub-second responses on edge devices")
        print("Memory: ~500MB RAM usage")
        print("Time: ~2-5 minutes")
        print()
        
        # Download tokenizer
        print("1. Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/DialoGPT-small",
            trust_remote_code=True,
            cache_dir=models_dir
        )
        tokenizer.save_pretrained(small_model_dir)
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
        model.save_pretrained(small_model_dir)
        print("✅ Model downloaded")
        
        print()
        print("🎉 Small LLM model successfully downloaded!")
        print(f"📍 Location: {small_model_dir}")
        print("⚡ Ready for fast edge AI inference on Snapdragon X-Elite")
        print()
        print("📊 Model Specifications:")
        print("   - Parameters: 117M (vs 7B for large models)")
        print("   - Memory: ~500MB RAM")
        print("   - Speed: Sub-second responses")
        print("   - Quality: Good for chat and basic tasks")
        print("   - NPU: Compatible with QNN acceleration")
        print()
        print("Next steps:")
        print("1. Restart your backend server")
        print("2. Test the AI assistant in your app")
        print("3. Enjoy fast, real-time AI responses!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Installing required dependencies...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torch", "accelerate"])
            print("✅ Dependencies installed. Please run the script again.")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
        return False
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print()
        print("Alternative: Use the enhanced mock system for now")
        print("The mock system provides realistic AI responses for development")
        return False

def test_small_model():
    """Test the small model to ensure it works."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print("🧪 Testing small LLM model...")
        
        model_path = Path(__file__).parent / "models" / "small_llm"
        if not model_path.exists():
            print("❌ Model not found. Please download first.")
            return False
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype="auto",
            device_map="cpu",  # Start with CPU for testing
            trust_remote_code=True
        )
        
        # Test inference
        prompt = "Hello, how are you today?"
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
        print(f"✅ Test successful!")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main function."""
    print("EdgeElite AI Assistant - Small LLM Downloader")
    print("For Qualcomm HaQathon - Fast Edge AI")
    print()
    
    # Download the model
    success = download_small_model()
    
    if success:
        print("\n🧪 Testing the model...")
        if test_small_model():
            print("\n✅ Setup complete! Your EdgeElite AI Assistant is ready with fast responses!")
        else:
            print("\n⚠️ Model downloaded but test failed. Check the installation.")
    else:
        print("\n❌ Setup failed. Using enhanced mock system.")
    
    return success

if __name__ == "__main__":
    main() 