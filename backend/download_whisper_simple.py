#!/usr/bin/env python3
"""
Simple Whisper v3 Turbo Downloader for EdgeElite ASR

This script downloads Whisper v3 Turbo and creates the necessary files
for ASR functionality without complex ONNX conversion.
"""

import os
import json
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

def download_whisper_v3():
    """Download Whisper v3 Turbo model and create necessary files."""
    
    print("🎤 Downloading Whisper v3 Turbo for EdgeElite ASR")
    print("=" * 60)
    
    # Create models directory
    models_dir = Path(__file__).parent / "models"
    asr_dir = models_dir / "asr"
    asr_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if model already exists
    model_path = asr_dir / "whisper-large-v3"
    if model_path.exists():
        print(f"✅ Whisper v3 Turbo already exists at: {model_path}")
        return True
    
    try:
        print("📥 Downloading Whisper v3 Turbo...")
        print("Model: openai/whisper-large-v3")
        print("Size: ~3GB (latest and most accurate)")
        print()
        
        model_name = "openai/whisper-large-v3"
        
        print("1. Downloading Whisper processor...")
        processor = WhisperProcessor.from_pretrained(model_name)
        
        print("2. Downloading Whisper model...")
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        print("3. Saving model and processor...")
        processor.save_pretrained(model_path)
        model.save_pretrained(model_path)
        
        print("4. Creating vocabulary files...")
        # Create vocab.json from processor
        vocab = processor.tokenizer.get_vocab()
        with open(asr_dir / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        
        # Create added_tokens.json
        added_tokens = {}
        for token_id, token in processor.tokenizer.added_tokens_decoder.items():
            if hasattr(token, 'content'):
                added_tokens[token_id] = token.content
            else:
                added_tokens[token_id] = str(token)
        with open(asr_dir / "added_tokens.json", "w", encoding="utf-8") as f:
            json.dump(added_tokens, f, ensure_ascii=False, indent=2)
        
        # Create special_tokens_map.json
        special_tokens = {
            "additional_special_tokens": [
                "<|startoftranscript|>",
                "<|en|>",
                "<|transcribe|>",
                "<|notimestamps|>",
                "<|endoftext|>",
                "<|nospeech|>"
            ],
            "bos_token": "<|startoftranscript|>",
            "eos_token": "<|endoftext|>",
            "unk_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>"
        }
        with open(asr_dir / "special_tokens_map.json", "w", encoding="utf-8") as f:
            json.dump(special_tokens, f, ensure_ascii=False, indent=2)
        
        print()
        print("🎉 Whisper v3 Turbo successfully downloaded!")
        print(f"📍 Location: {model_path}")
        print("⚡ Ready for ASR inference")
        print()
        print("Model files created:")
        print(f"  - whisper-large-v3/ (full model)")
        print(f"  - vocab.json")
        print(f"  - added_tokens.json")
        print(f"  - special_tokens_map.json")
        print()
        print("Next steps:")
        print("1. Update ASR code to use Transformers instead of ONNX")
        print("2. Restart your backend server")
        print("3. Test ASR functionality in your app")
        
        return True
        
    except ImportError:
        print("❌ Transformers not installed. Please install dependencies first:")
        print("   pip install transformers torch")
        return False
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def main():
    """Main function."""
    print("EdgeElite ASR - Whisper v3 Turbo Downloader")
    print("For Qualcomm Snapdragon X-Elite NPU")
    print()
    
    success = download_whisper_v3()
    
    if success:
        print("\n✅ Setup complete! Your EdgeElite ASR is ready.")
        print("🚀 Whisper v3 Turbo will provide high-quality transcription!")
    else:
        print("\n⚠️ Setup incomplete. ASR functionality will not work.")
        print("Please check your internet connection and try again.")

if __name__ == "__main__":
    main() 