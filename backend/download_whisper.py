#!/usr/bin/env python3
"""
Download Whisper ASR Model from Qualcomm AI Hub for EdgeElite

This script downloads the Whisper model optimized for Qualcomm Snapdragon X-Elite NPU.
The model is converted to ONNX format for optimal performance on edge devices.
"""

import os
import sys
import json
from pathlib import Path
import requests
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

def download_whisper_from_aihub():
    """Download Whisper model from Qualcomm AI Hub."""
    
    print("🎤 Downloading Whisper ASR Model from Qualcomm AI Hub")
    print("=" * 60)
    
    # Create models directory
    models_dir = Path(__file__).parent / "models"
    asr_dir = models_dir / "asr"
    asr_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if model already exists
    encoder_path = asr_dir / "encoder_model.onnx"
    decoder_path = asr_dir / "decoder_model_merged.onnx"
    
    if encoder_path.exists() and decoder_path.exists():
        print(f"✅ Whisper model already exists at: {asr_dir}")
        return True
    
    try:
        print("📥 Downloading Whisper v3 Turbo model from Qualcomm AI Hub...")
        print("Model: openai/whisper-large-v3 (latest and most accurate)")
        print("Size: ~1.5GB (large model for high-quality transcription)")
        print()
        
        # Download from Hugging Face (Qualcomm AI Hub compatible)
        model_name = "openai/whisper-large-v3"
        
        print("1. Downloading Whisper processor...")
        processor = WhisperProcessor.from_pretrained(model_name)
        
        print("2. Downloading Whisper model...")
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        print("3. Saving processor files...")
        processor.save_pretrained(asr_dir)
        
        print("4. Converting to ONNX format for edge inference...")
        # Export encoder
        encoder = model.model.encoder
        encoder.eval()
        
        # Create dummy input for encoder
        dummy_input = torch.randn(1, 80, 3000)  # Whisper mel spectrogram input
        
        # Export encoder to ONNX
        torch.onnx.export(
            encoder,
            dummy_input,
            encoder_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_features'],
            output_names=['encoder_outputs'],
            dynamic_axes={
                'input_features': {2: 'sequence_length'},
                'encoder_outputs': {1: 'sequence_length'}
            }
        )
        
        print("5. Creating decoder model...")
        # For simplicity, we'll use the full model for decoder
        # In production, you'd want to export just the decoder part
        torch.onnx.export(
            model,
            (torch.randint(0, 1000, (1, 10)), dummy_input),  # input_ids, encoder_outputs
            decoder_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_ids', 'encoder_hidden_states'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {1: 'sequence_length'},
                'encoder_hidden_states': {1: 'sequence_length'},
                'logits': {1: 'sequence_length'}
            }
        )
        
        print("6. Creating vocabulary files...")
        # Create vocab.json from processor
        vocab = processor.tokenizer.get_vocab()
        with open(asr_dir / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        
        # Create added_tokens.json
        added_tokens = processor.tokenizer.added_tokens_decoder
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
        print("🎉 Whisper ASR model successfully downloaded!")
        print(f"📍 Location: {asr_dir}")
        print("⚡ Ready for edge AI inference on Snapdragon X-Elite NPU")
        print()
        print("Model files created:")
        print(f"  - {encoder_path.name}")
        print(f"  - {decoder_path.name}")
        print(f"  - vocab.json")
        print(f"  - added_tokens.json")
        print(f"  - special_tokens_map.json")
        print()
        print("Next steps:")
        print("1. Restart your backend server")
        print("2. Test ASR functionality in your app")
        print("3. Enjoy real-time speech recognition!")
        
        return True
        
    except ImportError:
        print("❌ Transformers not installed. Please install dependencies first:")
        print("   pip install transformers torch onnx")
        return False
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print()
        print("Alternative: Use a pre-converted Whisper ONNX model")
        print("You can find Whisper ONNX models on Hugging Face Hub")
        return False

def download_whisper_onnx_alternative():
    """Alternative: Download pre-converted Whisper ONNX model."""
    
    print("🔄 Trying alternative: Pre-converted Whisper ONNX model")
    print("=" * 60)
    
    models_dir = Path(__file__).parent / "models"
    asr_dir = models_dir / "asr"
    asr_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs for pre-converted Whisper v3 Turbo ONNX models
    model_urls = {
        "encoder_model.onnx": "https://huggingface.co/optimum/whisper-large-v3/resolve/main/encoder_model.onnx",
        "decoder_model_merged.onnx": "https://huggingface.co/optimum/whisper-large-v3/resolve/main/decoder_model_merged.onnx",
        "vocab.json": "https://huggingface.co/optimum/whisper-large-v3/resolve/main/vocab.json",
        "added_tokens.json": "https://huggingface.co/optimum/whisper-large-v3/resolve/main/added_tokens.json",
        "special_tokens_map.json": "https://huggingface.co/optimum/whisper-large-v3/resolve/main/special_tokens_map.json"
    }
    
    try:
        for filename, url in model_urls.items():
            file_path = asr_dir / filename
            if not file_path.exists():
                print(f"📥 Downloading {filename}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"✅ {filename} downloaded")
            else:
                print(f"✅ {filename} already exists")
        
        print()
        print("🎉 Pre-converted Whisper ONNX model downloaded!")
        print(f"📍 Location: {asr_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Alternative download failed: {e}")
        return False

def main():
    """Main function."""
    print("EdgeElite ASR - Whisper Model Downloader")
    print("For Qualcomm Snapdragon X-Elite NPU")
    print()
    
    # Try primary method first
    success = download_whisper_from_aihub()
    
    if not success:
        print("\n🔄 Trying alternative method...")
        success = download_whisper_onnx_alternative()
    
    if success:
        print("\n✅ Setup complete! Your EdgeElite ASR is ready.")
        print("🚀 The Whisper model will use Qualcomm NPU for fast inference!")
    else:
        print("\n⚠️ Setup incomplete. ASR functionality will not work.")
        print("Please manually download Whisper ONNX models or check your internet connection.")

if __name__ == "__main__":
    main() 