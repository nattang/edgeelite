#!/usr/bin/env python3
"""
EdgeElite Flan-T5 Model Setup Script
====================================

This script downloads and sets up the Flan-T5-small model for local/offline use
in the EdgeElite pipeline. It downloads the tokenizer files and provides
instructions for ONNX model conversion.

Prerequisites:
- Python 3.8+
- Internet connection for downloading
- ~100MB disk space for model files

Usage:
    python setup_flan_t5_models.py

Author: EdgeElite Team
"""

import os
import sys
import json
import requests
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

def print_step(step_num, description):
    """Print a formatted step description."""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {description}")
    print(f"{'='*60}")

def print_success(message):
    """Print a success message."""
    print(f"✅ {message}")

def print_warning(message):
    """Print a warning message."""
    print(f"⚠️  {message}")

def print_error(message):
    """Print an error message."""
    print(f"❌ {message}")

def create_directory(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
    print_success(f"Created directory: {path}")

def download_file(url, filepath, description):
    """Download a file from URL with progress indication."""
    try:
        print(f"📥 Downloading {description}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print_success(f"Downloaded: {filepath}")
        return True
    except Exception as e:
        print_error(f"Failed to download {description}: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 EdgeElite Flan-T5 Model Setup")
    print("This script will download Flan-T5-small model files for local use.")
    
    # Step 1: Define paths and model info
    print_step(1, "Setting up directories and model information")
    
    # Model information
    MODEL_NAME = "google/flan-t5-small"
    MODEL_FILES = [
        "config.json",
        "generation_config.json", 
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json"
    ]
    
    # Directory paths
    script_dir = Path(__file__).parent
    model_dir = script_dir / "flan-t5-small-ONNX"
    onnx_dir = model_dir / "onnx"
    
    print(f"📁 Model directory: {model_dir}")
    print(f"📁 ONNX directory: {onnx_dir}")
    
    # Create directories
    create_directory(model_dir)
    create_directory(onnx_dir)
    
    # Step 2: Download tokenizer and config files
    print_step(2, "Downloading tokenizer and configuration files")
    
    print(f"🔗 Downloading from: {MODEL_NAME}")
    
    try:
        # Download all model files using huggingface_hub
        print("📥 Downloading model files...")
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            allow_patterns=["*.json", "*.txt", "*.model"],
            ignore_patterns=["*.safetensors", "*.bin", "*.onnx", "*.h5"]
        )
        print_success("All model files downloaded successfully!")
        
    except Exception as e:
        print_error(f"Failed to download model files: {e}")
        print_warning("Trying manual download...")
        
        # Manual download fallback
        base_url = f"https://huggingface.co/{MODEL_NAME}/resolve/main"
        success_count = 0
        
        for filename in MODEL_FILES:
            url = f"{base_url}/{filename}"
            filepath = model_dir / filename
            
            if download_file(url, filepath, filename):
                success_count += 1
        
        if success_count < len(MODEL_FILES):
            print_error(f"Only {success_count}/{len(MODEL_FILES)} files downloaded successfully")
            return False
    
    # Step 3: Verify downloaded files
    print_step(3, "Verifying downloaded files")
    
    missing_files = []
    for filename in MODEL_FILES:
        filepath = model_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print_success(f"✓ {filename} ({size:,} bytes)")
        else:
            missing_files.append(filename)
            print_error(f"✗ {filename} - MISSING")
    
    if missing_files:
        print_error(f"Missing files: {missing_files}")
        return False
    
    # Step 4: Create ONNX conversion instructions
    print_step(4, "ONNX Model Conversion Instructions")
    
    print("""
📋 ONNX MODEL CONVERSION REQUIRED

The Flan-T5 model needs to be converted to ONNX format for optimal performance.
You have several options:

OPTION 1: Use Optimum (Recommended)
-----------------------------------
1. Install optimum: pip install optimum[onnxruntime]
2. Run conversion:
   python -m optimum.exporters.onnx --model google/flan-t5-small --task text2text-generation --framework pt2 --output flan-t5-small-ONNX/onnx

OPTION 2: Use Transformers ONNX Export
--------------------------------------
1. Install: pip install transformers[onnx]
2. Run conversion:
   python -c "
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.onnx import export
import torch

model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')

export(tokenizer, model, 'flan-t5-small-ONNX/onnx', opset=12)
   "

OPTION 3: Download Pre-converted ONNX (if available)
---------------------------------------------------
Check if pre-converted ONNX files are available in the project repository
or from the model maintainer.

REQUIRED ONNX FILES:
- encoder_model.onnx (encoder for text processing)
- decoder_model.onnx (decoder for text generation)
- Optional: quantized versions for NPU acceleration
""")
    
    # Step 5: Create a test script
    print_step(5, "Creating model test script")
    
    test_script = model_dir / "test_model.py"
    test_script_content = '''#!/usr/bin/env python3
"""
Test script to verify Flan-T5 model setup
Run this after completing ONNX conversion
"""

import os
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

def test_model_loading():
    """Test if the model can be loaded successfully."""
    try:
        print("🧪 Testing Flan-T5 model loading...")
        
        # Test tokenizer loading
        from transformers import T5Tokenizer
        tokenizer_path = Path(__file__).parent
        tokenizer = T5Tokenizer.from_pretrained(str(tokenizer_path))
        print("✅ Tokenizer loaded successfully")
        
        # Test ONNX model loading (if available)
        onnx_dir = Path(__file__).parent / "onnx"
        encoder_path = onnx_dir / "encoder_model.onnx"
        decoder_path = onnx_dir / "decoder_model.onnx"
        
        if encoder_path.exists() and decoder_path.exists():
            import onnxruntime as ort
            encoder_session = ort.InferenceSession(str(encoder_path))
            decoder_session = ort.InferenceSession(str(decoder_path))
            print("✅ ONNX models loaded successfully")
        else:
            print("⚠️  ONNX models not found - conversion required")
        
        print("🎉 Model setup verification complete!")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

if __name__ == "__main__":
    test_model_loading()
'''
    
    with open(test_script, 'w') as f:
        f.write(test_script_content)
    
    print_success(f"Created test script: {test_script}")
    
    # Step 6: Final instructions
    print_step(6, "Setup Complete - Next Steps")
    
    print(f"""
🎉 FLAN-T5 MODEL SETUP COMPLETE!

📁 Files downloaded to: {model_dir}

📋 NEXT STEPS:
1. Convert the model to ONNX format (see instructions above)
2. Place ONNX files in: {onnx_dir}
3. Test the setup: python {test_script}
4. Run the backend: python main.py

🔧 TROUBLESHOOTING:
- If you see "Local tokenizer loading failed", the files are downloaded correctly
- If you see ONNX errors, complete the ONNX conversion step
- For NPU acceleration, use quantized ONNX models

📚 RESOURCES:
- Flan-T5 Documentation: https://huggingface.co/google/flan-t5-small
- ONNX Conversion: https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_export
- EdgeElite Documentation: Check project README

🚀 Ready to test EdgeElite with local Flan-T5!
""")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 Setup completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Setup failed. Please check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1) 