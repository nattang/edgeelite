#!/usr/bin/env python3
"""
Fix EdgeElite Dependencies
Installs all missing packages to resolve errors and warnings.
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip."""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Install all missing dependencies."""
    print("🔧 Fixing EdgeElite Dependencies")
    print("=" * 50)
    
    # List of packages to install
    packages = [
        "faiss-cpu",                    # Vector storage
        "charset-normalizer",           # Character detection
        "langchain-huggingface",        # Updated LangChain embeddings
        "opencv-python",                # OpenCV for image processing
        "easyocr",                      # OCR library
        "qai-hub-models",               # Qualcomm AI Hub models
        "scipy",                        # Scientific computing (for audio filtering)
    ]
    
    success_count = 0
    total_count = len(packages)
    
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print("=" * 50)
    print(f"✅ Installed {success_count}/{total_count} packages successfully")
    
    if success_count == total_count:
        print("🎉 All dependencies installed! Your EdgeElite should now run without errors.")
    else:
        print("⚠️ Some packages failed to install. Check the errors above.")
    
    print("\nNext steps:")
    print("1. Restart your Python environment")
    print("2. Run: python main.py")
    print("3. Test ASR, OCR, and journal functionality")

if __name__ == "__main__":
    main() 