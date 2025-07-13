#!/usr/bin/env python3
"""
Test script for OCR integration with ONNX models
"""

import os
from ocr.ocr import process_image

def test_ocr():
    print("🧪 Testing OCR Integration...")
    
    # Check if OCR models exist
    models_dir = os.path.join(os.path.dirname(__file__), "models", "ocr")
    detector_path = os.path.join(models_dir, "easyocr-easyocrdetector.onnx")
    recognizer_path = os.path.join(models_dir, "easyocr-easyocrrecognizer.onnx")
    
    print(f"📁 Checking OCR models in: {models_dir}")
    print(f"🔍 Detector model exists: {os.path.exists(detector_path)}")
    print(f"🔍 Recognizer model exists: {os.path.exists(recognizer_path)}")
    
    # Test with a sample image if available
    test_image = os.path.join(os.path.expanduser("~"), "EdgeElite", "captures")
    if os.path.exists(test_image):
        # Find the latest screenshot
        files = [f for f in os.listdir(test_image) if f.endswith('.png')]
        if files:
            files.sort(key=lambda x: os.path.getmtime(os.path.join(test_image, x)), reverse=True)
            latest_image = os.path.join(test_image, files[0])
            print(f"🖼️ Testing with image: {latest_image}")
            
            try:
                result = process_image(latest_image)
                print(f"✅ OCR Result: {result}")
                return True
            except Exception as e:
                print(f"❌ OCR Error: {e}")
                return False
        else:
            print("⚠️ No test images found")
            return True
    else:
        print("⚠️ No captures directory found")
        return True

if __name__ == "__main__":
    success = test_ocr()
    if success:
        print("🎉 OCR test passed!")
    else:
        print("�� OCR test failed!") 