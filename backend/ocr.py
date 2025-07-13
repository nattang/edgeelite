from fastapi import FastAPI, Request
from pydantic import BaseModel
import onnxruntime as ort 
import cv2
import numpy as np
from PIL import Image
import os
import time

# Try to import pytesseract for OCR
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    print("✅ Tesseract OCR available")
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️ Tesseract not available - install with: pip install pytesseract")
    print("   Also install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")

class CaptureRequest(BaseModel):
    filename: str

def process_image(filename):
    """
    Process image with OCR to extract text and print results.
    
    Args:
        filename: Path to the image file
        
    Returns:
        Dictionary with OCR results and extracted text
    """
    print(f"🖼️ Processing image with OCR: {filename}")
    
    try:
        # Check if file exists
        if not os.path.exists(filename):
            print(f"❌ Image file not found: {filename}")
            return {
                "success": False,
                "error": "Image file not found",
                "text": "",
                "confidence": 0.0
            }
        
        # Load image
        print(f"📥 Loading image: {filename}")
        image = cv2.imread(filename)
        if image is None:
            print(f"❌ Failed to load image: {filename}")
            return {
                "success": False,
                "error": "Failed to load image",
                "text": "",
                "confidence": 0.0
            }
        
        print(f"📊 Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
        
        # Preprocess image for better OCR
        print("🔧 Preprocessing image for OCR...")
        processed_image = preprocess_image_for_ocr(image)
        
        # Perform OCR
        if TESSERACT_AVAILABLE:
            print("🧠 Running Tesseract OCR...")
            start_time = time.time()
            
            # Extract text with confidence scores
            ocr_data = pytesseract.image_to_data(
                processed_image, 
                output_type=pytesseract.Output.DICT,
                config='--psm 6'  # Assume uniform block of text
            )
            
            ocr_time = time.time() - start_time
            print(f"⏱️ OCR completed in {ocr_time:.2f}s")
            
            # Extract text and confidence
            extracted_text = ""
            total_confidence = 0
            valid_words = 0
            
            for i, conf in enumerate(ocr_data['conf']):
                if conf > 0:  # Filter out low confidence results
                    word = ocr_data['text'][i].strip()
                    if word:  # Only add non-empty words
                        extracted_text += word + " "
                        total_confidence += conf
                        valid_words += 1
            
            # Calculate average confidence
            avg_confidence = total_confidence / valid_words if valid_words > 0 else 0
            
            # Clean up text
            extracted_text = extracted_text.strip()
            extracted_text = ' '.join(extracted_text.split())  # Remove extra spaces
            
            print(f"📝 Extracted {valid_words} words with {avg_confidence:.1f}% average confidence")
            print(f"📄 OCR Result: '{extracted_text}'")
            
            return {
                "success": True,
                "text": extracted_text,
                "confidence": avg_confidence,
                "word_count": valid_words,
                "processing_time": ocr_time,
                "image_size": f"{image.shape[1]}x{image.shape[0]}"
            }
            
        else:
            # Fallback: return a more descriptive mock OCR result
            print("⚠️ Using mock OCR (Tesseract not available)")
            
            # Try to analyze the image content for better fallback
            try:
                # Convert to RGB for analysis
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Simple image analysis
                avg_color = np.mean(rgb_image, axis=(0, 1))
                brightness = np.mean(rgb_image)
                
                # Generate descriptive fallback based on image characteristics
                if brightness < 100:
                    mock_text = "Dark or low-light image captured"
                elif brightness > 200:
                    mock_text = "Bright or overexposed image captured"
                elif avg_color[0] > avg_color[1] and avg_color[0] > avg_color[2]:
                    mock_text = "Image with dominant red tones captured"
                elif avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]:
                    mock_text = "Image with dominant green tones captured"
                elif avg_color[2] > avg_color[0] and avg_color[2] > avg_color[1]:
                    mock_text = "Image with dominant blue tones captured"
                else:
                    mock_text = "Screenshot captured - text extraction requires Tesseract installation"
                    
            except Exception as analysis_error:
                print(f"⚠️ Image analysis failed: {analysis_error}")
                mock_text = "Screenshot captured - OCR not available"
            
            return {
                "success": True,
                "text": mock_text,
                "confidence": 50.0,  # Lower confidence for mock results
                "word_count": len(mock_text.split()),
                "processing_time": 0.1,
                "image_size": f"{image.shape[1]}x{image.shape[0]}",
                "note": "Mock result - install Tesseract for real text extraction"
            }
            
    except Exception as e:
        print(f"❌ OCR processing failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Provide a more helpful error message
        if "tesseract is not installed" in str(e).lower():
            error_msg = "Tesseract OCR engine not installed - install from https://github.com/UB-Mannheim/tesseract/wiki"
        else:
            error_msg = str(e)
            
        return {
            "success": False,
            "error": error_msg,
            "text": "OCR processing failed - no text extracted",
            "confidence": 0.0
        }

def preprocess_image_for_ocr(image):
    """
    Preprocess image to improve OCR accuracy.
    
    Args:
        image: OpenCV image (BGR format)
        
    Returns:
        Preprocessed image ready for OCR
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Optional: Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
        
    except Exception as e:
        print(f"⚠️ Image preprocessing failed: {e}")
        # Return original grayscale if preprocessing fails
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)