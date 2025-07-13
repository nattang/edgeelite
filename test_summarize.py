#!/usr/bin/env python3
"""
Test script to debug the summarize function
"""

from backend.llm import LLMService
import time

def test_summarize():
    print("🧪 Testing Summarize Function...")
    print("=" * 50)
    
    # Initialize LLM service
    llm = LLMService()
    
    # Load model
    print("📥 Loading model...")
    llm.load_model()
    
    # Test context
    test_context = [
        {
            "source": "ocr",
            "text": "Screenshot showing code editor with Python files",
            "metadata": {"timestamp": "2025-07-13T00:00:00Z"}
        },
        {
            "source": "asr", 
            "text": "Audio recording about setting up the EdgeElite project",
            "metadata": {"timestamp": "2025-07-13T00:01:00Z"}
        },
        {
            "source": "ocr",
            "text": "Screenshot of terminal with npm commands",
            "metadata": {"timestamp": "2025-07-13T00:02:00Z"}
        }
    ]
    
    # Test summarize
    print("\n🔍 Testing summarize with context...")
    print(f"Context items: {len(test_context)}")
    
    start_time = time.time()
    
    response = llm.generate_response(
        "Summarize what I have been working on",
        test_context
    )
    
    end_time = time.time()
    
    print(f"\n⏱️ Response time: {end_time - start_time:.2f} seconds")
    print(f"📏 Response length: {len(response)} characters")
    print(f"\n📝 Response:")
    print("-" * 50)
    print(response)
    print("-" * 50)
    
    # Check if it's using real AI or mock
    if "mock" in response.lower():
        print("\n⚠️ Using mock response")
    else:
        print("\n✅ Using real AI response")
    
    # Check if response is too short
    if len(response) < 100:
        print("\n⚠️ Response is very short - this might indicate an issue")
    else:
        print(f"\n✅ Response length looks good ({len(response)} chars)")

if __name__ == "__main__":
    test_summarize() 