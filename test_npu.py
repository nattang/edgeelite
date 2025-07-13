#!/usr/bin/env python3
"""
Test script to verify NPU usage for EdgeElite AI Assistant
"""

from backend.llm import LLMService
import time

def test_npu_usage():
    print("🤖 Testing EdgeElite AI Assistant with NPU...")
    print("=" * 50)
    
    # Initialize LLM service
    llm = LLMService()
    
    # Load model
    print("📥 Loading model...")
    llm.load_model()
    
    print(f"✅ Model loaded: {llm.model_loaded}")
    print(f"🚀 QNN NPU enabled: {llm.use_qnn}")
    print(f"🤖 Text generator ready: {llm.text_generator is not None}")
    
    # Test real AI generation
    print("\n🧪 Testing real AI generation...")
    start_time = time.time()
    
    response = llm.generate_response(
        "Hello! Are you using the Qualcomm Snapdragon X-Elite NPU for inference?",
        []
    )
    
    end_time = time.time()
    generation_time = end_time - start_time
    
    print(f"⏱️ Generation time: {generation_time:.2f} seconds")
    print(f"🤖 AI Response: {response}")
    
    # Test with context
    print("\n🧪 Testing with context...")
    context = [
        {
            "source": "ocr",
            "text": "User is working on EdgeElite AI Assistant project",
            "metadata": {"timestamp": "2025-07-13T00:00:00Z"}
        },
        {
            "source": "asr", 
            "text": "User asked about NPU usage and real AI interface",
            "metadata": {"timestamp": "2025-07-13T00:01:00Z"}
        }
    ]
    
    response_with_context = llm.generate_response(
        "Summarize what I've been working on",
        context
    )
    
    print(f"🤖 Contextual AI Response: {response_with_context}")
    
    print("\n✅ NPU Test Complete!")

if __name__ == "__main__":
    test_npu_usage() 