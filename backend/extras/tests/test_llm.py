#!/usr/bin/env python3
"""
Test script for Flan-T5 LLM integration
"""

from llm import LLMService

def test_llm():
    print("🧪 Testing Flan-T5 LLM Integration...")
    
    # Create LLM service
    llm_service = LLMService()
    
    # Load model
    print("📥 Loading Flan-T5 model...")
    llm_service.load_model()
    
    # Test response generation
    print("🤖 Testing response generation...")
    test_prompt = "What is artificial intelligence?"
    
    try:
        response = llm_service.generate_response(test_prompt, [])
        print(f"✅ Response: {response}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_llm()
    if success:
        print("🎉 Flan-T5 LLM test passed!")
    else:
        print("💥 Flan-T5 LLM test failed!") 