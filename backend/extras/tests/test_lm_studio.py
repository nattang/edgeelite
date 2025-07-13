#!/usr/bin/env python3
"""
Test LM Studio Connection for EdgeElite AI Assistant
This script tests the connection to LM Studio and shows available models.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def test_lm_studio_connection():
    """Test LM Studio connection and show available models."""
    print("🧪 Testing LM Studio Connection for EdgeElite AI Assistant")
    print("=" * 70)
    
    try:
        from lm_studio_client import lm_studio_client
        
        print("📡 Testing connection to LM Studio...")
        
        # Check connection status
        status = lm_studio_client.get_status()
        
        if status["connected"]:
            print("✅ Successfully connected to LM Studio!")
            print(f"📍 URL: {status['base_url']}")
            print(f"📊 Available models: {status['available_models']}")
            print(f"🎯 Default model: {status['default_model']}")
            
            if status["models"]:
                print("\n📋 Available Models:")
                for i, model in enumerate(status["models"], 1):
                    print(f"   {i}. {model}")
            else:
                print("\n⚠️ No models found. Make sure you have loaded a model in LM Studio.")
            
            # Test a simple completion
            print("\n🧪 Testing model response...")
            try:
                test_prompt = "Hello, how are you today?"
                response = lm_studio_client.generate_completion(test_prompt)
                print(f"✅ Test successful!")
                print(f"Prompt: {test_prompt}")
                print(f"Response: {response}")
                
                print("\n🎉 LM Studio is ready for EdgeElite AI Assistant!")
                print("You can now use any model loaded in LM Studio for real AI responses.")
                
            except Exception as e:
                print(f"❌ Model test failed: {e}")
                print("Make sure you have a model loaded and running in LM Studio.")
                
        else:
            print("❌ Failed to connect to LM Studio")
            print("\n🔧 Troubleshooting:")
            print("1. Make sure LM Studio is installed and running")
            print("2. Enable the API server in LM Studio:")
            print("   - Go to Settings > API Server")
            print("   - Enable 'Start API Server'")
            print("   - Default URL: http://localhost:1234")
            print("3. Load a model in LM Studio")
            print("4. Try running this test again")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure lm_studio_client.py is in the same directory")
    except Exception as e:
        print(f"❌ Test failed: {e}")

def show_lm_studio_instructions():
    """Show instructions for setting up LM Studio."""
    print("\n📋 LM Studio Setup Instructions:")
    print("=" * 50)
    print("1. Download LM Studio from: https://lmstudio.ai/")
    print("2. Install and launch LM Studio")
    print("3. Download a model (e.g., Llama 3, Mistral, etc.)")
    print("4. Enable API Server:")
    print("   - Go to Settings > API Server")
    print("   - Check 'Start API Server'")
    print("   - Note the URL (default: http://localhost:1234)")
    print("5. Load your model in LM Studio")
    print("6. Run this test script again")
    print("\n🎯 Benefits:")
    print("   - Access to many models without downloading locally")
    print("   - Easy model switching")
    print("   - No storage space needed on your device")
    print("   - Real AI responses in EdgeElite!")

if __name__ == "__main__":
    test_lm_studio_connection()
    show_lm_studio_instructions() 