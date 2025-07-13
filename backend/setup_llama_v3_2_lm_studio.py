#!/usr/bin/env python3
"""
Setup Llama-v3.2-3B-Instruct in LM Studio for EdgeElite AI Assistant
This script helps you configure LM Studio to use Llama-v3.2-3B-Instruct.
"""

import sys
import os

def show_setup_instructions():
    """Show detailed setup instructions for Llama-v3.2-3B-Instruct in LM Studio."""
    print("🚀 Setting up Llama-v3.2-3B-Instruct in LM Studio for EdgeElite")
    print("=" * 80)
    
    print("\n📋 Step-by-Step Setup Instructions:")
    print("=" * 50)
    
    print("\n1️⃣ Download and Install LM Studio")
    print("   • Visit: https://lmstudio.ai/")
    print("   • Download and install LM Studio")
    print("   • Launch LM Studio")
    
    print("\n2️⃣ Download Llama-v3.2-3B-Instruct")
    print("   • In LM Studio, go to 'Search' tab")
    print("   • Search for: 'llama-v3.2-3b-instruct'")
    print("   • Look for: 'llama-v3.2-3b-instruct' by Meta")
    print("   • Click 'Download' (size: ~2GB)")
    print("   • Wait for download to complete")
    
    print("\n3️⃣ Load the Model")
    print("   • Go to 'Local Server' tab")
    print("   • Find 'llama-v3.2-3b-instruct' in your models")
    print("   • Click 'Load' to load the model")
    print("   • Wait for model to load (may take a few minutes)")
    
    print("\n4️⃣ Enable API Server")
    print("   • Go to 'Settings' tab")
    print("   • Click on 'API Server' in the left sidebar")
    print("   • Check 'Start API Server'")
    print("   • Note the URL (default: http://localhost:1234)")
    print("   • Click 'Save'")
    
    print("\n5️⃣ Test Connection")
    print("   • Run: python test_lm_studio.py")
    print("   • Should show: '✅ Successfully connected to LM Studio!'")
    print("   • Should show: 'llama-v3.2-3b-instruct' in available models")
    
    print("\n6️⃣ Start EdgeElite Backend")
    print("   • Run: python main.py")
    print("   • Should show: '✅ Connected to LM Studio - Using remote models!'")
    print("   • Your EdgeElite AI Assistant is now using Llama-v3.2-3B-Instruct!")
    
    print("\n🎯 Model Specifications:")
    print("   • Model: Llama-v3.2-3B-Instruct")
    print("   • Parameters: 3.2B (much better than 117M small model)")
    print("   • Context: 4096 tokens")
    print("   • Quality: High-quality instruction-tuned responses")
    print("   • Speed: Fast inference via LM Studio")
    print("   • Storage: No local storage needed")
    
    print("\n💡 Benefits of Llama-v3.2-3B-Instruct:")
    print("   • Much better response quality than small models")
    print("   • Instruction-tuned for better chat responses")
    print("   • Longer context window (4096 tokens)")
    print("   • Better understanding of complex queries")
    print("   • More natural and helpful responses")
    print("   • Optimized for edge devices")
    
    print("\n🔧 Troubleshooting:")
    print("   • If connection fails, check LM Studio is running")
    print("   • If model not found, make sure it's downloaded and loaded")
    print("   • If API server not working, check settings")
    print("   • Default URL should be: http://localhost:1234")
    
    print("\n🎉 Once setup is complete, your EdgeElite AI Assistant will:")
    print("   • Use Llama-v3.2-3B-Instruct for all responses")
    print("   • Provide high-quality AI insights")
    print("   • Handle complex queries better")
    print("   • Give more natural and helpful responses")
    print("   • Work seamlessly with your Snapdragon X-Elite")

def check_current_status():
    """Check current LM Studio connection status."""
    print("\n🔍 Checking Current Status:")
    print("=" * 30)
    
    try:
        from lm_studio_client import lm_studio_client
        
        status = lm_studio_client.get_status()
        
        if status["connected"]:
            print("✅ LM Studio is connected!")
            print(f"📍 URL: {status['base_url']}")
            print(f"📊 Available models: {status['available_models']}")
            
            if "llama-v3.2-3b-instruct" in str(status["models"]):
                print("✅ Llama-v3.2-3B-Instruct is available!")
                print("🎉 Your EdgeElite AI Assistant is ready to use Llama-v3.2-3B-Instruct!")
            else:
                print("⚠️ Llama-v3.2-3B-Instruct not found in available models")
                print("   Please download and load the model in LM Studio")
        else:
            print("❌ LM Studio is not connected")
            print("   Please follow the setup instructions above")
            
    except ImportError:
        print("❌ LM Studio client not available")
    except Exception as e:
        print(f"❌ Error checking status: {e}")

def main():
    """Main function."""
    print("EdgeElite AI Assistant - Llama-v3.2-3B-Instruct Setup")
    print("For Qualcomm HaQathon - High-Quality Edge AI")
    print()
    
    show_setup_instructions()
    check_current_status()
    
    print("\n" + "=" * 80)
    print("🚀 Ready to set up Llama-v3.2-3B-Instruct for EdgeElite!")
    print("Follow the instructions above to get high-quality AI responses.")

if __name__ == "__main__":
    main() 