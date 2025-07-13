#!/usr/bin/env python3
"""
Test Official LM Studio Client Integration for EdgeElite AI Assistant
This script tests the official LM Studio client and server setup.
"""

import sys
import os

def test_lm_studio_installation():
    """Test if LM Studio is properly installed."""
    print("🧪 Testing LM Studio Installation")
    print("=" * 50)
    
    try:
        import lmstudio
        print("✅ LM Studio package installed successfully")
        print(f"   Version: {lmstudio.__version__ if hasattr(lmstudio, '__version__') else 'Unknown'}")
        return True
    except ImportError as e:
        print(f"❌ LM Studio package not installed: {e}")
        print("   Run: pip install lmstudio")
        return False

def test_lm_studio_client():
    """Test LM Studio client initialization."""
    print("\n🧪 Testing LM Studio Client")
    print("=" * 40)
    
    try:
        from lmstudio import Client
        
        # Test client initialization
        client = Client(base_url="http://localhost:8080")
        print("✅ LM Studio client initialized successfully")
        print(f"   Base URL: {client.base_url}")
        
        # Test connection (this will fail if server is not running)
        try:
            models = client.list_models()
            print("✅ Successfully connected to LM Studio server!")
            print(f"   Available models: {models}")
            return True
        except Exception as e:
            print(f"⚠️ Could not connect to LM Studio server: {e}")
            print("   This is expected if the server is not running")
            return False
            
    except Exception as e:
        print(f"❌ LM Studio client test failed: {e}")
        return False

def show_server_setup_instructions():
    """Show instructions for setting up LM Studio server."""
    print("\n📋 LM Studio Server Setup Instructions:")
    print("=" * 60)
    
    print("\n1️⃣ Install LM Studio Desktop App")
    print("   • Download from: https://lmstudio.ai/")
    print("   • Install and launch the desktop app")
    
    print("\n2️⃣ Download Llama-v3.2-3B-Instruct")
    print("   • In LM Studio, go to 'Search' tab")
    print("   • Search for: 'llama-v3.2-3b-instruct'")
    print("   • Download the model (~2GB)")
    
    print("\n3️⃣ Start LM Studio Server")
    print("   • Open terminal/command prompt")
    print("   • Run: lm-studio serve --host 0.0.0.0 --port 8080")
    print("   • This starts the server on http://localhost:8080")
    
    print("\n4️⃣ Alternative: Use LM Studio Desktop API")
    print("   • In LM Studio desktop app")
    print("   • Go to Settings > API Server")
    print("   • Enable 'Start API Server'")
    print("   • Set port to 8080")
    
    print("\n5️⃣ Test Connection")
    print("   • Run this script again")
    print("   • Should show: '✅ Successfully connected to LM Studio server!'")
    
    print("\n🎯 Benefits:")
    print("   • Official LM Studio client integration")
    print("   • Access to Llama-v3.2-3B-Instruct")
    print("   • High-quality AI responses")
    print("   • Easy model management")

def show_edgeelite_integration():
    """Show how EdgeElite integrates with LM Studio."""
    print("\n🔗 EdgeElite + LM Studio Integration:")
    print("=" * 50)
    
    print("\n✅ What's Ready:")
    print("   • Official LM Studio client installed")
    print("   • LLM service updated to use LM Studio")
    print("   • Fallback to local small model")
    print("   • Enhanced mock responses as backup")
    
    print("\n🚀 Priority Order:")
    print("   1. LM Studio (Llama-v3.2-3B-Instruct) - Highest quality")
    print("   2. Local Small Model (117M) - Fast fallback")
    print("   3. Enhanced Mock - Development backup")
    
    print("\n📊 Model Comparison:")
    print("   • Llama-v3.2-3B-Instruct: 3.2B parameters, high quality")
    print("   • Local Small Model: 117M parameters, fast")
    print("   • Enhanced Mock: No parameters, instant responses")

def main():
    """Main function."""
    print("EdgeElite AI Assistant - Official LM Studio Integration Test")
    print("For Qualcomm HaQathon - High-Quality Edge AI")
    print("=" * 80)
    
    # Test installation
    if not test_lm_studio_installation():
        show_server_setup_instructions()
        return
    
    # Test client
    client_works = test_lm_studio_client()
    
    # Show setup instructions
    show_server_setup_instructions()
    
    # Show integration status
    show_edgeelite_integration()
    
    print("\n" + "=" * 80)
    if client_works:
        print("🎉 LM Studio integration is working! Your EdgeElite AI Assistant is ready!")
    else:
        print("🚀 LM Studio client is ready! Start the server to use Llama-v3.2-3B-Instruct!")

if __name__ == "__main__":
    main() 