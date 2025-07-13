#!/usr/bin/env python3
"""
Test the integrated QNN Whisper ASR service
"""

import os
import sys
from asr import transcribe_audio

def test_asr_integration():
    """Test the integrated ASR service."""
    print("🧪 Testing integrated QNN Whisper ASR...")
    
    # Check if audio file exists
    test_file = "../recordings/audio-2025-07-13T07-04-08-677Z.wav"
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False
    
    try:
        print(f"🎵 Testing with: {test_file}")
        results = transcribe_audio(test_file)
        
        print("\n📋 ASR Test Results:")
        for result in results:
            print(f"  {result['start']}s - {result['end']}s: {result['text']}")
        
        return True
        
    except Exception as e:
        print(f"❌ ASR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_asr_integration()
    
    if success:
        print("✅ ASR integration test completed successfully!")
    else:
        print("❌ ASR integration test failed!") 