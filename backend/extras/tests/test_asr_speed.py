#!/usr/bin/env python3
"""
Test ASR speed improvements
"""

import time
import asr

def test_asr_speed():
    """Test ASR processing speed"""
    print("🧪 Testing ASR speed improvements...")
    
    # Find a test audio file
    import os
    recordings_dir = os.path.join(os.path.dirname(__file__), "..", "recordings")
    if os.path.exists(recordings_dir):
        audio_files = [f for f in os.listdir(recordings_dir) if f.endswith('.wav')]
        if audio_files:
            test_file = os.path.join(recordings_dir, audio_files[-1])  # Latest file
            print(f"🎵 Testing with: {test_file}")
            
            start_time = time.time()
            try:
                result = asr.transcribe_audio(test_file)
                end_time = time.time()
                
                print(f"✅ ASR completed in {end_time - start_time:.2f} seconds")
                print(f"📝 Result: {result}")
                
                if end_time - start_time < 10:
                    print("🚀 ASR is fast enough for demo!")
                else:
                    print("⚠️ ASR still too slow, may need further optimization")
                    
            except Exception as e:
                print(f"❌ ASR test failed: {e}")
        else:
            print("❌ No audio files found for testing")
    else:
        print("❌ Recordings directory not found")

if __name__ == "__main__":
    test_asr_speed() 