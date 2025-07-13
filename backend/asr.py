#!/usr/bin/env python3
"""
ASR Service for EdgeElite Backend - QNN NPU Optimized
Integrates the working QNN ASR service into the backend pipeline
"""

import os
import time
from typing import List, Dict, Any

# Import the fixed QNN ASR service
from asr_qnn_fixed import QNNASRServiceFixed as QNNASRService

# Global ASR service instance
_asr_service = None

def get_asr_service():
    """Get or create the ASR service instance."""
    global _asr_service
    if _asr_service is None:
        try:
            print("🚀 Initializing QNN ASR Service...")
            _asr_service = QNNASRService()
            print("✅ QNN ASR Service initialized")
        except Exception as e:
            print(f"❌ Failed to initialize QNN ASR Service: {e}")
            import traceback
            traceback.print_exc()
            raise
    return _asr_service

def process_audio(filename: str) -> List[Dict[str, Any]]:
    """
    Process audio file and return transcription results.
    
    Args:
        filename: Path to the audio file
        
    Returns:
        List of transcription segments with start, end, and text
    """
    try:
        print(f"🎤 Processing audio file: {filename}")
        
        # Get ASR service
        try:
            asr_service = get_asr_service()
        except Exception as service_error:
            print(f"❌ ASR service initialization failed: {service_error}")
            return [{"start": 0, "end": 0, "text": f"ASR Service Error: {str(service_error)}"}]
        
        # Check if file exists
        if not os.path.exists(filename):
            print(f"❌ Audio file not found: {filename}")
            return [{"start": 0, "end": 0, "text": "Audio file not found"}]
        
        # Transcribe audio using QNN NPU
        try:
            start_time = time.time()
            results = asr_service.transcribe_audio(filename)
            total_time = time.time() - start_time
            
            print(f"✅ Transcription completed in {total_time:.2f}s")
            print(f"📝 Found {len(results)} segments")
            
            # Format results for backend
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "start": result["start"],
                    "end": result["end"], 
                    "text": result["text"]
                })
            
            return formatted_results
            
        except Exception as transcribe_error:
            print(f"❌ Transcription failed: {transcribe_error}")
            import traceback
            traceback.print_exc()
            return [{"start": 0, "end": 0, "text": f"Transcription Error: {str(transcribe_error)}"}]
        
    except Exception as e:
        print(f"❌ ASR processing failed: {e}")
        import traceback
        traceback.print_exc()
        return [{"start": 0, "end": 0, "text": f"ASR Error: {str(e)}"}]

def process_latest_audio(recordings_dir: str) -> str:
    """
    Process the latest audio file in the recordings directory.
    
    Args:
        recordings_dir: Path to recordings directory
        
    Returns:
        Transcribed text as string
    """
    try:
        if not os.path.exists(recordings_dir):
            print(f"❌ Recordings directory not found: {recordings_dir}")
            return "Recordings directory not found"
        
        # Find latest WAV file
        wav_files = [f for f in os.listdir(recordings_dir) if f.endswith('.wav')]
        if not wav_files:
            print("❌ No audio files found in recordings directory")
            return "No audio files found"
        
        # Sort by modification time (newest first)
        wav_files.sort(key=lambda x: os.path.getmtime(os.path.join(recordings_dir, x)), reverse=True)
        latest_audio_file = os.path.join(recordings_dir, wav_files[0])
        
        print(f"🎵 Processing latest audio: {latest_audio_file}")
        
        # Process the audio
        results = process_audio(latest_audio_file)
        
        # Combine all text segments
        if results:
            combined_text = " ".join([r["text"] for r in results if r["text"]]).strip()
            print(f"📝 Combined transcription: {combined_text}")
            return combined_text
        else:
            return "No transcription generated"
            
    except Exception as e:
        print(f"❌ Error processing latest audio: {e}")
        return f"Error: {str(e)}"

# Test function for development
def test_asr():
    """Test the ASR service with a sample audio file."""
    print("🧪 Testing ASR Service...")
    
    # Test with existing audio file
    test_file = "../recordings/audio-2025-07-13T07-04-08-677Z.wav"
    
    if os.path.exists(test_file):
        print(f"🎵 Testing with: {test_file}")
        results = process_audio(test_file)
        
        print("\n📋 Test Results:")
        for result in results:
            print(f"  {result['start']}s - {result['end']}s: {result['text']}")
    else:
        print(f"❌ Test file not found: {test_file}")

if __name__ == "__main__":
    test_asr()
