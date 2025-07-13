#!/usr/bin/env python3
"""
Debug script to check storage system and database contents.
"""

import os
import sys
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from storage.interface import (
    store_raw_ocr_event,
    store_raw_audio_event,
    get_session_stats,
    get_system_stats,
    clear_all_data
)
from storage.db import StorageDB

def test_storage_system():
    """Test the storage system to see if events are being stored properly."""
    
    print("🔍 Debugging EdgeElite Storage System")
    print("=" * 50)
    
    # Test session ID
    test_session_id = "test_session_debug"
    
    # Clear existing data first
    print("🧹 Clearing existing data...")
    clear_all_data()
    
    # Test OCR event storage
    print("\n📝 Testing OCR event storage...")
    try:
        ocr_event_id = store_raw_ocr_event(
            session_id=test_session_id,
            source="ocr",
            ts=time.time(),
            text="Test OCR content from debug script",
            metadata={"debug": True}
        )
        print(f"✅ OCR event stored with ID: {ocr_event_id}")
    except Exception as e:
        print(f"❌ OCR event storage failed: {e}")
        return False
    
    # Test audio event storage
    print("\n🎤 Testing audio event storage...")
    try:
        audio_data = [{
            "timestamp": time.time(),
            "text": "Test audio transcription from debug script",
            "debug": True
        }]
        audio_event_ids = store_raw_audio_event(
            session_id=test_session_id,
            source="audio",
            audio_data=audio_data
        )
        print(f"✅ Audio events stored with IDs: {audio_event_ids}")
    except Exception as e:
        print(f"❌ Audio event storage failed: {e}")
        return False
    
    # Check session stats
    print("\n📊 Checking session stats...")
    try:
        stats = get_session_stats(test_session_id)
        print(f"Session stats: {stats}")
    except Exception as e:
        print(f"❌ Failed to get session stats: {e}")
        return False
    
    # Check system stats
    print("\n🏢 Checking system stats...")
    try:
        system_stats = get_system_stats()
        print(f"System stats: {system_stats}")
    except Exception as e:
        print(f"❌ Failed to get system stats: {e}")
        return False
    
    # Direct database check
    print("\n🗄️ Direct database check...")
    try:
        db = StorageDB()
        raw_events = db.get_raw_events_by_session(test_session_id)
        print(f"Raw events in database: {len(raw_events)}")
        for i, event in enumerate(raw_events):
            print(f"  Event {i+1}: {event['source']} - {event['text'][:50]}...")
    except Exception as e:
        print(f"❌ Direct database check failed: {e}")
        return False
    
    print("\n✅ Storage system test completed successfully!")
    return True

if __name__ == "__main__":
    test_storage_system() 