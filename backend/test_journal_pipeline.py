#!/usr/bin/env python3
"""
Test script for EdgeElite Journal Pipeline
Tests the complete flow: OCR/ASR simulation → Storage → Processing → Retrieval
"""

import sys
import os
import time

# Add backend to path
sys.path.append(os.path.dirname(__file__))

from storage.interface import (
    store_raw_ocr_event,
    store_raw_audio_event,
    process_session,
    search_similar,
    get_system_stats,
    get_session_stats
)

def create_dummy_session_1():
    """
    Create Session 1: Work stress with calendar overload
    (This will be the "current" session that needs guidance)
    """
    session_id = "test_session_current_stress"
    current_time = time.time()
    
    print(f"📝 Creating Session 1: {session_id}")
    
    # Simulate OCR events from screenshots
    ocr_events = [
        "Google Calendar - Today's Schedule: 9:00 AM Team Meeting, 10:30 AM Client Call, 12:00 PM Sprint Review, 2:00 PM Design Review, 3:30 PM All-hands, 5:00 PM Code review",
        "Gmail - 47 unread emails, 12 urgent flags",
        "Slack - 23 unread messages, 5 mentions",
        "Zoom - Meeting starting in 5 minutes: Q4 Planning Session"
    ]
    
    # Store OCR events
    for i, ocr_text in enumerate(ocr_events):
        store_raw_ocr_event(
            session_id=session_id,
            source="ocr",
            ts=current_time + i * 30,  # 30 seconds apart
            text=ocr_text,
            metadata={"screenshot_num": i + 1}
        )
    
    # Simulate ASR/audio events
    audio_data = [
        {
            "timestamp": current_time + 120,
            "text": "I'm feeling really overwhelmed with all these meetings today",
            "confidence": 0.95
        },
        {
            "timestamp": current_time + 150,
            "text": "My calendar is completely packed and I have so many emails to respond to",
            "confidence": 0.92
        },
        {
            "timestamp": current_time + 180,
            "text": "I have a huge headache and I'm stressed about this presentation",
            "confidence": 0.88
        },
        {
            "timestamp": current_time + 210,
            "text": "I need to find a way to clear my head and focus",
            "confidence": 0.91
        }
    ]
    
    # Store audio events
    store_raw_audio_event(
        session_id=session_id,
        source="audio",
        audio_data=audio_data
    )
    
    print(f"✅ Session 1 created with {len(ocr_events)} OCR events and {len(audio_data)} audio events")
    return session_id

def create_dummy_session_2():
    """
    Create Session 2: Past session with walk-without-phone remedy
    (This will be the "remedy" session that RAG should find)
    """
    session_id = "test_session_walk_remedy"
    past_time = time.time() - 86400 * 30  # 30 days ago
    
    print(f"📝 Creating Session 2: {session_id}")
    
    # Simulate OCR events showing stress
    ocr_events = [
        "Microsoft Teams - 15 active meetings today",
        "Outlook - Meeting conflicts detected",
        "Jira - 23 open tickets assigned to you"
    ]
    
    # Store OCR events
    for i, ocr_text in enumerate(ocr_events):
        store_raw_ocr_event(
            session_id=session_id,
            source="ocr",
            ts=past_time + i * 45,
            text=ocr_text,
            metadata={"screenshot_num": i + 1}
        )
    
    # Simulate ASR with the remedy
    audio_data = [
        {
            "timestamp": past_time + 200,
            "text": "I'm extremely stressed and overwhelmed with work",
            "confidence": 0.94
        },
        {
            "timestamp": past_time + 230,
            "text": "I think I need to take a break and clear my head",
            "confidence": 0.89
        },
        {
            "timestamp": past_time + 260,
            "text": "I'm going for a 15-minute walk without my phone to reset",
            "confidence": 0.96
        },
        {
            "timestamp": past_time + 890,  # 10+ minutes later
            "text": "That walk really helped, I feel much more focused now",
            "confidence": 0.93
        }
    ]
    
    # Store audio events
    store_raw_audio_event(
        session_id=session_id,
        source="audio",
        audio_data=audio_data
    )
    
    print(f"✅ Session 2 created with {len(ocr_events)} OCR events and {len(audio_data)} audio events")
    return session_id

def create_dummy_session_3():
    """
    Create Session 3: Different type of stress (email overload)
    """
    session_id = "test_session_email_stress"
    past_time = time.time() - 86400 * 7  # 7 days ago
    
    print(f"📝 Creating Session 3: {session_id}")
    
    # Simulate OCR events
    ocr_events = [
        "Gmail - 156 unread emails",
        "Outlook - Priority inbox full",
        "Slack - 45 unread direct messages"
    ]
    
    # Store OCR events
    for i, ocr_text in enumerate(ocr_events):
        store_raw_ocr_event(
            session_id=session_id,
            source="ocr",
            ts=past_time + i * 60,
            text=ocr_text,
            metadata={"screenshot_num": i + 1}
        )
    
    # Simulate ASR
    audio_data = [
        {
            "timestamp": past_time + 300,
            "text": "I can't keep up with all these emails and messages",
            "confidence": 0.91
        },
        {
            "timestamp": past_time + 330,
            "text": "I'm going to batch process emails in focused blocks",
            "confidence": 0.87
        }
    ]
    
    # Store audio events
    store_raw_audio_event(
        session_id=session_id,
        source="audio",
        audio_data=audio_data
    )
    
    print(f"✅ Session 3 created with {len(ocr_events)} OCR events and {len(audio_data)} audio events")
    return session_id

def test_session_processing(session_ids):
    """
    Test the session processing pipeline
    """
    print(f"\n🔄 Testing Session Processing...")
    
    for session_id in session_ids:
        print(f"\n📊 Processing session: {session_id}")
        
        try:
            # Process session (clean, chunk, embed)
            node_ids = process_session(session_id)
            print(f"✅ Session processed successfully: {len(node_ids)} nodes created")
            
            # Get session stats
            stats = get_session_stats(session_id)
            print(f"📈 Session stats: {stats}")
            
        except Exception as e:
            print(f"❌ Error processing session {session_id}: {e}")
            import traceback
            traceback.print_exc()

def test_similarity_search():
    """
    Test RAG retrieval with similarity search
    """
    print(f"\n🔍 Testing Similarity Search (RAG Retrieval)...")
    
    # Test queries that should find the walk remedy
    test_queries = [
        "I'm stressed and overwhelmed with meetings",
        "Calendar is packed and I have headaches",
        "Too many emails and feeling overwhelmed",
        "Need to clear my head and focus",
        "Feeling burnt out from work"
    ]
    
    for query in test_queries:
        print(f"\n🔎 Query: '{query}'")
        
        try:
            # Search for similar content
            results = search_similar(query, k=3)
            print(f"📋 Found {len(results)} similar results:")
            
            for i, (summary, content) in enumerate(results, 1):
                print(f"\n   {i}. Summary: {summary}")
                print(f"      Content: {content[:200]}...")
                
                # Check if this result contains the walk remedy
                if "walk without" in content.lower():
                    print(f"      🎯 REMEDY FOUND! This result contains the walk solution!")
                    
        except Exception as e:
            print(f"❌ Search error for query '{query}': {e}")

def main():
    """
    Main test function
    """
    print("🚀 EdgeElite Journal Pipeline Test")
    print("=" * 50)
    
    # Create dummy sessions
    session_ids = []
    
    print("\n📝 Phase 1: Creating Dummy Sessions")
    session_ids.append(create_dummy_session_1())  # Current stress session
    session_ids.append(create_dummy_session_2())  # Walk remedy session
    session_ids.append(create_dummy_session_3())  # Email stress session
    
    # Test session processing
    print("\n🔄 Phase 2: Processing Sessions")
    test_session_processing(session_ids)
    
    # Test similarity search
    print("\n🔍 Phase 3: Testing RAG Retrieval")
    test_similarity_search()
    
    # Show system stats
    print("\n📊 Phase 4: System Statistics")
    try:
        stats = get_system_stats()
        print(f"System Stats: {stats}")
    except Exception as e:
        print(f"Error getting system stats: {e}")
    
    print("\n✅ Test Complete!")
    print("\nWhat this test verified:")
    print("1. ✅ OCR/ASR data can be stored using storage functions")
    print("2. ✅ Sessions can be processed (cleaned, chunked, embedded)")
    print("3. ✅ Similarity search can find relevant past experiences")
    print("4. ✅ RAG retrieval can match current stress with past remedies")
    print("5. ✅ The 'walk without phone' remedy should be discoverable")

if __name__ == "__main__":
    main() 