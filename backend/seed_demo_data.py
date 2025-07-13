"""
Demo Data Seeder for EdgeElite Journal Feature

This script creates realistic historical session data that demonstrates
the RAG capabilities of the journal feature. It creates a session where
the user mentions a "walk without phone" remedy that can be retrieved
later by semantic search.
"""

import time
import sys
import os

# Add backend to path for imports
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

try:
    from storage.interface import (
        store_raw_ocr_event,
        store_raw_audio_event,
        process_session,
        search_similar,
        get_session_stats,
        clear_all_data
    )
    print("✅ Successfully imported storage functions")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the backend directory")
    sys.exit(1)

def create_may_5th_demo_session():
    """
    Create a demo session from "May 5th" with the walk-without-phone remedy.
    This session will be used for RAG retrieval in the journal feature.
    """
    print("\n🌟 Creating May 5th Demo Session...")
    
    # Demo session details
    session_id = "2025-05-05-demo"
    may_5th_timestamp = 1715000000  # Approximate May 5, 2025 timestamp
    
    # OCR events - simulating what user might see on screen during stress
    ocr_events = [
        {
            'ts': may_5th_timestamp,
            'text': 'Google Calendar - May 5, 2025',
            'metadata': {'window': 'chrome', 'app': 'calendar'}
        },
        {
            'ts': may_5th_timestamp + 5,
            'text': '9:00 AM - Team Meeting (1 hour)',
            'metadata': {'calendar_event': True}
        },
        {
            'ts': may_5th_timestamp + 10,
            'text': '10:30 AM - Client Presentation (2 hours)',
            'metadata': {'calendar_event': True}
        },
        {
            'ts': may_5th_timestamp + 15,
            'text': '1:00 PM - Sprint Review (1 hour)',
            'metadata': {'calendar_event': True}
        },
        {
            'ts': may_5th_timestamp + 20,
            'text': '2:30 PM - Code Review (1 hour)',
            'metadata': {'calendar_event': True}
        },
        {
            'ts': may_5th_timestamp + 25,
            'text': '4:00 PM - Design Meeting (1 hour)',
            'metadata': {'calendar_event': True}
        },
        {
            'ts': may_5th_timestamp + 30,
            'text': 'Slack - 15 unread messages',
            'metadata': {'window': 'slack', 'urgent': True}
        },
        {
            'ts': may_5th_timestamp + 35,
            'text': 'Gmail - 47 unread emails',
            'metadata': {'window': 'gmail', 'urgent': True}
        }
    ]
    
    # Audio events - user expressing stress and the remedy
    audio_events = [
        {
            'timestamp': may_5th_timestamp + 40,
            'text': 'Oh my god, my calendar is completely insane today',
            'metadata': {'emotion': 'stressed', 'confidence': 0.95}
        },
        {
            'timestamp': may_5th_timestamp + 45,
            'text': 'I have back-to-back meetings all day and my inbox is overflowing',
            'metadata': {'emotion': 'overwhelmed', 'confidence': 0.93}
        },
        {
            'timestamp': may_5th_timestamp + 50,
            'text': 'My head is starting to hurt from all this stress',
            'metadata': {'emotion': 'physical_stress', 'confidence': 0.97}
        },
        {
            'timestamp': may_5th_timestamp + 55,
            'text': 'I need to take a break. I\'m going for a 15-minute walk without my phone',
            'metadata': {'remedy': 'walk_without_phone', 'confidence': 0.98}
        },
        {
            'timestamp': may_5th_timestamp + 60,
            'text': 'Just a quick walk to clear my head and get some fresh air',
            'metadata': {'remedy': 'walk_without_phone', 'confidence': 0.96}
        },
        {
            'timestamp': may_5th_timestamp + 1800,  # 30 minutes later
            'text': 'That walk was exactly what I needed. I feel much calmer now',
            'metadata': {'remedy_result': 'success', 'confidence': 0.94}
        },
        {
            'timestamp': may_5th_timestamp + 1805,
            'text': 'I can think clearly again and I\'m ready to tackle the rest of the day',
            'metadata': {'remedy_result': 'success', 'confidence': 0.92}
        }
    ]
    
    # Store OCR events
    print(f"📷 Storing {len(ocr_events)} OCR events...")
    for event in ocr_events:
        event_id = store_raw_ocr_event(
            session_id=session_id,
            source='ocr',
            ts=event['ts'],
            text=event['text'],
            metadata=event.get('metadata', {})
        )
        print(f"  ✅ Stored OCR: {event['text'][:50]}...")
    
    # Store Audio events
    print(f"🎤 Storing {len(audio_events)} Audio events...")
    audio_event_ids = store_raw_audio_event(
        session_id=session_id,
        source='audio',
        audio_data=audio_events
    )
    print(f"  ✅ Stored {len(audio_event_ids)} audio events")
    
    return session_id

def create_additional_demo_sessions():
    """
    Create additional demo sessions for better RAG testing.
    """
    print("\n🎯 Creating Additional Demo Sessions...")
    
    # Demo session 2: Work productivity session
    session_id_2 = "2025-04-20-work-session"
    april_20_timestamp = 1713600000  # April 20, 2025
    
    audio_events_2 = [
        {
            'timestamp': april_20_timestamp,
            'text': 'Working on the EdgeElite project today',
            'metadata': {'activity': 'work', 'confidence': 0.95}
        },
        {
            'timestamp': april_20_timestamp + 600,
            'text': 'The code is compiling successfully now',
            'metadata': {'activity': 'coding', 'confidence': 0.93}
        },
        {
            'timestamp': april_20_timestamp + 1200,
            'text': 'Need to implement the audio capture functionality',
            'metadata': {'activity': 'planning', 'confidence': 0.97}
        }
    ]
    
    store_raw_audio_event(session_id_2, 'audio', audio_events_2)
    
    # Demo session 3: Meeting stress session
    session_id_3 = "2025-04-15-meeting-stress"
    april_15_timestamp = 1713168000  # April 15, 2025
    
    audio_events_3 = [
        {
            'timestamp': april_15_timestamp,
            'text': 'This client meeting is not going well',
            'metadata': {'emotion': 'stressed', 'confidence': 0.94}
        },
        {
            'timestamp': april_15_timestamp + 300,
            'text': 'I need to step outside for some air',
            'metadata': {'remedy': 'fresh_air', 'confidence': 0.96}
        },
        {
            'timestamp': april_15_timestamp + 600,
            'text': 'A few minutes outside helped me refocus',
            'metadata': {'remedy_result': 'success', 'confidence': 0.92}
        }
    ]
    
    store_raw_audio_event(session_id_3, 'audio', audio_events_3)
    
    return [session_id_2, session_id_3]

def seed_demo_data():
    """
    Main function to seed all demo data.
    """
    print("🚀 Starting EdgeElite Journal Demo Data Seeding...")
    
    # Create the main May 5th session (with walk remedy)
    main_session_id = create_may_5th_demo_session()
    
    # Create additional sessions for context
    additional_sessions = create_additional_demo_sessions()
    
    all_sessions = [main_session_id] + additional_sessions
    
    print(f"\n🔄 Processing {len(all_sessions)} demo sessions...")
    
    # Process all sessions to create embeddings
    for session_id in all_sessions:
        print(f"\n📊 Processing session: {session_id}")
        try:
            # Get session stats before processing
            stats_before = get_session_stats(session_id)
            print(f"  Raw events: {stats_before['total_raw_events']} "
                  f"(OCR: {stats_before['ocr_events']}, Audio: {stats_before['audio_events']})")
            
            # Process session
            node_ids = process_session(session_id)
            print(f"  ✅ Created {len(node_ids)} searchable chunks")
            
            # Get session stats after processing
            stats_after = get_session_stats(session_id)
            print(f"  Processed: {stats_after['is_processed']}")
            
        except Exception as e:
            print(f"  ❌ Error processing session {session_id}: {e}")
            continue
    
    print("\n✅ Demo data seeding complete!")
    return all_sessions

def test_rag_retrieval():
    """
    Test the RAG retrieval system with demo data.
    """
    print("\n🔍 Testing RAG Retrieval with Demo Data...")
    
    # Test queries that should find the walk remedy
    test_queries = [
        "I'm stressed and need to take a break",
        "headache from work pressure",
        "overwhelmed with meetings and emails",
        "need to clear my head",
        "walking without phone to reduce stress"
    ]
    
    for query in test_queries:
        print(f"\n🔎 Query: '{query}'")
        try:
            results = search_similar(query, k=3)
            print(f"   Found {len(results)} relevant results:")
            
            for i, (summary, content) in enumerate(results, 1):
                print(f"   {i}. Summary: {summary[:60]}...")
                print(f"      Content: {content[:100]}...")
                
                # Check if this result mentions the walk remedy
                if "walk without" in content.lower() or "15-minute" in content.lower():
                    print(f"      🎯 FOUND THE WALK REMEDY!")
                    
        except Exception as e:
            print(f"   ❌ Search error: {e}")
    
    print("\n✅ RAG retrieval test complete!")

if __name__ == "__main__":
    print("EdgeElite Journal Feature - Demo Data Seeder")
    print("=" * 50)
    
    # Check if user wants to clear existing data
    response = input("🗑️  Clear existing data first? (y/N): ").lower()
    if response == 'y':
        print("Clearing existing data...")
        clear_all_data()
        print("✅ Data cleared!")
    
    # Seed demo data
    demo_sessions = seed_demo_data()
    
    # Test RAG retrieval
    test_rag_retrieval()
    
    print("\n🎉 Demo setup complete!")
    print("\nDemo Sessions Created:")
    for session_id in demo_sessions:
        print(f"  - {session_id}")
    
    print("\n📝 Next Steps:")
    print("1. Start the FastAPI backend: python -m uvicorn main:app --reload")
    print("2. Start the Electron frontend: npm run dev")
    print("3. Navigate to the Journal page")
    print("4. Start a new session and say: 'Huge headache, calendar is insane'")
    print("5. Take a screenshot of a busy calendar")
    print("6. End the session")
    print("7. Check if the journal references the May 5th walk remedy!") 