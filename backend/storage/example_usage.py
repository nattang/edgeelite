"""
Example usage of the Storage & Embedding System.

This script demonstrates how to:
1. Store raw events from OCR and Audio pipelines
2. Process a session to create embeddings
3. Search for similar content
"""

import time
from storage.interface import (
    store_raw_event,
    process_session,
    search_similar,
    get_session_stats,
    get_system_stats,
    clear_all_data
)


def example_usage():
    """Demonstrate the storage system functionality."""
    print("=== Storage & Embedding System Demo ===\n")
    
    # Clear any existing data for clean demo
    print("1. Clearing existing data...")
    clear_all_data()
    
    # Example session ID
    session_id = "demo_session_001"
    current_time = time.time()
    
    print(f"2. Storing raw events for session: {session_id}")
    
    # Store some OCR events (simulating screen captures)
    ocr_events = [
        "Welcome to Visual Studio Code",
        "File Edit Selection View Go Run Terminal Help",
        "def calculate_total(items):",
        "    return sum(item.price for item in items)",
        "# TODO: Add tax calculation"
    ]
    
    for i, text in enumerate(ocr_events):
        event_id = store_raw_event(
            session_id=session_id,
            source="ocr",
            ts=current_time + i,
            text=text,
            metadata={"screen_region": "main_editor"}
        )
        print(f"   Stored OCR event: {event_id[:8]}... -> '{text[:30]}...'")
    
    # Store some Audio events (simulating microphone input)
    audio_events = [
        "Let me work on this function",
        "I need to calculate the total price",
        "Don't forget to add tax calculation later",
        "This should return the sum of all item prices"
    ]
    
    for i, text in enumerate(audio_events):
        event_id = store_raw_event(
            session_id=session_id,
            source="audio",
            ts=current_time + i + 0.5,  # Slightly offset from OCR
            text=text,
            metadata={"confidence": 0.95}
        )
        print(f"   Stored Audio event: {event_id[:8]}... -> '{text[:30]}...'")
    
    # Check session stats before processing
    print(f"\n3. Session stats before processing:")
    stats = get_session_stats(session_id)
    print(f"   Total events: {stats['total_raw_events']}")
    print(f"   OCR events: {stats['ocr_events']}")
    print(f"   Audio events: {stats['audio_events']}")
    print(f"   Is processed: {stats['is_processed']}")
    
    # Process the session
    print(f"\n4. Processing session: {session_id}")
    try:
        node_ids = process_session(session_id)
        print(f"   Created {len(node_ids)} nodes: {[nid[:8] + '...' for nid in node_ids]}")
    except Exception as e:
        print(f"   Error processing session: {e}")
        return
    
    # Check session stats after processing
    print(f"\n5. Session stats after processing:")
    stats = get_session_stats(session_id)
    print(f"   Is processed: {stats['is_processed']}")
    
    # Search for similar content
    print(f"\n6. Searching for similar content:")
    
    queries = [
        "code editor programming",
        "calculate price total",
        "tax calculation",
        "function definition"
    ]
    
    for query in queries:
        print(f"\n   Query: '{query}'")
        try:
            results = search_similar(query, k=2)
            if results:
                for i, (summary, full_data) in enumerate(results):
                    print(f"      Result {i+1}:")
                    print(f"        Summary: {summary[:60]}...")
                    print(f"        Full data: {full_data[:80]}...")
            else:
                print("      No results found")
        except Exception as e:
            print(f"      Error searching: {e}")
    
    # Demonstrate advanced search with filter (if we had session metadata)
    print(f"\n   Advanced Search Example:")
    print(f"   Note: Filters work on document metadata stored in FAISS")
    print(f"   Current implementation stores: node_id, summary, full_data")
    
    # Show system stats
    print(f"\n7. System statistics:")
    system_stats = get_system_stats()
    print(f"   Total sessions processed: {system_stats['total_sessions_processed']}")
    print(f"   Total nodes: {system_stats['total_nodes']}")
    print(f"   FAISS index exists: {system_stats['faiss_stats']['index_exists']}")
    print(f"   Database path: {system_stats['database_path']}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    example_usage() 