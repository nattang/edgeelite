#!/usr/bin/env python3
"""
End-to-End Integration Test for EdgeElite Storage System
Tests the complete pipeline: Data Ingestion → Processing → RAG Retrieval
"""

import time
import sys
import os
import json
from typing import List, Dict, Any, Tuple

# Add the backend directory to Python path to make imports work
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, backend_dir)

# Add storage directory to path  
sys.path.insert(0, os.path.join(backend_dir, 'storage'))

def create_realistic_session_data() -> Dict[str, Any]:
    """Create realistic session data simulating actual EdgeElite usage."""
    
    base_time = time.time()
    
    # Session 1: Software Development Work
    dev_session = {
        'session_id': 'dev_work_session',
        'ocr_events': [
            {'ts': base_time + 1, 'text': 'Visual Studio Code - EdgeElite Project', 'metadata': {'window': 'vscode'}},
            {'ts': base_time + 5, 'text': 'src/components/AudioCapture.js', 'metadata': {'file': 'js'}},
            {'ts': base_time + 10, 'text': 'const startRecording = async () => {', 'metadata': {'code': 'function'}},
            {'ts': base_time + 15, 'text': 'navigator.mediaDevices.getUserMedia({audio: true})', 'metadata': {'code': 'api'}},
            {'ts': base_time + 20, 'text': 'ERROR: Cannot read property of undefined', 'metadata': {'error': 'runtime'}},
            {'ts': base_time + 25, 'text': 'console.log("Recording started successfully")', 'metadata': {'code': 'debug'}},
        ],
        'audio_events': [
            {'timestamp': base_time + 2, 'text': 'Let me work on the audio capture component'},
            {'timestamp': base_time + 7, 'text': 'I need to implement the recording functionality'},
            {'timestamp': base_time + 12, 'text': 'This should handle the media stream properly'},
            {'timestamp': base_time + 18, 'text': 'Hmm, getting an error here, let me debug this'},
            {'timestamp': base_time + 23, 'text': 'Great, the recording is working now'},
        ]
    }
    
    # Session 2: Research and Documentation
    research_session = {
        'session_id': 'ai_research_session',
        'ocr_events': [
            {'ts': base_time + 100, 'text': 'Google Scholar - "semantic search embeddings"', 'metadata': {'website': 'scholar'}},
            {'ts': base_time + 105, 'text': 'FAISS: A Library for Efficient Similarity Search', 'metadata': {'paper': 'title'}},
            {'ts': base_time + 110, 'text': 'Abstract: We present FAISS, a library for efficient similarity search...', 'metadata': {'paper': 'abstract'}},
            {'ts': base_time + 115, 'text': 'GitHub - facebookresearch/faiss', 'metadata': {'website': 'github'}},
            {'ts': base_time + 120, 'text': 'pip install faiss-cpu', 'metadata': {'code': 'command'}},
            {'ts': base_time + 125, 'text': 'LangChain Documentation - Vector Stores', 'metadata': {'docs': 'langchain'}},
        ],
        'audio_events': [
            {'timestamp': base_time + 102, 'text': 'I need to research the best vector database for our semantic search'},
            {'timestamp': base_time + 108, 'text': 'FAISS looks like a good option, developed by Facebook AI Research'},
            {'timestamp': base_time + 113, 'text': 'Let me check the GitHub repository for implementation details'},
            {'timestamp': base_time + 118, 'text': 'I should install the CPU version for our development environment'},
            {'timestamp': base_time + 123, 'text': 'LangChain has good integration with FAISS, perfect for our use case'},
        ]
    }
    
    # Session 3: Meeting and Communication
    meeting_session = {
        'session_id': 'team_meeting_session',
        'ocr_events': [
            {'ts': base_time + 200, 'text': 'Microsoft Teams - EdgeElite Weekly Standup', 'metadata': {'app': 'teams'}},
            {'ts': base_time + 205, 'text': 'Slide 1: Project Progress Update', 'metadata': {'presentation': 'slide'}},
            {'ts': base_time + 210, 'text': 'Sprint 3 Goals: Implement Storage System', 'metadata': {'presentation': 'goals'}},
            {'ts': base_time + 215, 'text': 'Chat: John: "Great work on the OCR pipeline!"', 'metadata': {'chat': 'message'}},
            {'ts': base_time + 220, 'text': 'Screen Share: Database Schema Diagram', 'metadata': {'screen': 'share'}},
            {'ts': base_time + 225, 'text': 'Action Items: 1. Complete FAISS integration', 'metadata': {'action': 'items'}},
        ],
        'audio_events': [
            {'timestamp': base_time + 203, 'text': 'Good morning everyone, let me share the progress update'},
            {'timestamp': base_time + 208, 'text': 'We have successfully implemented the OCR pipeline last week'},
            {'timestamp': base_time + 213, 'text': 'This week we are focusing on the storage and embedding system'},
            {'timestamp': base_time + 218, 'text': 'John, thank you for the feedback on the OCR accuracy'},
            {'timestamp': base_time + 223, 'text': 'Let me share the database schema we have designed'},
        ]
    }
    
    # Session 4: Email and Communication
    email_session = {
        'session_id': 'email_communication_session',
        'ocr_events': [
            {'ts': base_time + 300, 'text': 'Gmail - Inbox (47 unread messages)', 'metadata': {'app': 'gmail'}},
            {'ts': base_time + 305, 'text': 'From: client@company.com Subject: EdgeElite Demo Request', 'metadata': {'email': 'header'}},
            {'ts': base_time + 310, 'text': 'We would like to schedule a demo of EdgeElite for our team', 'metadata': {'email': 'body'}},
            {'ts': base_time + 315, 'text': 'Compose: Re: EdgeElite Demo Request', 'metadata': {'email': 'compose'}},
            {'ts': base_time + 320, 'text': 'Hi, I would be happy to arrange a demo session...', 'metadata': {'email': 'reply'}},
            {'ts': base_time + 325, 'text': 'Calendar: Demo scheduled for Friday 2PM', 'metadata': {'calendar': 'event'}},
        ],
        'audio_events': [
            {'timestamp': base_time + 302, 'text': 'Let me check my emails and respond to the client'},
            {'timestamp': base_time + 307, 'text': 'Great, we have a demo request from a potential client'},
            {'timestamp': base_time + 312, 'text': 'I should respond quickly and schedule a meeting'},
            {'timestamp': base_time + 317, 'text': 'Let me craft a professional response'},
            {'timestamp': base_time + 322, 'text': 'Perfect, I have scheduled the demo for Friday afternoon'},
        ]
    }
    
    # Session 5: Testing and Debugging
    debug_session = {
        'session_id': 'debug_testing_session',
        'ocr_events': [
            {'ts': base_time + 400, 'text': 'Terminal - npm test', 'metadata': {'terminal': 'command'}},
            {'ts': base_time + 405, 'text': 'FAIL: AudioCapture.test.js - Recording not starting', 'metadata': {'test': 'failure'}},
            {'ts': base_time + 410, 'text': 'Browser DevTools - Console', 'metadata': {'browser': 'devtools'}},
            {'ts': base_time + 415, 'text': 'TypeError: Cannot read properties of undefined (reading stream)', 'metadata': {'error': 'js'}},
            {'ts': base_time + 420, 'text': 'Stack trace: AudioCapture.js:45:12', 'metadata': {'error': 'stack'}},
            {'ts': base_time + 425, 'text': 'Fixed: Added null check for mediaStream', 'metadata': {'fix': 'code'}},
            {'ts': base_time + 430, 'text': 'PASS: All tests passing (5/5)', 'metadata': {'test': 'success'}},
        ],
        'audio_events': [
            {'timestamp': base_time + 402, 'text': 'Let me run the test suite to check if everything is working'},
            {'timestamp': base_time + 407, 'text': 'Looks like the audio capture test is failing'},
            {'timestamp': base_time + 412, 'text': 'Let me open the browser console to debug this'},
            {'timestamp': base_time + 417, 'text': 'I see the issue, there is a null reference error'},
            {'timestamp': base_time + 422, 'text': 'I need to add a null check for the media stream'},
            {'timestamp': base_time + 427, 'text': 'Great, all tests are passing now after the fix'},
        ]
    }
    
    return {
        'dev_work_session': dev_session,
        'ai_research_session': research_session,
        'team_meeting_session': meeting_session,
        'email_communication_session': email_session,
        'debug_testing_session': debug_session
    }

def test_data_ingestion(session_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Test the data ingestion pipeline (OCR + Audio events)."""
    
    print("=== Testing Data Ingestion ===")
    
    try:
        import storage.interface as interface
        import storage.db as db_module
        print("✅ Successfully imported modules!")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Trying alternative import method...")
        
        # Alternative: change directory and import
        try:
            original_dir = os.getcwd()
            storage_dir = os.path.join(backend_dir, 'storage')
            os.chdir(storage_dir)
            
            # Add storage to path
            sys.path.insert(0, storage_dir)
            
            # Now import
            import interface
            import db as db_module
            print("✅ Successfully imported modules with alternative method!")
        except Exception as e2:
            print(f"❌ Final import attempt failed: {e2}")
            raise
        finally:
            os.chdir(original_dir)
    
    # Clear existing data to ensure fresh start
    print("🧹 Clearing existing data for fresh test...")
    clear_success = interface.clear_all_data()
    if not clear_success:
        raise RuntimeError("Failed to clear existing data")
    
    # Verify system is empty
    stats = interface.get_system_stats()
    if stats['total_nodes'] != 0 or stats['total_sessions_processed'] != 0:
        raise RuntimeError("System not properly cleared before test")
    
    # Track event IDs for validation
    all_event_ids = {}
    
    for session_name, session in session_data.items():
        session_id = session['session_id']
        print(f"\nIngesting session: {session_id}")
        
        # Store OCR events
        ocr_event_ids = []
        for event in session['ocr_events']:
            event_id = interface.store_raw_ocr_event(
                session_id=session_id,
                source='ocr',
                ts=event['ts'],
                text=event['text'],
                metadata=event.get('metadata', {})
            )
            ocr_event_ids.append(event_id)
        
        print(f"  Stored {len(ocr_event_ids)} OCR events")
        
        # Store Audio events (batch processing)
        audio_event_ids = interface.store_raw_audio_event(
            session_id=session_id,
            source='audio',
            audio_data=session['audio_events']
        )
        
        print(f"  Stored {len(audio_event_ids)} Audio events")
        
        all_event_ids[session_id] = ocr_event_ids + audio_event_ids
    
    # Validate data in SQLite
    print("\n--- Validating SQLite Storage ---")
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "storage.db")
    db = db_module.StorageDB(db_path)
    
    for session_id, event_ids in all_event_ids.items():
        raw_events = db.get_raw_events_by_session(session_id)
        print(f"Session {session_id}: {len(raw_events)} events in database")
        
        # Verify event count matches
        assert len(raw_events) == len(event_ids), f"Event count mismatch for {session_id}"
        
        # Verify both OCR and Audio events are present
        ocr_count = len([e for e in raw_events if e['source'] == 'ocr'])
        audio_count = len([e for e in raw_events if e['source'] == 'audio'])
        print(f"  OCR events: {ocr_count}, Audio events: {audio_count}")
    
    print("✅ Data ingestion test passed!")
    return all_event_ids

def test_processing_pipeline(session_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Test the processing pipeline (cleaning + chunking + embedding)."""
    
    print("\n=== Testing Processing Pipeline ===")
    
    try:
        import storage.interface as interface
        print("✅ Successfully imported interface module!")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Trying alternative import method...")
        
        # Alternative: change directory and import
        try:
            original_dir = os.getcwd()
            storage_dir = os.path.join(backend_dir, 'storage')
            os.chdir(storage_dir)
            
            # Add storage to path
            sys.path.insert(0, storage_dir)
            
            import interface
            print("✅ Successfully imported interface module with alternative method!")
        except Exception as e2:
            print(f"❌ Final import attempt failed: {e2}")
            raise
        finally:
            os.chdir(original_dir)
    
    all_node_ids = {}
    
    for session_name, session in session_data.items():
        session_id = session['session_id']
        print(f"\nProcessing session: {session_id}")
        
        # Get session stats before processing
        stats_before = interface.get_session_stats(session_id)
        print(f"  Before processing: {stats_before['total_raw_events']} raw events")
        
        # Process the session
        node_ids = interface.process_session(session_id)
        print(f"  Created {len(node_ids)} chunks/nodes")
        
        # Get session stats after processing
        stats_after = interface.get_session_stats(session_id)
        print(f"  After processing: processed={stats_after['is_processed']}")
        
        all_node_ids[session_id] = node_ids
        
        # Verify processing worked
        assert stats_after['is_processed'], f"Session {session_id} not marked as processed"
        assert len(node_ids) > 0, f"No nodes created for session {session_id}"
    
    # Validate FAISS index
    print("\n--- Validating FAISS Index ---")
    try:
        import storage.interface as interface
        print("✅ Successfully imported interface module!")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        raise
    
    system_stats = interface.get_system_stats()
    print(f"Total nodes in system: {system_stats['total_nodes']}")
    print(f"Sessions processed: {system_stats['total_sessions_processed']}")
    print(f"FAISS stats: {system_stats['faiss_stats']}")
    
    # Verify all sessions are processed
    expected_sessions = len(session_data)
    assert system_stats['total_sessions_processed'] == expected_sessions, \
        f"Expected {expected_sessions} sessions, got {system_stats['total_sessions_processed']}"
    
    print("✅ Processing pipeline test passed!")
    return all_node_ids

def test_rag_retrieval(session_data: Dict[str, Any]) -> Dict[str, Any]:
    """Test the RAG retrieval system with various queries."""
    
    print("\n=== Testing RAG Retrieval ===")
    
    try:
        import storage.interface as interface
        print("✅ Successfully imported interface module!")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Trying alternative import method...")
        
        # Alternative: change directory and import
        try:
            original_dir = os.getcwd()
            storage_dir = os.path.join(backend_dir, 'storage')
            os.chdir(storage_dir)
            
            # Add storage to path
            sys.path.insert(0, storage_dir)
            
            import interface
            print("✅ Successfully imported interface module with alternative method!")
        except Exception as e2:
            print(f"❌ Final import attempt failed: {e2}")
            raise
        finally:
            os.chdir(original_dir)
    
    # Define test queries with expected relevance
    test_queries = [
        {
            'query': 'code development programming javascript',
            'expected_sessions': ['dev_work_session', 'debug_testing_session'],
            'k': 3,
            'description': 'Should find development and debugging content'
        },
        {
            'query': 'FAISS vector database semantic search research',
            'expected_sessions': ['ai_research_session'],
            'k': 2,
            'description': 'Should find AI research content'
        },
        {
            'query': 'team meeting standup presentation slides',
            'expected_sessions': ['team_meeting_session'],
            'k': 3,
            'description': 'Should find meeting content'
        },
        {
            'query': 'email client demo schedule communication',
            'expected_sessions': ['email_communication_session'],
            'k': 2,
            'description': 'Should find email communication content'
        },
        {
            'query': 'testing debugging error npm console',
            'expected_sessions': ['debug_testing_session'],
            'k': 3,
            'description': 'Should find debugging content'
        },
        {
            'query': 'audio recording media stream getUserMedia',
            'expected_sessions': ['dev_work_session'],
            'k': 2,
            'description': 'Should find audio capture development content'
        }
    ]
    
    retrieval_results = {}
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: {test_case['query']} ---")
        print(f"Description: {test_case['description']}")
        print(f"Requesting {test_case['k']} chunks")
        
        # Perform search
        results = interface.search_similar(test_case['query'], k=test_case['k'])
        
        print(f"Retrieved {len(results)} results:")
        
        # Analyze results
        found_sessions = set()
        for j, (summary, full_data) in enumerate(results, 1):
            # Extract session info from summary
            session_info = "unknown"
            if "session" in summary:
                session_part = summary.split("session ")[1]
                session_info = session_part.split(":")[0] if ":" in session_part else session_part.split(" ")[0]
            
            found_sessions.add(session_info)
            
            print(f"  {j}. Session: {session_info}")
            print(f"     Summary: {summary[:80]}...")
            print(f"     Relevance: {full_data[:100].replace(chr(10), ' ')}...")
        
        retrieval_results[test_case['query']] = {
            'requested_k': test_case['k'],
            'returned_count': len(results),
            'found_sessions': list(found_sessions),
            'expected_sessions': test_case['expected_sessions']
        }
        
        # Validate results
        assert len(results) <= test_case['k'], f"Returned more results than requested (k={test_case['k']})"
        assert len(results) > 0, f"No results returned for query: {test_case['query']}"
    
    print("\n--- RAG Retrieval Performance Test ---")
    
    # Test retrieval speed
    start_time = time.time()
    for _ in range(10):
        interface.search_similar("programming development code", k=5)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 10
    print(f"Average retrieval time: {avg_time:.3f} seconds")
    
    # Test different k values
    print("\n--- Testing Different K Values ---")
    for k in [1, 3, 5, 10]:
        results = interface.search_similar("development programming", k=k)
        print(f"k={k}: returned {len(results)} results")
    
    print("✅ RAG retrieval test passed!")
    return retrieval_results

def generate_test_report(ingestion_results: Dict[str, List[str]], 
                        processing_results: Dict[str, List[str]], 
                        retrieval_results: Dict[str, Any]) -> None:
    """Generate a comprehensive test report."""
    
    print("\n" + "="*60)
    print("           END-TO-END TEST REPORT")
    print("="*60)
    
    print(f"\n📊 DATA INGESTION SUMMARY")
    print(f"Sessions processed: {len(ingestion_results)}")
    total_events = sum(len(events) for events in ingestion_results.values())
    print(f"Total events stored: {total_events}")
    
    for session_id, event_ids in ingestion_results.items():
        print(f"  {session_id}: {len(event_ids)} events")
    
    print(f"\n🔄 PROCESSING PIPELINE SUMMARY")
    print(f"Sessions processed: {len(processing_results)}")
    total_nodes = sum(len(nodes) for nodes in processing_results.values())
    print(f"Total nodes/chunks created: {total_nodes}")
    
    for session_id, node_ids in processing_results.items():
        print(f"  {session_id}: {len(node_ids)} chunks")
    
    print(f"\n🔍 RAG RETRIEVAL SUMMARY")
    print(f"Test queries executed: {len(retrieval_results)}")
    
    for query, results in retrieval_results.items():
        print(f"\nQuery: '{query[:50]}...'")
        print(f"  Requested: {results['requested_k']} chunks")
        print(f"  Returned: {results['returned_count']} chunks")
        print(f"  Found sessions: {results['found_sessions']}")
        print(f"  Expected sessions: {results['expected_sessions']}")
    
    print(f"\n✅ END-TO-END TEST COMPLETED SUCCESSFULLY")
    print("All pipeline components are working correctly!")

def main():
    """Run the complete end-to-end test."""
    
    print("🚀 Starting EdgeElite Storage System End-to-End Test")
    print("Testing complete pipeline: Ingestion → Processing → RAG Retrieval")
    print("="*60)
    
    try:
        # Step 1: Create realistic test data
        print("Creating realistic session data...")
        session_data = create_realistic_session_data()
        print(f"Created {len(session_data)} test sessions")
        
        # Step 2: Test data ingestion
        ingestion_results = test_data_ingestion(session_data)
        
        # Step 3: Test processing pipeline
        processing_results = test_processing_pipeline(session_data)
        
        # Step 4: Test RAG retrieval
        retrieval_results = test_rag_retrieval(session_data)
        
        # Step 5: Generate comprehensive report
        generate_test_report(ingestion_results, processing_results, retrieval_results)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 