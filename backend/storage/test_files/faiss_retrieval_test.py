#!/usr/bin/env python3
"""
FAISS Retrieval Test Script
Tests the quality and relevance of semantic search in the vector store.
"""

import time
import sys
import os
import tempfile
import shutil

# Add the parent storage directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_test_sessions():
    """Create diverse test sessions for comprehensive testing."""
    
    base_time = time.time()
    
    # Session 1: Coding/Development
    coding_session = [
        {'id': 'c1', 'session_id': 'coding_session', 'source': 'ocr', 'ts': base_time + 1, 'text': 'Visual Studio Code - main.py', 'metadata': {}},
        {'id': 'c2', 'session_id': 'coding_session', 'source': 'audio', 'ts': base_time + 2, 'text': 'Let me implement the user authentication function', 'metadata': {}},
        {'id': 'c3', 'session_id': 'coding_session', 'source': 'ocr', 'ts': base_time + 3, 'text': 'def authenticate_user(username, password):', 'metadata': {}},
        {'id': 'c4', 'session_id': 'coding_session', 'source': 'audio', 'ts': base_time + 4, 'text': 'I need to hash the password and check against database', 'metadata': {}},
        {'id': 'c5', 'session_id': 'coding_session', 'source': 'ocr', 'ts': base_time + 5, 'text': 'import hashlib', 'metadata': {}},
        {'id': 'c6', 'session_id': 'coding_session', 'source': 'ocr', 'ts': base_time + 6, 'text': 'return bcrypt.checkpw(password.encode(), stored_hash)', 'metadata': {}},
    ]
    
    # Session 2: Email/Communication
    email_session = [
        {'id': 'e1', 'session_id': 'email_session', 'source': 'ocr', 'ts': base_time + 10, 'text': 'Gmail - Inbox (23 unread)', 'metadata': {}},
        {'id': 'e2', 'session_id': 'email_session', 'source': 'audio', 'ts': base_time + 11, 'text': 'Let me reply to the client about the project deadline', 'metadata': {}},
        {'id': 'e3', 'session_id': 'email_session', 'source': 'ocr', 'ts': base_time + 12, 'text': 'Subject: Project Timeline Update', 'metadata': {}},
        {'id': 'e4', 'session_id': 'email_session', 'source': 'audio', 'ts': base_time + 13, 'text': 'I should inform them about the delay in the authentication module', 'metadata': {}},
        {'id': 'e5', 'session_id': 'email_session', 'source': 'ocr', 'ts': base_time + 14, 'text': 'Dear client, regarding the authentication system development...', 'metadata': {}},
    ]
    
    # Session 3: Research/Documentation
    research_session = [
        {'id': 'r1', 'session_id': 'research_session', 'source': 'ocr', 'ts': base_time + 20, 'text': 'Google Search: best password hashing algorithms 2024', 'metadata': {}},
        {'id': 'r2', 'session_id': 'research_session', 'source': 'audio', 'ts': base_time + 21, 'text': 'I need to research secure password hashing methods', 'metadata': {}},
        {'id': 'r3', 'session_id': 'research_session', 'source': 'ocr', 'ts': base_time + 22, 'text': 'bcrypt vs argon2 vs scrypt comparison', 'metadata': {}},
        {'id': 'r4', 'session_id': 'research_session', 'source': 'audio', 'ts': base_time + 23, 'text': 'Argon2 seems to be the most secure option according to OWASP', 'metadata': {}},
        {'id': 'r5', 'session_id': 'research_session', 'source': 'ocr', 'ts': base_time + 24, 'text': 'OWASP recommends Argon2id for password hashing', 'metadata': {}},
    ]
    
    # Session 4: Meeting/Presentation
    meeting_session = [
        {'id': 'm1', 'session_id': 'meeting_session', 'source': 'ocr', 'ts': base_time + 30, 'text': 'Zoom Meeting - Security Review', 'metadata': {}},
        {'id': 'm2', 'session_id': 'meeting_session', 'source': 'audio', 'ts': base_time + 31, 'text': 'Today we will discuss the security architecture of our application', 'metadata': {}},
        {'id': 'm3', 'session_id': 'meeting_session', 'source': 'ocr', 'ts': base_time + 32, 'text': 'Slide 1: Authentication Security Best Practices', 'metadata': {}},
        {'id': 'm4', 'session_id': 'meeting_session', 'source': 'audio', 'ts': base_time + 33, 'text': 'We need to implement two-factor authentication for better security', 'metadata': {}},
        {'id': 'm5', 'session_id': 'meeting_session', 'source': 'ocr', 'ts': base_time + 34, 'text': 'Multi-factor authentication reduces breach risk by 99.9%', 'metadata': {}},
    ]
    
    # Session 5: Shopping/Personal
    shopping_session = [
        {'id': 's1', 'session_id': 'shopping_session', 'source': 'ocr', 'ts': base_time + 40, 'text': 'Amazon - Search: wireless headphones', 'metadata': {}},
        {'id': 's2', 'session_id': 'shopping_session', 'source': 'audio', 'ts': base_time + 41, 'text': 'I need new headphones for my work calls', 'metadata': {}},
        {'id': 's3', 'session_id': 'shopping_session', 'source': 'ocr', 'ts': base_time + 42, 'text': 'Sony WH-1000XM4 Noise Canceling Headphones', 'metadata': {}},
        {'id': 's4', 'session_id': 'shopping_session', 'source': 'audio', 'ts': base_time + 43, 'text': 'These have great reviews for noise cancellation', 'metadata': {}},
        {'id': 's5', 'session_id': 'shopping_session', 'source': 'ocr', 'ts': base_time + 44, 'text': 'Price: $348.00 - 4.4/5 stars (47,000 reviews)', 'metadata': {}},
    ]
    
    return [coding_session, email_session, research_session, meeting_session, shopping_session]

def test_retrieval_quality():
    """Test the quality and relevance of FAISS retrieval."""
    
    print("=== FAISS Retrieval Quality Test ===\n")
    
    # Create a temporary directory for testing
    test_dir = tempfile.mkdtemp()
    
    try:
        # Import required modules
        from interface import store_raw_ocr_event, process_session, search_similar, clear_all_data
        
        print("1. Setting up test environment...")
        
        # Clear any existing data
        clear_all_data()
        
        print("2. Creating diverse test sessions...")
        
        # Create and store test sessions
        test_sessions = create_test_sessions()
        
        for session_events in test_sessions:
            session_id = session_events[0]['session_id']
            print(f"   Storing session: {session_id}")
            
            # Store events for this session
            for event in session_events:
                store_raw_ocr_event(
                    session_id=event['session_id'],
                    source=event['source'],
                    ts=event['ts'],
                    text=event['text'],
                    metadata=event['metadata']
                )
            
            # Process the session
            node_ids = process_session(session_id)
            print(f"   Processed into {len(node_ids)} chunks")
        
        print("\n3. Testing semantic search queries...")
        
        # Define test queries with expected relevance
        test_queries = [
            {
                'query': 'password authentication security',
                'expected_topics': ['coding', 'research', 'meeting'],
                'description': 'Should find coding, research, and security content'
            },
            {
                'query': 'email client communication',
                'expected_topics': ['email'],
                'description': 'Should primarily find email-related content'
            },
            {
                'query': 'headphones noise canceling reviews',
                'expected_topics': ['shopping'],
                'description': 'Should find shopping/headphones content'
            },
            {
                'query': 'argon2 bcrypt hashing algorithms',
                'expected_topics': ['research', 'coding'],
                'description': 'Should find research and coding content'
            },
            {
                'query': 'zoom meeting presentation slides',
                'expected_topics': ['meeting'],
                'description': 'Should find meeting/presentation content'
            },
            {
                'query': 'project deadline timeline',
                'expected_topics': ['email'],
                'description': 'Should find email about project timeline'
            }
        ]
        
        # Test each query
        for i, test_case in enumerate(test_queries, 1):
            print(f"\n--- Test Query {i}: {test_case['query']} ---")
            print(f"Description: {test_case['description']}")
            
            # Perform search
            results = search_similar(test_case['query'], k=5)
            
            print(f"Found {len(results)} results:")
            
            for j, (summary, full_data) in enumerate(results, 1):
                # Extract session info from summary
                session_info = "unknown"
                if "session" in summary:
                    session_info = summary.split("session ")[1].split(":")[0] if ":" in summary else summary.split("session ")[1].split(" ")[0]
                
                print(f"  {j}. Session: {session_info}")
                print(f"     Summary: {summary[:100]}...")
                print(f"     Content preview: {full_data[:150].replace(chr(10), ' ')}...")
                print()
        
        print("4. Testing edge cases...")
        
        # Test empty query
        try:
            results = search_similar("", k=3)
            print(f"   Empty query: {len(results)} results")
        except Exception as e:
            print(f"   Empty query error (expected): {e}")
        
        # Test very specific query
        results = search_similar("bcrypt.checkpw password encode", k=3)
        print(f"   Very specific query: {len(results)} results")
        
        # Test unrelated query
        results = search_similar("quantum physics space travel", k=3)
        print(f"   Unrelated query: {len(results)} results")
        
        print("\n5. Testing search performance...")
        
        start_time = time.time()
        for _ in range(10):
            search_similar("authentication security", k=5)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        print(f"   Average search time: {avg_time:.3f} seconds")
        
        print("\n6. Testing result consistency...")
        
        # Run same query multiple times
        query = "password hashing security"
        results1 = search_similar(query, k=3)
        results2 = search_similar(query, k=3)
        
        consistent = all(r1[0] == r2[0] for r1, r2 in zip(results1, results2))
        print(f"   Results consistent across runs: {consistent}")
        
        print("\n✅ FAISS retrieval test completed!")
        print("\nTo evaluate quality:")
        print("1. Check if relevant content appears in top results")
        print("2. Verify cross-session search works")
        print("3. Confirm semantic similarity (not just keyword matching)")
        print("4. Test performance is acceptable (< 1 second)")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        try:
            shutil.rmtree(test_dir)
        except:
            pass

if __name__ == "__main__":
    test_retrieval_quality() 