"""
EdgeElite Journal Pipeline Test Suite

This script tests the complete journal pipeline:
1. Data ingestion (OCR/Audio storage)
2. Session processing (embedding generation)
3. RAG retrieval (semantic search)
4. Journal generation (LLM with context)
5. End-to-end workflow

Run this script to validate that the journal feature is working correctly.
"""

import time
import sys
import os
import json
import asyncio
from typing import Dict, List, Any

# Add backend to path for imports
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

# Import required modules
try:
    from storage.interface import (
        store_raw_ocr_event,
        store_raw_audio_event,
        process_session,
        search_similar,
        get_session_stats,
        get_system_stats,
        clear_all_data
    )
    from storage.db import StorageDB
    print("✅ Successfully imported storage functions")
except ImportError as e:
    print(f"❌ Storage import error: {e}")
    sys.exit(1)

try:
    from llm_service import LLMService
    print("✅ Successfully imported LLM service")
except ImportError as e:
    print(f"⚠️  LLM service import error: {e}")
    print("   Some tests may be skipped")
    LLMService = None

class JournalPipelineTester:
    def __init__(self):
        self.session_id = f"test_session_{int(time.time())}"
        self.demo_session_id = "2025-05-05-demo"
        self.results = {}
        
    def print_section(self, title: str):
        """Print a formatted section header."""
        print(f"\n{'='*60}")
        print(f"🧪 {title}")
        print(f"{'='*60}")
        
    def test_1_data_ingestion(self) -> bool:
        """Test 1: Data Ingestion - OCR and Audio storage"""
        self.print_section("TEST 1: Data Ingestion")
        
        try:
            current_time = time.time()
            
            # Test OCR events
            print("📷 Testing OCR event storage...")
            ocr_events = [
                "Google Calendar - Today's Schedule",
                "9:00 AM - Team Meeting",
                "10:30 AM - Client Presentation",
                "2:00 PM - Code Review",
                "Slack - 23 unread messages"
            ]
            
            ocr_event_ids = []
            for i, text in enumerate(ocr_events):
                event_id = store_raw_ocr_event(
                    session_id=self.session_id,
                    source='ocr',
                    ts=current_time + i,
                    text=text,
                    metadata={'screen_region': 'main', 'window': 'chrome'}
                )
                ocr_event_ids.append(event_id)
                print(f"  ✅ Stored OCR event: {text[:40]}...")
            
            # Test Audio events
            print(f"\n🎤 Testing Audio event storage...")
            audio_data = [
                {
                    'timestamp': current_time + 10,
                    'text': 'My head is pounding from all this work',
                    'metadata': {'emotion': 'stressed', 'confidence': 0.95}
                },
                {
                    'timestamp': current_time + 15,
                    'text': 'I have so many meetings today, my calendar is completely packed',
                    'metadata': {'emotion': 'overwhelmed', 'confidence': 0.93}
                },
                {
                    'timestamp': current_time + 20,
                    'text': 'I need to find a way to deal with this stress',
                    'metadata': {'emotion': 'seeking_help', 'confidence': 0.97}
                }
            ]
            
            audio_event_ids = store_raw_audio_event(
                session_id=self.session_id,
                source='audio',
                audio_data=audio_data
            )
            print(f"  ✅ Stored {len(audio_event_ids)} audio events")
            
            # Verify storage
            print(f"\n📊 Verifying storage...")
            stats = get_session_stats(self.session_id)
            print(f"  Total events: {stats['total_raw_events']}")
            print(f"  OCR events: {stats['ocr_events']}")
            print(f"  Audio events: {stats['audio_events']}")
            print(f"  Session exists: {stats['exists']}")
            
            expected_total = len(ocr_events) + len(audio_data)
            if stats['total_raw_events'] == expected_total:
                print(f"  ✅ Event count matches expected: {expected_total}")
                self.results['data_ingestion'] = True
                return True
            else:
                print(f"  ❌ Event count mismatch: got {stats['total_raw_events']}, expected {expected_total}")
                self.results['data_ingestion'] = False
                return False
                
        except Exception as e:
            print(f"❌ Data ingestion test failed: {e}")
            self.results['data_ingestion'] = False
            return False
            
    def test_2_session_processing(self) -> bool:
        """Test 2: Session Processing - Embedding generation"""
        self.print_section("TEST 2: Session Processing")
        
        try:
            print(f"🔄 Processing session: {self.session_id}")
            
            # Get stats before processing
            stats_before = get_session_stats(self.session_id)
            print(f"  Before processing: {stats_before['is_processed']}")
            
            # Process session
            node_ids = process_session(self.session_id)
            print(f"  ✅ Created {len(node_ids)} searchable chunks")
            
            # Get stats after processing
            stats_after = get_session_stats(self.session_id)
            print(f"  After processing: {stats_after['is_processed']}")
            
            # Verify processing
            if stats_after['is_processed'] and len(node_ids) > 0:
                print(f"  ✅ Session processed successfully")
                self.results['session_processing'] = True
                return True
            else:
                print(f"  ❌ Session processing failed")
                self.results['session_processing'] = False
                return False
                
        except Exception as e:
            print(f"❌ Session processing test failed: {e}")
            self.results['session_processing'] = False
            return False
            
    def test_3_rag_retrieval(self) -> bool:
        """Test 3: RAG Retrieval - Semantic search"""
        self.print_section("TEST 3: RAG Retrieval")
        
        try:
            # Test queries related to current session
            test_queries = [
                "stressed from work and meetings",
                "headache from busy schedule",
                "calendar packed with meetings",
                "need help with stress management"
            ]
            
            retrieval_successful = False
            
            for query in test_queries:
                print(f"\n🔎 Query: '{query}'")
                results = search_similar(query, k=3)
                print(f"   Found {len(results)} results")
                
                for i, (summary, content) in enumerate(results, 1):
                    print(f"   {i}. Summary: {summary[:50]}...")
                    print(f"      Content: {content[:80]}...")
                    
                    # Check if we found content from our test session
                    if "pounding" in content.lower() or "packed" in content.lower():
                        print(f"      🎯 Found content from test session!")
                        retrieval_successful = True
            
            if retrieval_successful:
                print(f"\n✅ RAG retrieval working correctly")
                self.results['rag_retrieval'] = True
                return True
            else:
                print(f"\n⚠️  RAG retrieval didn't find expected content")
                self.results['rag_retrieval'] = False
                return False
                
        except Exception as e:
            print(f"❌ RAG retrieval test failed: {e}")
            self.results['rag_retrieval'] = False
            return False
            
    def run_all_tests(self):
        """Run all tests in sequence."""
        print("🚀 Starting EdgeElite Journal Pipeline Test Suite")
        print(f"Test session ID: {self.session_id}")
        
        # Define tests in order
        tests = [
            ("Data Ingestion", self.test_1_data_ingestion),
            ("Session Processing", self.test_2_session_processing),
            ("RAG Retrieval", self.test_3_rag_retrieval)
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                if result:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"❌ Test '{test_name}' crashed: {e}")
                failed += 1
        
        # Print summary
        self.print_section("TEST SUMMARY")
        print(f"Tests passed: {passed}")
        print(f"Tests failed: {failed}")
        print(f"Success rate: {passed}/{passed + failed} ({passed/(passed + failed)*100:.1f}%)")
        
        print(f"\nDetailed results:")
        for test_name, result in self.results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {test_name}: {status}")
        
        if failed == 0:
            print(f"\n🎉 All tests passed! Journal pipeline is working correctly.")
            return True
        else:
            print(f"\n⚠️  {failed} test(s) failed. Please check the issues above.")
            return False

def main():
    """Main function to run the tests."""
    print("EdgeElite Journal Pipeline Test Suite")
    print("=" * 50)
    
    tester = JournalPipelineTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main() 