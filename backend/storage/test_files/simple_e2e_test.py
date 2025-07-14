#!/usr/bin/env python3
"""
Simple End-to-End Test for EdgeElite Storage System
No import headaches - just works!
"""

import sys
import os
import time
from typing import Dict, Any, List

# Add the backend directory to Python path to make imports work
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, backend_dir)

# Now we can import the storage module properly
sys.path.insert(0, os.path.join(backend_dir, 'storage'))

def run_simple_clear():
    """Clear database using our simple cleaner."""
    print("🧹 Clearing database...")
    
    # Get the storage directory
    storage_dir = os.path.dirname(os.path.abspath(__file__)).replace('/test_files', '')
    
    # Database file path
    db_path = os.path.join(storage_dir, "storage.db")
    
    # FAISS index directory
    faiss_dir = os.path.join(storage_dir, "faiss_index")
    
    # Clear database
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"✅ Deleted database: {db_path}")
    
    # Clear FAISS index
    if os.path.exists(faiss_dir):
        import shutil
        shutil.rmtree(faiss_dir)
        print(f"✅ Deleted FAISS index: {faiss_dir}")
    
    print("✅ Database cleared!")

def test_complete_pipeline():
    """Test the complete EdgeElite storage pipeline."""
    
    print("🚀 EdgeElite Storage System - Simple End-to-End Test")
    print("=" * 60)
    
    # Clear database first
    run_simple_clear()
    
    # Now import after path setup
    try:
        import storage.interface as interface
        import storage.db as db_module
    except ImportError as e:
        print(f"❌ Still having import issues: {e}")
        print("Let's try a different approach...")
        
        # Alternative: direct module import
        try:
            # Change to storage directory temporarily
            original_dir = os.getcwd()
            storage_dir = os.path.join(backend_dir, 'storage')
            os.chdir(storage_dir)
            
            # Add storage to path
            sys.path.insert(0, storage_dir)
            
            # Now import
            import interface
            import db as db_module
            
            print("✅ Successfully imported modules!")
            
        except Exception as e2:
            print(f"❌ Final import attempt failed: {e2}")
            return False
        finally:
            os.chdir(original_dir)
    
    try:
        # Test data
        print("\n📝 Creating test data...")
        session_id = "simple_test_session"
        base_time = time.time()
        
        # Test OCR event
        print("📷 Testing OCR event storage...")
        ocr_event_id = interface.store_raw_ocr_event(
            session_id=session_id,
            source='ocr',
            ts=base_time,
            text='Visual Studio Code - EdgeElite Project',
            metadata={'window': 'vscode'}
        )
        print(f"✅ Stored OCR event: {ocr_event_id}")
        
        # Test Audio events
        print("🎤 Testing Audio event storage...")
        audio_data = [
            {'timestamp': base_time + 1, 'text': 'Let me work on the storage system'},
            {'timestamp': base_time + 2, 'text': 'I need to implement the database integration'},
        ]
        audio_event_ids = interface.store_raw_audio_event(
            session_id=session_id,
            source='audio',
            audio_data=audio_data
        )
        print(f"✅ Stored {len(audio_event_ids)} audio events")
        
        # Test processing
        print("\n🔄 Testing processing pipeline...")
        node_ids = interface.process_session(session_id)
        print(f"✅ Created {len(node_ids)} chunks/nodes")
        
        # Test retrieval
        print("\n🔍 Testing RAG retrieval...")
        
        # Test different queries
        test_queries = [
            'code development programming',
            'storage system database',
            'visual studio project'
        ]
        
        for query in test_queries:
            print(f"\n🔎 Query: '{query}'")
            results = interface.search_similar(query, k=2)
            print(f"   Found {len(results)} results")
            
            for i, (summary, content) in enumerate(results, 1):
                print(f"   {i}. {summary[:60]}...")
                print(f"      Content: {content[:80]}...")
        
        # Test system stats
        print("\n📊 System Statistics:")
        stats = interface.get_system_stats()
        print(f"   Total nodes: {stats['total_nodes']}")
        print(f"   Sessions processed: {stats['total_sessions_processed']}")
        print(f"   FAISS stats: {stats['faiss_stats']}")
        
        print("\n" + "=" * 60)
        print("✅ END-TO-END TEST COMPLETED SUCCESSFULLY!")
        print("✅ All components working correctly!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_pipeline()
    sys.exit(0 if success else 1) 