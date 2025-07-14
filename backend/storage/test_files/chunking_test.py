#!/usr/bin/env python3
"""
Test script for the new semantic chunking functionality.
Demonstrates how the chunking system works with sample data.
"""

import time
import sys
import os

# Add the parent storage directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_chunking():
    """Test the semantic chunking functionality."""
    
    print("=== Semantic Chunking Test ===\n")
    
    # Sample session data (simulating OCR/Audio events)
    sample_events = [
        {
            'id': 'evt1',
            'session_id': 'test_session',
            'source': 'ocr',
            'ts': time.time(),
            'text': 'Welcome to Visual Studio Code',
            'metadata': {}
        },
        {
            'id': 'evt2', 
            'session_id': 'test_session',
            'source': 'audio',
            'ts': time.time() + 1,
            'text': 'Let me open a new file',
            'metadata': {}
        },
        {
            'id': 'evt3',
            'session_id': 'test_session', 
            'source': 'ocr',
            'ts': time.time() + 2,
            'text': 'def calculate_total(items):',
            'metadata': {}
        },
        {
            'id': 'evt4',
            'session_id': 'test_session',
            'source': 'audio', 
            'ts': time.time() + 3,
            'text': 'I need to implement this function',
            'metadata': {}
        },
        {
            'id': 'evt5',
            'session_id': 'test_session',
            'source': 'ocr',
            'ts': time.time() + 4,
            'text': '    return sum(item.price for item in items)',
            'metadata': {}
        },
        {
            'id': 'evt6',
            'session_id': 'test_session',
            'source': 'audio',
            'ts': time.time() + 5,
            'text': 'This should calculate the total price',
            'metadata': {}
        }
    ]
    
    try:
        from cleaner import clean_session_data, chunk_session_data
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        print("1. Testing data cleaning...")
        
        # Clean the session data
        full_data = clean_session_data(sample_events)
        print(f"Cleaned data length: {len(full_data)} characters")
        print(f"Cleaned data:\n{full_data}\n")
        
        print("2. Testing semantic chunking...")
        
        # Initialize embedding model (same as in production)
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Create chunks
        chunks = chunk_session_data(
            full_data=full_data,
            session_id='test_session',
            embedding_model=embedding_model
        )
        
        print(f"Created {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Content: {chunk['content']}")
            print(f"Metadata: {chunk['metadata']}")
        
        print("\n3. Testing fallback chunking...")
        
        # Test fallback chunking (simulate langchain_experimental not available)
        from cleaner import _fallback_chunking
        
        fallback_chunks = _fallback_chunking(full_data, 'test_session', 200)
        print(f"Fallback created {len(fallback_chunks)} chunks:")
        for i, chunk in enumerate(fallback_chunks):
            print(f"\n--- Fallback Chunk {i+1} ---")
            print(f"Content: {chunk['content'][:100]}...")
            print(f"Metadata: {chunk['metadata']}")
        
        print("\n✅ All tests passed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install langchain langchain_experimental sentence-transformers")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chunking() 