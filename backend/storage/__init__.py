"""
Storage & Embedding System for EdgeElite AI Assistant

This package provides storage and semantic search capabilities for an on-device AI assistant.
It combines SQLite for data persistence with FAISS for vector-based semantic search.

Main Components:
- Raw event storage (OCR/Audio pipeline inputs)
- Session processing with LLM-based cleaning
- Vector embeddings for semantic search
- FAISS-based similarity search

Usage:
    from storage import store_raw_event, process_session, search_similar
    
    # Store raw events
    event_id = store_raw_event("session123", "ocr", timestamp, "some text")
    
    # Process session after collection
    node_ids = process_session("session123")
    
    # Search for similar content
    results = search_similar("what was I working on?", k=3)
    
    # Search with metadata filter
    results = search_similar("my query", k=5, filter={"session_id": "123"})
"""

from .interface import (
    store_raw_event,
    process_session,
    search_similar,
    get_session_stats,
    get_system_stats,
    clear_all_data
)

__version__ = "1.0.0"
__all__ = [
    "store_raw_event",
    "process_session", 
    "search_similar",
    "get_session_stats",
    "get_system_stats",
    "clear_all_data"
] 