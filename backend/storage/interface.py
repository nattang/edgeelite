"""
Public API interface for the storage and embedding system.
Provides the main functions for storing, processing, and searching data.
"""
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from .db import StorageDB
from .faiss_store import FAISSStore
from .cleaner import clean_session_data
from .utils import validate_session_id, validate_source, current_timestamp


# Global instances - initialized on first use
_db_instance: Optional[StorageDB] = None
_faiss_instance: Optional[FAISSStore] = None


def _get_db_instance() -> StorageDB:
    """Get or create the database instance."""
    global _db_instance
    if _db_instance is None:
        # Use a path in the storage directory
        db_path = os.path.join(os.path.dirname(__file__), "storage.db")
        _db_instance = StorageDB(db_path)
    return _db_instance


def _get_faiss_instance() -> FAISSStore:
    """Get or create the FAISS instance."""
    global _faiss_instance
    if _faiss_instance is None:
        # Use a path in the storage directory
        index_dir = os.path.join(os.path.dirname(__file__), "faiss_index")
        _faiss_instance = FAISSStore(index_dir)
        
        # Initialize FAISS with existing nodes from database
        db = _get_db_instance()
        nodes = db.get_all_nodes()
        if nodes:
            _faiss_instance.initialize_from_nodes(nodes)
    
    return _faiss_instance


def store_raw_event(
    session_id: str,
    source: str,
    ts: float,
    text: str,
    metadata: Dict[str, Any] = None
) -> str:
    """
    Store a raw event from OCR or Audio pipeline.
    
    Args:
        session_id: Unique identifier for the session
        source: Source type ('ocr' or 'audio')
        ts: Timestamp of the event
        text: Raw text content
        metadata: Additional metadata (optional)
        
    Returns:
        Unique event ID
        
    Raises:
        ValueError: If invalid parameters are provided
    """
    # Validate inputs
    if not validate_session_id(session_id):
        raise ValueError("Invalid session_id: must be a non-empty string")
    
    if not validate_source(source):
        raise ValueError("Invalid source: must be 'ocr' or 'audio'")
    
    if not isinstance(ts, (int, float)) or ts <= 0:
        raise ValueError("Invalid timestamp: must be a positive number")
    
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Invalid text: must be a non-empty string")
    
    # Store in database
    db = _get_db_instance()
    event_id = db.store_raw_event(session_id, source, ts, text, metadata)
    
    return event_id


def process_session(session_id: str) -> List[str]:
    """
    Process all raw events for a session and create embeddings.
    
    This function:
    1. Retrieves all raw events for the session
    2. Cleans and processes the data using the cleaning agent
    3. Creates embeddings for the processed data
    4. Stores the results in both SQLite and FAISS
    
    Args:
        session_id: Unique identifier for the session
        
    Returns:
        List of node IDs created for the session
        
    Raises:
        ValueError: If invalid session_id or session doesn't exist
        RuntimeError: If processing fails
    """
    # Validate session_id
    if not validate_session_id(session_id):
        raise ValueError("Invalid session_id: must be a non-empty string")
    
    db = _get_db_instance()
    
    # Check if session exists
    if not db.session_exists(session_id):
        raise ValueError(f"Session {session_id} does not exist")
    
    # Check if session already processed
    if db.session_processed(session_id):
        # Return existing node IDs
        nodes = db.get_all_nodes()
        return [node['id'] for node in nodes if node['session_id'] == session_id]
    
    # Get raw events for the session
    raw_events = db.get_raw_events_by_session(session_id)
    
    if not raw_events:
        raise ValueError(f"No raw events found for session {session_id}")
    
    try:
        # Clean and process session data
        summary, full_data = clean_session_data(raw_events)
        
        if not summary or not full_data:
            raise RuntimeError("Cleaning agent returned empty results")
        
        # Create embedding for the summary
        faiss_store = _get_faiss_instance()
        
        # Generate embedding using the same model as FAISS
        embedding = faiss_store.embedding_model.embed_query(summary)
        embedding_bytes = np.array(embedding).tobytes()
        
        # Store in database
        node_id = db.store_session_node(session_id, summary, full_data, embedding_bytes)
        
        # Add to FAISS index
        faiss_store.add_node(node_id, summary, full_data)
        
        return [node_id]
        
    except Exception as e:
        raise RuntimeError(f"Failed to process session {session_id}: {str(e)}")


def search_similar(query: str, k: int = 5, filter: dict = None) -> List[Tuple[str, str]]:
    """
    Search for similar content using semantic similarity.
    
    Args:
        query: Search query
        k: Number of results to return
        filter: Optional metadata filter dictionary
        
    Returns:
        List of tuples (summary, full_data) for the most similar content
        Note: Results are ordered by similarity (most similar first)
        
    Raises:
        ValueError: If invalid parameters are provided
    """
    # Validate inputs
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Invalid query: must be a non-empty string")
    
    if not isinstance(k, int) or k <= 0:
        raise ValueError("Invalid k: must be a positive integer")
    
    if filter is not None and not isinstance(filter, dict):
        raise ValueError("Invalid filter: must be a dictionary or None")
    
    try:
        # Search FAISS for similar nodes
        faiss_store = _get_faiss_instance()
        node_ids = faiss_store.get_node_ids_by_similarity(query, k, filter)
        
        if not node_ids:
            return []
        
        # Get full node data from database
        db = _get_db_instance()
        nodes = db.get_nodes_by_ids(node_ids)
        
        # Format results (maintain order from similarity search)
        results = []
        for node_id in node_ids:
            # Find the corresponding node
            node = next((n for n in nodes if n['id'] == node_id), None)
            if node:
                results.append((node['summary'], node['full_data']))
        
        return results
        
    except Exception as e:
        raise RuntimeError(f"Failed to search similar content: {str(e)}")


def get_session_stats(session_id: str) -> Dict[str, Any]:
    """
    Get statistics for a specific session.
    
    Args:
        session_id: Unique identifier for the session
        
    Returns:
        Dictionary with session statistics
    """
    if not validate_session_id(session_id):
        raise ValueError("Invalid session_id: must be a non-empty string")
    
    db = _get_db_instance()
    
    # Get raw events count
    raw_events = db.get_raw_events_by_session(session_id)
    ocr_count = len([e for e in raw_events if e['source'] == 'ocr'])
    audio_count = len([e for e in raw_events if e['source'] == 'audio'])
    
    return {
        'session_id': session_id,
        'total_raw_events': len(raw_events),
        'ocr_events': ocr_count,
        'audio_events': audio_count,
        'is_processed': db.session_processed(session_id),
        'exists': db.session_exists(session_id)
    }


def get_system_stats() -> Dict[str, Any]:
    """
    Get overall system statistics.
    
    Returns:
        Dictionary with system statistics
    """
    db = _get_db_instance()
    faiss_store = _get_faiss_instance()
    
    # Get database stats
    all_nodes = db.get_all_nodes()
    
    # Get FAISS stats
    faiss_stats = faiss_store.get_stats()
    
    return {
        'total_sessions_processed': len(set(node['session_id'] for node in all_nodes)),
        'total_nodes': len(all_nodes),
        'faiss_stats': faiss_stats,
        'database_path': db.db_path,
        'faiss_index_path': faiss_store.index_dir
    }


def clear_all_data() -> bool:
    """
    Clear all data from the system (for testing/debugging).
    
    Returns:
        True if cleared successfully, False otherwise
    """
    try:
        # Clear FAISS index
        faiss_store = _get_faiss_instance()
        faiss_store.clear_index()
        
        # Clear database (recreate tables)
        db = _get_db_instance()
        db.init_database()
        
        return True
        
    except Exception as e:
        print(f"Failed to clear all data: {e}")
        return False


# Initialize system on module import
def _initialize_system():
    """Initialize the system components."""
    try:
        # Initialize database
        _get_db_instance()
        
        # Initialize FAISS (will load existing index if available)
        _get_faiss_instance()
        
    except Exception as e:
        print(f"Warning: Failed to initialize storage system: {e}")


# Auto-initialize when module is imported
_initialize_system() 