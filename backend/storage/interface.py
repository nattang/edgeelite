"""
Public API interface for the storage and embedding system.
Provides the main functions for storing, processing, and searching data.
"""
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from .db import StorageDB
from .faiss_store import FAISSStore
from .cleaner import clean_session_data, chunk_session_data
from .utils import validate_session_id, validate_source, current_timestamp


# Global instances - initialized on first use
_db_instance: Optional[StorageDB] = None
_faiss_instance: Optional[FAISSStore] = None

##Initialization functions for the database and FAISS instance
def _get_db_instance() -> StorageDB:
    """Get a singleton instance of the StorageDB with absolute path."""
    global _db_instance
    if '_db_instance' not in globals() or _db_instance is None:
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage.db")
        _db_instance = StorageDB(db_path=db_path)
    return _db_instance


def _get_faiss_instance() -> 'FAISSStore':
    """Get a singleton instance of the FAISSStore with absolute path."""
    global _faiss_instance
    if '_faiss_instance' not in globals() or _faiss_instance is None:
        index_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index")
        from .faiss_store import FAISSStore
        _faiss_instance = FAISSStore(index_dir=index_dir)
        
        # Initialize FAISS with existing nodes from database
        db = _get_db_instance()
        nodes = db.get_all_nodes()
        if nodes:
            _faiss_instance.initialize_from_nodes(nodes)
    
    return _faiss_instance

##THIS IS THE FUNCTION THAT IS INVOKED FROM OCR PIPELINE
def store_raw_ocr_event(
    session_id: str,
    source: str,
    ts: float,
    text: str,
    metadata: Dict[str, Any] = None
) -> str:
    """
    Store a raw OCR event.
    
    Args:
        session_id: Unique identifier for the session
        source: Source type (should be 'ocr')
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


##THIS IS THE FUNCTION THAT IS INVOKED FROM AUDIO PIPELINE
def store_raw_audio_event(
    session_id: str,
    source: str,
    audio_data: List[Dict[str, Any]]
) -> List[str]:
    """
    Store multiple raw audio events from a batch.
    
    Args:
        session_id: Unique identifier for the session
        source: Source type (should be 'audio')
        audio_data: List of dictionaries with "timestamp" and "text" keys
                   Example: [
                       {"timestamp": 1234567.8, "text": "Hello world"},
                       {"timestamp": 1234568.2, "text": "How are you"}
                   ]
        
    Returns:
        List of unique event IDs
        
    Raises:
        ValueError: If invalid parameters are provided
    """
    # Validate inputs
    if not validate_session_id(session_id):
        raise ValueError("Invalid session_id: must be a non-empty string")
    
    if not validate_source(source):
        raise ValueError("Invalid source: must be 'ocr' or 'audio'")
    
    if not isinstance(audio_data, list) or not audio_data:
        raise ValueError("Invalid audio_data: must be a non-empty list of dictionaries")
    
    # Validate each audio data entry
    for i, entry in enumerate(audio_data):
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid audio_data[{i}]: must be a dictionary")
        
        if "timestamp" not in entry or "text" not in entry:
            raise ValueError(f"Invalid audio_data[{i}]: must contain 'timestamp' and 'text' keys")
        
        timestamp = entry["timestamp"]
        text = entry["text"]
        
        if not isinstance(timestamp, (int, float)) or timestamp <= 0:
            raise ValueError(f"Invalid audio_data[{i}].timestamp: must be a positive number")
        
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Invalid audio_data[{i}].text: must be a non-empty string")
    
    # Store each event in database
    db = _get_db_instance()
    event_ids = []
    
    for entry in audio_data:
        timestamp = entry["timestamp"]
        text = entry["text"]
        metadata = {k: v for k, v in entry.items() if k not in ["timestamp", "text"]}  # Include any extra fields as metadata
        
        event_id = db.store_raw_event(session_id, source, timestamp, text, metadata if metadata else None)
        event_ids.append(event_id)
    
    return event_ids


# Backward compatibility - keep the old function name for existing code
def store_raw_event(
    session_id: str,
    source: str,
    ts: float,
    text: str,
    metadata: Dict[str, Any] = None
) -> str:
    """
    DEPRECATED: Use store_raw_ocr_event or store_raw_audio_event instead.
    
    Store a raw event from OCR or Audio pipeline.
    """
    return store_raw_ocr_event(session_id, source, ts, text, metadata)


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
    raw_events = db.get_raw_events_by_session(session_id) #Receives a list of raw events
    
    #Schema of raw_events:List of dictionaries
        #id: str
        #session_id: str
        #source: str
        #ts: float
        #text: str
        #metadata: dict
    
    if not raw_events:
        raise ValueError(f"No raw events found for session {session_id}")
    
    try:
        # Clean and process session data
        full_data = clean_session_data(raw_events)
        
        if not full_data:
            raise RuntimeError("Cleaning agent returned empty results")
        
        # Chunking logic for the full_data
        faiss_store = _get_faiss_instance()
        
        # Create semantic chunks from the full data
        chunks = chunk_session_data(
            full_data=full_data,
            session_id=session_id,
            embedding_model=faiss_store.embedding_model
        )
        
        if not chunks:
            raise RuntimeError("Chunking produced no results")
        
        db = _get_db_instance()
        node_ids = []
        
        # Store each chunk separately
        for chunk in chunks:
            chunk_content = chunk['content']
            chunk_metadata = chunk['metadata']
            
            # Generate embedding for the chunk content
            embedding = faiss_store.embedding_model.embed_query(chunk_content)
            embedding_bytes = np.array(embedding).tobytes()
            
            # Create a summary for the chunk (for database storage)
            chunk_summary = _create_chunk_summary(chunk_content, chunk_metadata)
            
            # Store in database
            node_id = db.store_session_node(
                session_id=session_id,
                summary=chunk_summary,
                full_data=chunk_content,
                embedding=embedding_bytes
            )
            
            # Add to FAISS index
            faiss_store.add_node(node_id, chunk_summary, chunk_content)
            
            node_ids.append(node_id)
        
        return node_ids
        
    except Exception as e:
        raise RuntimeError(f"Failed to process session {session_id}: {str(e)}")

def _create_chunk_summary(chunk_content: str, chunk_metadata: Dict[str, Any]) -> str:
    """
    Create a summary for a chunk for database storage.
    
    Args:
        chunk_content: The chunk content
        chunk_metadata: Metadata about the chunk
        
    Returns:
        Summary string for the chunk
    """
    session_id = chunk_metadata.get('session_id', 'unknown')
    chunk_index = chunk_metadata.get('chunk_index', 0)
    total_chunks = chunk_metadata.get('total_chunks', 1)
    chunk_size = chunk_metadata.get('chunk_size', len(chunk_content))
    
    # Get a snippet of the content for the summary
    content_snippet = chunk_content[:100].replace('\n', ' ').strip()
    if len(chunk_content) > 100:
        content_snippet += "..."
    
    summary = f"Chunk {chunk_index + 1}/{total_chunks} ({chunk_size} chars) from session {session_id}: {content_snippet}"
    
    return summary


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
        print("🧹 Clearing all data from EdgeElite storage system...")
        
        # Clear FAISS index
        print("🔍 Clearing FAISS vector index...")
        faiss_store = _get_faiss_instance()
        faiss_store.clear_index()
        
        # Clear database (drop and recreate tables)
        print("🗄️ Clearing SQLite database...")
        db = _get_db_instance()
        db.init_database(clear_existing=True)
        
        # Reset global instances to ensure fresh state
        global _db_instance, _faiss_instance
        _db_instance = None
        _faiss_instance = None
        
        print("✅ All data cleared successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to clear all data: {e}")
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