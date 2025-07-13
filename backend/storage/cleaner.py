"""
Text cleaning and processing for session data.
Contains functions for cleaning and aligning OCR/Audio data with chronological stream approach.
"""
import re
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

# Configuration constants (kept for backward compatibility)
DEFAULT_ALIGNMENT_TOLERANCE = 1.0  # seconds
DEFAULT_ALIGNMENT_STRATEGY = "window"  # "window", "nearest", or "bucket"

# Chunking configuration
DEFAULT_CHUNK_MIN_SIZE = 100  # minimum chunk size in characters
DEFAULT_BREAKPOINT_THRESHOLD_TYPE = "percentile"  # most reliable default

def clean_and_normalize_text(text: str) -> str:
    """
    Perform comprehensive text cleaning and normalization.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned and normalized text
    """
    if not text:
        return ""
    
    # Remove extra whitespace (multiple spaces, tabs, newlines)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Remove non-printable characters (keep only ASCII printable)
    text = re.sub(r'[^\x20-\x7E]', '', text)
    
    # Remove duplicate punctuation
    text = re.sub(r'([.!?]){2,}', r'\1', text)
    
    # Clean up common OCR artifacts
    text = re.sub(r'[|\\/_~`]', '', text)  # Remove common OCR noise
    text = re.sub(r'\b[a-zA-Z]\b', '', text)  # Remove single characters
    
    # Final cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def create_chronological_stream(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create a simple chronological stream of all events sorted by timestamp.
    
    Args:
        events: List of raw events from a session
        
    Returns:
        List of events sorted by timestamp with cleaned text
    """
    if not events:
        return []
    
    # Clean text for all events and filter out empty ones
    # Pre-processing of each 'event' block before combining them into a chronological stream
    cleaned_events = []
    for event in events:
        cleaned_text = clean_and_normalize_text(event['text'])
        if cleaned_text:  # Only keep events with meaningful text
            event_copy = event.copy()
            event_copy['text'] = cleaned_text #Replacing the 'text' field with the cleaned text
            cleaned_events.append(event_copy)
    
    # Sort by timestamp for chronological order
    return sorted(cleaned_events, key=lambda x: x['ts'])

#This makes the "string" that will be used to create the chronological stream - which is what is eventually fed to our LLM
def format_chronological_event(event: Dict[str, Any]) -> str:
    """
    Format a single event for the chronological stream.
    
    Args:
        event: Single event dictionary
        
    Returns:
        Formatted string representation of the event
    """
    import datetime
    
    # Convert timestamp to readable format
    dt = datetime.datetime.fromtimestamp(event['ts'])
    time_str = dt.strftime("%H:%M:%S")
    
    source_label = event['source'].upper()
    text = event['text']
    
    return f"[{source_label} @ {time_str}] {text}"

def clean_session_data(
    events: List[Dict[str, Any]], 
    alignment_tolerance: float = DEFAULT_ALIGNMENT_TOLERANCE,
    alignment_strategy: str = DEFAULT_ALIGNMENT_STRATEGY
) -> str:
    """
    Clean and process session data into a chronological stream.
    
    This function:
    1. Cleans and normalizes text content
    2. Creates a chronological stream of OCR and Audio events
    3. Returns the full chronological data for chunking
    
    Args:
        events: List of raw events from a session
        alignment_tolerance: Not used in this implementation (kept for compatibility)
        alignment_strategy: Not used in this implementation (kept for compatibility)
        
    Returns:
        Full chronological data as a string
    """
    if not events:
        return ""
    
    # Create chronological stream
    chronological_events = create_chronological_stream(events)
    
    if not chronological_events:
        return ""
    
    # Format each event for the stream
    formatted_events = []
    for event in chronological_events:
        formatted_event = format_chronological_event(event)
        formatted_events.append(formatted_event)
    
    # Create full data as chronological narrative
    full_data = "\n".join(formatted_events)
    
    return full_data

def chunk_session_data(
    full_data: str, 
    session_id: str, 
    embedding_model,
    min_chunk_size: int = DEFAULT_CHUNK_MIN_SIZE,
    breakpoint_threshold_type: str = DEFAULT_BREAKPOINT_THRESHOLD_TYPE
) -> List[Dict[str, Any]]:
    """
    Chunk session data using SemanticChunker for better semantic coherence.
    
    Args:
        full_data: The cleaned chronological stream text
        session_id: Session identifier for metadata
        embedding_model: The embedding model instance (from FAISS store)
        min_chunk_size: Minimum chunk size in characters
        breakpoint_threshold_type: Semantic chunking threshold type
        
    Returns:
        List of chunk dictionaries with content and metadata
    """
    if not full_data or not full_data.strip():
        return []
    
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_core.documents import Document
        
        # Initialize SemanticChunker with the existing embedding model
        text_splitter = SemanticChunker(
            embedding_model,
            breakpoint_threshold_type=breakpoint_threshold_type,
            min_chunk_size=min_chunk_size
        )
        
        # Create documents from the full data
        documents = text_splitter.create_documents([full_data])
        
        # Convert documents to our chunk format
        chunks = []
        for i, doc in enumerate(documents):
            chunk = {
                'content': doc.page_content,
                'metadata': {
                    'session_id': session_id,
                    'chunk_index': i,
                    'total_chunks': len(documents),
                    'chunk_size': len(doc.page_content),
                    'breakpoint_threshold_type': breakpoint_threshold_type,
                    **doc.metadata  # Include any metadata from SemanticChunker
                }
            }
            chunks.append(chunk)
        
        return chunks
        
    except ImportError:
        print("Warning: langchain_experimental not available, falling back to basic chunking")
        return _fallback_chunking(full_data, session_id, min_chunk_size)
    
    except Exception as e:
        print(f"Error in semantic chunking: {e}, falling back to basic chunking")
        return _fallback_chunking(full_data, session_id, min_chunk_size)

def _fallback_chunking(
    full_data: str, 
    session_id: str, 
    chunk_size: int = 1000
) -> List[Dict[str, Any]]:
    """
    Fallback chunking method when SemanticChunker is not available.
    
    Args:
        full_data: The text to chunk
        session_id: Session identifier
        chunk_size: Size of each chunk in characters
        
    Returns:
        List of chunk dictionaries
    """
    if not full_data:
        return []
    
    # Split by newlines to respect event boundaries
    lines = full_data.split('\n')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for line in lines:
        line_size = len(line) + 1  # +1 for newline
        
        if current_size + line_size > chunk_size and current_chunk:
            # Create chunk from current lines
            chunk_content = '\n'.join(current_chunk)
            chunks.append({
                'content': chunk_content,
                'metadata': {
                    'session_id': session_id,
                    'chunk_index': len(chunks),
                    'chunk_size': len(chunk_content),
                    'chunking_method': 'fallback'
                }
            })
            current_chunk = [line]
            current_size = line_size
        else:
            current_chunk.append(line)
            current_size += line_size
    
    # Add the last chunk
    if current_chunk:
        chunk_content = '\n'.join(current_chunk)
        chunks.append({
            'content': chunk_content,
            'metadata': {
                'session_id': session_id,
                'chunk_index': len(chunks),
                'chunk_size': len(chunk_content),
                'chunking_method': 'fallback'
            }
        })
    
    # Update total_chunks in metadata
    for chunk in chunks:
        chunk['metadata']['total_chunks'] = len(chunks)
    
    return chunks

def preprocess_for_embedding(text: str) -> str:
    """
    Preprocess text before embedding generation.
    
    Args:
        text: Text to preprocess
        
    Returns:
        Preprocessed text optimized for embedding
    """
    if not text:
        return ""
    
    # Basic cleaning
    text = clean_and_normalize_text(text)
    
    # Normalize case for better embedding consistency
    text = text.lower()
    
    # Remove excessive punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    # Limit length for embedding model
    if len(text) > 512:
        text = text[:512]
    
    return text 


