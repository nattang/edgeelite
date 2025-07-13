"""
Utility functions for the storage system.
Provides UUID generation, timestamp handling, and JSON serialization.
"""
import uuid
import json
import time
from datetime import datetime
from typing import Dict, Any


def generate_uuid() -> str:
    """Generate a unique UUID string."""
    return str(uuid.uuid4())


def current_timestamp() -> float:
    """Get current timestamp as float."""
    return time.time()


def timestamp_to_datetime(ts: float) -> datetime:
    """Convert timestamp to datetime object."""
    return datetime.fromtimestamp(ts)


def datetime_to_timestamp(dt: datetime) -> float:
    """Convert datetime to timestamp."""
    return dt.timestamp()


def serialize_metadata(metadata: Dict[str, Any]) -> str:
    """Serialize metadata dictionary to JSON string."""
    if metadata is None:
        return "{}"
    return json.dumps(metadata)


def deserialize_metadata(metadata_str: str) -> Dict[str, Any]:
    """Deserialize JSON string to metadata dictionary."""
    if not metadata_str:
        return {}
    try:
        return json.loads(metadata_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def validate_session_id(session_id: str) -> bool:
    """Validate that session_id is a non-empty string."""
    return isinstance(session_id, str) and len(session_id.strip()) > 0


def validate_source(source: str) -> bool:
    """Validate that source is either 'ocr' or 'audio'."""
    return source in ['ocr', 'audio'] 