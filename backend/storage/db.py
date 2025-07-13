"""
SQLite database operations for the storage system.
Manages raw events and processed session nodes.
"""
import sqlite3
import os
from typing import List, Dict, Any, Optional, Tuple
from .utils import generate_uuid, serialize_metadata, deserialize_metadata


class StorageDB:
    """SQLite database manager for the storage system."""
    
    def __init__(self, db_path: str = "storage.db"):
        """Initialize database connection and create tables."""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self, clear_existing: bool = False):
        """Create database tables, optionally clearing existing data first."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if clear_existing:
                # Drop existing tables and indexes
                cursor.execute("DROP TABLE IF EXISTS raw_events")
                cursor.execute("DROP TABLE IF EXISTS nodes")
                cursor.execute("DROP INDEX IF EXISTS idx_raw_events_session")
                cursor.execute("DROP INDEX IF EXISTS idx_nodes_session")
                print("🗑️ Dropped existing database tables")
            
            # Create raw_events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS raw_events (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    ts DATETIME NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Create nodes table (session-level processed data)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    full_data TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_events_session ON raw_events(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_session ON nodes(session_id)")
            
            conn.commit()
            
            if clear_existing:
                print("✅ Created fresh database tables")
    
    def store_raw_event(self, session_id: str, source: str, ts: float, text: str, metadata: Dict[str, Any] = None) -> str:
        """Store a raw event from OCR or Audio pipeline."""
        event_id = generate_uuid()
        metadata_str = serialize_metadata(metadata)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO raw_events (id, session_id, source, ts, text, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (event_id, session_id, source, ts, text, metadata_str))
            conn.commit()
        
        return event_id
    
    def get_raw_events_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all raw events for a given session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, session_id, source, ts, text, metadata
                FROM raw_events
                WHERE session_id = ?
                ORDER BY ts ASC
            """, (session_id,))
            
            rows = cursor.fetchall()
            events = []
            for row in rows:
                events.append({
                    'id': row[0],
                    'session_id': row[1],
                    'source': row[2],
                    'ts': row[3],
                    'text': row[4],
                    'metadata': deserialize_metadata(row[5])
                })
            
            return events
    
    def store_session_node(self, session_id: str, summary: str, full_data: str, embedding: bytes) -> str:
        """Store a processed session node with embedding."""
        node_id = generate_uuid()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO nodes (id, session_id, summary, full_data, embedding)
                VALUES (?, ?, ?, ?, ?)
            """, (node_id, session_id, summary, full_data, embedding))
            conn.commit()
        
        return node_id
    
    def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific node by its ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, session_id, summary, full_data, embedding
                FROM nodes
                WHERE id = ?
            """, (node_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'session_id': row[1],
                    'summary': row[2],
                    'full_data': row[3],
                    'embedding': row[4]
                }
            return None
    
    def get_nodes_by_ids(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve multiple nodes by their IDs."""
        if not node_ids:
            return []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            placeholders = ','.join(['?' for _ in node_ids])
            cursor.execute(f"""
                SELECT id, session_id, summary, full_data, embedding
                FROM nodes
                WHERE id IN ({placeholders})
            """, node_ids)
            
            rows = cursor.fetchall()
            nodes = []
            for row in rows:
                nodes.append({
                    'id': row[0],
                    'session_id': row[1],
                    'summary': row[2],
                    'full_data': row[3],
                    'embedding': row[4]
                })
            
            return nodes
    
    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Retrieve all nodes for FAISS initialization."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, session_id, summary, full_data, embedding
                FROM nodes
                ORDER BY session_id
            """)
            
            rows = cursor.fetchall()
            nodes = []
            for row in rows:
                nodes.append({
                    'id': row[0],
                    'session_id': row[1],
                    'summary': row[2],
                    'full_data': row[3],
                    'embedding': row[4]
                })
            
            return nodes
    
    def session_exists(self, session_id: str) -> bool:
        """Check if a session has any raw events."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM raw_events WHERE session_id = ?
            """, (session_id,))
            count = cursor.fetchone()[0]
            return count > 0
    
    def session_processed(self, session_id: str) -> bool:
        """Check if a session has been processed into nodes."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM nodes WHERE session_id = ?
            """, (session_id,))
            count = cursor.fetchone()[0]
            return count > 0 