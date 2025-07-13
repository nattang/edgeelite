#!/usr/bin/env python3
"""
Simple Database Cleaner - No complex imports, just works!
"""

import os
import sqlite3
import shutil

def clear_database():
    """Clear the database directly - simple and effective."""
    
    # Get the storage directory (parent of test_files)
    storage_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Database file path
    db_path = os.path.join(storage_dir, "storage.db")
    
    # FAISS index directory
    faiss_dir = os.path.join(storage_dir, "faiss_index")
    
    print("🧹 Simple Database Cleaner")
    print("=" * 30)
    
    # Clear SQLite database
    print("\n🗄️ Clearing SQLite database...")
    try:
        if os.path.exists(db_path):
            # Delete the database file
            os.remove(db_path)
            print(f"✅ Deleted: {db_path}")
        else:
            print(f"ℹ️  Database file not found: {db_path}")
    except Exception as e:
        print(f"❌ Error clearing database: {e}")
    
    # Clear FAISS index
    print("\n🔍 Clearing FAISS index...")
    try:
        if os.path.exists(faiss_dir):
            # Delete the entire FAISS directory
            shutil.rmtree(faiss_dir)
            print(f"✅ Deleted: {faiss_dir}")
        else:
            print(f"ℹ️  FAISS directory not found: {faiss_dir}")
    except Exception as e:
        print(f"❌ Error clearing FAISS: {e}")
    
    # Clear __pycache__ directories
    print("\n🧹 Clearing cache files...")
    try:
        pycache_dir = os.path.join(storage_dir, "__pycache__")
        if os.path.exists(pycache_dir):
            shutil.rmtree(pycache_dir)
            print(f"✅ Deleted: {pycache_dir}")
        else:
            print(f"ℹ️  Cache directory not found: {pycache_dir}")
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")
    
    print("\n✅ Database cleared successfully!")
    print("Ready for fresh data!")

if __name__ == "__main__":
    clear_database() 