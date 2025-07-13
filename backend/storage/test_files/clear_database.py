#!/usr/bin/env python3
"""
Database Clearing Script for EdgeElite Storage System
Use this script to completely reset the database and FAISS index before testing.
"""

import sys
import os

# Add the parent storage directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def clear_database():
    """Clear all data from the storage system."""
    
    print("🧹 EdgeElite Storage System - Database Cleaner")
    print("=" * 50)
    
    try:
        from interface import clear_all_data, get_system_stats
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please run from the backend directory using:")
        print("python3 -m storage.test_files.clear_database")
        return False
    
    try:
        # Show current system state
        print("\n📊 Current System State:")
        try:
            stats = get_system_stats()
            print(f"  Total nodes: {stats['total_nodes']}")
            print(f"  Sessions processed: {stats['total_sessions_processed']}")
            print(f"  FAISS stats: {stats['faiss_stats']}")
        except Exception as e:
            print(f"  Could not get stats (system may be empty): {e}")
        
        # Ask for confirmation
        print("\n⚠️  This will permanently delete ALL data including:")
        print("  - All raw OCR and Audio events")
        print("  - All processed sessions and chunks")
        print("  - Complete FAISS vector index")
        
        confirm = input("\nAre you sure you want to proceed? (yes/no): ").strip().lower()
        
        if confirm not in ['yes', 'y']:
            print("❌ Operation cancelled.")
            return False
        
        # Clear all data
        print("\n🚀 Starting database clearing process...")
        success = clear_all_data()
        
        if success:
            print("\n📊 Verifying system is empty...")
            try:
                stats = get_system_stats()
                print(f"  Total nodes: {stats['total_nodes']}")
                print(f"  Sessions processed: {stats['total_sessions_processed']}")
                print(f"  FAISS stats: {stats['faiss_stats']}")
                
                if stats['total_nodes'] == 0 and stats['total_sessions_processed'] == 0:
                    print("\n✅ Database successfully cleared! System is ready for fresh data.")
                else:
                    print("\n⚠️  Warning: System may not be completely empty.")
                    
            except Exception as e:
                print(f"  Could not verify empty state: {e}")
                
        return success
        
    except Exception as e:
        print(f"❌ Error during database clearing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function with menu options."""
    
    if len(sys.argv) > 1 and sys.argv[1] in ['--force', '-f']:
        # Force clear without confirmation
        print("🚀 Force clearing database (no confirmation)...")
        try:
            from interface import clear_all_data
            success = clear_all_data()
            sys.exit(0 if success else 1)
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("Please run from the backend directory using:")
            print("python3 -m storage.test_files.clear_database --force")
            sys.exit(1)
    
    # Interactive mode
    success = clear_database()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 