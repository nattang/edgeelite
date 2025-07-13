#!/usr/bin/env python3
"""
Phase 4 Automated Test Runner for EdgeElite Journal Feature

This script automates the complete Phase 4 testing process:
1. Sets up demo data
2. Tests the pipeline
3. Verifies everything works correctly

Run this script to quickly validate that the journal feature is ready for demo.
"""

import subprocess
import sys
import os
import time
from typing import List, Tuple

def run_command(cmd: List[str], description: str) -> Tuple[bool, str]:
    """Run a command and return success status and output."""
    try:
        print(f"🔄 {description}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True, result.stdout
        else:
            print(f"❌ {description} failed")
            print(f"Error: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out")
        return False, "Timeout"
    except Exception as e:
        print(f"💥 {description} crashed: {e}")
        return False, str(e)

def check_dependencies() -> bool:
    """Check if required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    # Check if we're in the right directory
    if not os.path.exists("storage"):
        print("❌ Not in backend directory. Please run from backend/")
        return False
    
    # Check if storage module can be imported
    try:
        sys.path.insert(0, os.getcwd())
        from storage.interface import store_raw_ocr_event
        print("✅ Storage module available")
        return True
    except ImportError as e:
        print(f"❌ Storage module not available: {e}")
        return False

def main():
    """Main test runner function."""
    print("🚀 EdgeElite Journal Feature - Phase 4 Automated Test Runner")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please ensure you're in the backend directory.")
        return False
    
    # Test sequence
    tests = [
        {
            "name": "Demo Data Setup",
            "command": [sys.executable, "seed_demo_data.py"],
            "description": "Setting up demo data with walk remedy",
            "required": True
        },
        {
            "name": "Pipeline Testing",
            "command": [sys.executable, "test_journal_pipeline.py"],
            "description": "Testing storage and RAG pipeline",
            "required": True
        }
    ]
    
    results = {}
    
    print(f"\n📋 Running {len(tests)} test suites...")
    
    for test in tests:
        print(f"\n{'='*40}")
        print(f"🧪 {test['name']}")
        print(f"{'='*40}")
        
        success, output = run_command(test["command"], test["description"])
        results[test["name"]] = success
        
        if not success and test["required"]:
            print(f"\n❌ Required test '{test['name']}' failed!")
            print("Stopping test suite...")
            break
        
        # Brief pause between tests
        time.sleep(1)
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    if passed == total:
        print(f"\n🎉 All Phase 4 tests passed!")
        print(f"\n📝 Next steps:")
        print(f"1. Start the backend: python -m uvicorn main:app --reload")
        print(f"2. Start the frontend: npm run dev")
        print(f"3. Test the journal feature in the UI")
        print(f"4. Say 'huge headache, calendar is insane' and check for walk remedy!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        print(f"Please check the errors above and fix them before proceeding.")
        return False

def quick_test():
    """Run a quick test of the RAG system."""
    print("\n🔍 Quick RAG Test...")
    
    try:
        sys.path.insert(0, os.getcwd())
        from storage.interface import search_similar
        
        # Test search for walk remedy
        results = search_similar("stressed headache need break", k=3)
        
        print(f"Found {len(results)} similar results:")
        remedy_found = False
        
        for i, (summary, content) in enumerate(results, 1):
            print(f"  {i}. {summary[:50]}...")
            if "walk without" in content.lower() or "15-minute" in content.lower():
                print(f"     🎯 FOUND THE WALK REMEDY!")
                remedy_found = True
        
        if remedy_found:
            print("✅ RAG system is working correctly!")
            return True
        else:
            print("⚠️  Walk remedy not found in search results")
            return False
            
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Full automated test suite")
    print("2. Quick RAG test only")
    print("3. Demo data setup only")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
    except KeyboardInterrupt:
        print("\nTest cancelled by user")
        sys.exit(1)
    
    if choice == "1":
        success = main()
        sys.exit(0 if success else 1)
    elif choice == "2":
        success = quick_test()
        sys.exit(0 if success else 1)
    elif choice == "3":
        success, _ = run_command([sys.executable, "seed_demo_data.py"], "Setting up demo data")
        sys.exit(0 if success else 1)
    else:
        print("Invalid choice, running full test suite...")
        success = main()
        sys.exit(0 if success else 1) 