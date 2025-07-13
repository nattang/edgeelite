#!/usr/bin/env python3
"""
Simple Interactive Search Test
Quick way to test individual search queries and evaluate results.
"""

import sys
import os

# Add the parent storage directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def interactive_search_test():
    """Interactive search testing tool."""
    
    print("=== Interactive FAISS Search Test ===")
    print("Type search queries to test retrieval quality.")
    print("Type 'quit' to exit, 'stats' for system stats.\n")
    
    try:
        from interface import search_similar, get_system_stats
        
        # Show system stats
        stats = get_system_stats()
        print(f"System Status:")
        print(f"  Total nodes: {stats['total_nodes']}")
        print(f"  Sessions processed: {stats['total_sessions_processed']}")
        print(f"  FAISS stats: {stats['faiss_stats']}")
        print()
        
        while True:
            try:
                # Get user input
                query = input("Search query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                elif query.lower() == 'stats':
                    stats = get_system_stats()
                    print(f"\nSystem Stats: {stats}\n")
                    continue
                elif not query:
                    print("Please enter a search query.\n")
                    continue
                
                # Perform search
                print(f"\nSearching for: '{query}'")
                print("-" * 50)
                
                results = search_similar(query, k=5)
                
                if not results:
                    print("No results found.\n")
                    continue
                
                # Display results
                for i, (summary, full_data) in enumerate(results, 1):
                    print(f"\n{i}. {summary}")
                    print(f"   Content: {full_data[:200].replace(chr(10), ' ')}...")
                    
                    # Ask for relevance rating
                    while True:
                        try:
                            rating = input(f"   Relevance (1-5, or 's' to skip): ").strip().lower()
                            if rating == 's':
                                break
                            elif rating in ['1', '2', '3', '4', '5']:
                                relevance_desc = {
                                    '1': 'Not relevant',
                                    '2': 'Slightly relevant', 
                                    '3': 'Moderately relevant',
                                    '4': 'Very relevant',
                                    '5': 'Perfectly relevant'
                                }
                                print(f"   Rating: {rating}/5 ({relevance_desc[rating]})")
                                break
                            else:
                                print("   Please enter 1-5 or 's' to skip")
                        except KeyboardInterrupt:
                            return
                
                print("\n" + "="*60 + "\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}\n")
        
        print("Thanks for testing!")
        
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure you're in the storage directory and dependencies are installed.")
    except Exception as e:
        print(f"Error: {e}")

def quick_quality_check():
    """Quick automated quality check with common queries."""
    
    print("=== Quick Quality Check ===\n")
    
    try:
        from interface import search_similar
        
        # Common test queries
        test_queries = [
            "code programming function",
            "email message client", 
            "password security authentication",
            "meeting presentation slides",
            "shopping headphones reviews"
        ]
        
        for query in test_queries:
            print(f"Query: '{query}'")
            results = search_similar(query, k=3)
            
            if results:
                for i, (summary, _) in enumerate(results, 1):
                    print(f"  {i}. {summary[:80]}...")
            else:
                print("  No results found")
            print()
            
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Main function with menu."""
    
    print("Choose test mode:")
    print("1. Interactive search test")
    print("2. Quick quality check")
    print("3. Exit")
    
    while True:
        try:
            choice = input("\nEnter choice (1-3): ").strip()
            
            if choice == '1':
                interactive_search_test()
                break
            elif choice == '2':
                quick_quality_check()
                break
            elif choice == '3':
                break
            else:
                print("Please enter 1, 2, or 3")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main() 