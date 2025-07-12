import sys
from pathlib import Path
from src.retrieval.retriever import QueryEngine

def main():
    print("Handbook Query Tool")
    print("Type 'exit' or 'quit' to end the session")
    print("-" * 50)
    
    # Initialize the query engine
    engine = QueryEngine()
    
    while True:
        try:
            # Get user query
            query = input("\nEnter your question: ").strip()
            
            if query.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
                
            if not query:
                continue
                
            # Search for relevant content with more context
            print("\n🔍 Searching for:", query)
            print("Improving query...")
            improved_query = engine._improve_query(query)
            if improved_query != query:
                print(f"Improved query: {improved_query}")
                
            results = engine.search(improved_query, top_k=3)
            
            if not results:
                print("No relevant information found. Try rephrasing your question.")
                continue
                
            # Display results with better formatting and navigation
            print(f"\n🔍 Found {len(results)} relevant results:")
            for i, result in enumerate(results, 1):
                print(f"\n{'='*120}")
                print(f"📄 Result {i}/{len(results)} (Relevance: {result['score']:.2f})")
                print(f"📂 Source: {result['metadata'].get('source', 'N/A')}")
                print(f"📄 Page: {result['metadata'].get('page', 'N/A')}")
                print(f"📊 Score: {result['score']:.4f}")
                
                # Show section info if available
                if 'metadata' in result and 'section_number' in result['metadata']:
                    print(f"📑 Section: {result['metadata']['section_number']}")
                if 'metadata' in result and 'title' in result['metadata']:
                    print(f"📌 Title: {result['metadata']['title']}")
                
                print(f"\n📝 Content:")
                print("-" * 120)
                
                # Print the complete text with better formatting
                import textwrap
                from textwrap import fill
                
                # Print the full chunk text with wrapping
                chunk_text = result['text'].strip()
                print(fill(chunk_text, width=120, initial_indent='  ', subsequent_indent='  '))
                
                # Show chunk position if available
                if 'metadata' in result and 'chunk_number' in result['metadata'] and 'total_chunks' in result['metadata']:
                    print(f"\n📚 Chunk {result['metadata']['chunk_number']} of {result['metadata']['total_chunks']}")
                
                print("-" * 120)
                
                # Show metadata if available (excluding already shown fields)
                shown_fields = {'source', 'page', 'section_number', 'title', 'chunk_number', 'total_chunks'}
                if 'metadata' in result and result['metadata']:
                    metadata_items = [(k, v) for k, v in result['metadata'].items() 
                                   if k not in shown_fields and not k.startswith('_')]
                    if metadata_items:
                        print("\n📋 Additional Metadata:")
                        for key, value in metadata_items:
                            print(f"  • {key}: {value}")
                
                print("="*120)
                
                # Add navigation prompt
                if i < len(results):
                    print("\nPress Enter to see next result or type 'q' to quit...")
                    if input().strip().lower() == 'q':
                        break
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            continue

if __name__ == "__main__":
    main()
