import sys
from pathlib import Path
from typing import Dict, Any, List
from textwrap import fill

# Add src to path to allow absolute imports
sys.path.append(str(Path(__file__).parent))

from src.retrieval.retriever import QueryEngine, SearchResult
from src.generation.generator import ResponseGenerator, GenerationConfig

def format_result(result: SearchResult) -> str:
    """Format a single search result for display."""
    output = []
    
    # Basic result info
    output.append(f"\n{'='*120}")
    output.append(f"📄 Result (Relevance: {result.score*100:.1f}%)")
    
    # Add metadata if available
    if result.metadata:
        if 'source' in result.metadata:
            output.append(f"📂 Source: {result.metadata['source']}")
        if 'page' in result.metadata:
            output.append(f"📄 Page: {result.metadata['page']}")
        if 'section' in result.metadata:
            output.append(f"📑 Section: {result.metadata['section']}")
        if 'title' in result.metadata:
            output.append(f"📌 Title: {result.metadata['title']}")
    
    # Add the content with proper wrapping
    output.append("\n📝 Content:")
    output.append("-" * 120)
    output.append(fill(result.text.strip(), width=120, 
                     initial_indent='  ', 
                     subsequent_indent='  '))
    output.append("-" * 120)
    
    # Show additional metadata
    shown_fields = {'source', 'page', 'section', 'title', 'chunk_number', 'total_chunks'}
    if result.metadata:
        metadata_items = [
            (k, v) for k, v in result.metadata.items()
            if k not in shown_fields and not k.startswith('_') and v is not None
        ]
        if metadata_items:
            output.append("\n📋 Additional Metadata:")
            for key, value in metadata_items:
                if isinstance(value, (str, int, float, bool)):
                    output.append(f"  • {key}: {value}")
                else:
                    output.append(f"  • {key}: {str(value)[:100]}...")
    
    output.append("=" * 120)
    return "\n".join(output)

def main():
    print("📚 UoV FAS Handbook Query Tool with RAG")
    print("Type 'exit' or 'quit' to end the session")
    print("-" * 50)
    
    try:
        # Initialize the query engine
        print("\n🔧 Initializing query engine...")
        engine = QueryEngine()
        print("✅ Query engine ready!")
        
        # Initialize the response generator
        print("🔧 Initializing response generator...")
        generator = ResponseGenerator(GenerationConfig(
            model_name="command-a-03-2025",
            temperature=0.7,
            max_tokens=500
        ))
        print("✅ Response generator ready!")
        
        while True:
            try:
                # Get user query
                query = input("\n❓ Enter your question: ").strip()
                
                if not query:
                    continue
                    
                if query.lower() in ['exit', 'quit']:
                    print("\n👋 Goodbye!")
                    break
                
                # Search for relevant content
                print("\n🔍 Searching for relevant information...")
                results = engine.search(query, top_k=3)
                
                if not results:
                    print("\n❌ No relevant information found. Try rephrasing your question.")
                    continue
                
                # Generate RAG response
                print("🤖 Generating response...")
                response = generator.generate_response(query, results)
                
                # Display the generated response
                print(f"\n💬 {response}\n")
                
                # Show source references
                print("\n📚 Source References:")
                for i, result in enumerate(results, 1):
                    source = result.metadata.get('source', 'Unknown source')
                    page = result.metadata.get('page', 'N/A')
                    print(f"{i}. {source} (Page {page})")
                    print(f"   {result.text[:150]}...\n")
                
                # Option to view full results
                view_full = input("\nView full search results? (y/n): ").strip().lower()
                if view_full == 'y':
                    print(f"\n🔍 Found {len(results)} relevant results:")
                    for i, result in enumerate(results, 1):
                        print(f"\n📊 Showing result {i} of {len(results)}:")
                        print(format_result(result))
                        
            except KeyboardInterrupt:
                print("\n\n🛑 Operation cancelled by user.")
                break
            except Exception as e:
                print(f"\n⚠️ An error occurred: {str(e)}")
                continue
                
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()