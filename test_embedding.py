import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.embedding.embedder import TextEmbedder

def test_embedding():
    print("Testing TextEmbedder...")
    
    # Initialize the embedder
    embedder = TextEmbedder()
    
    # Test data
    test_docs = [
        {
            "text": "The University of Vavuniya offers various undergraduate programs.",
            "metadata": {"source": "test", "page": 1, "doc_type": "test"}
        },
        {
            "text": "The Faculty of Applied Sciences provides quality education in computing.",
            "metadata": {"source": "test", "page": 2, "doc_type": "test"}
        }
    ]
    
    # Test adding documents
    print("\nAdding test documents...")
    result = embedder.add_documents(test_docs)
    print(f"Added {result['total_documents']} documents as {result['total_chunks']} chunks")
    
    # Test search
    print("\nTesting search...")
    query = "What programs does the university offer?"
    results = embedder.search(query, top_k=2)
    
    print(f"\nSearch results for: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i} (Score: {result['score']:.4f}):")
        print(f"Text: {result['text']}")
        print(f"Metadata: {result['metadata']}")
    
    # Clean up
    print("\nCleaning up test data...")
    embedder.delete_collection()
    print("Test completed successfully!")

if __name__ == "__main__":
    test_embedding()
