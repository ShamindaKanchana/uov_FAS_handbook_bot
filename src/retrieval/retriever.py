import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.embedding.config import (
    DEFAULT_MODEL,
    DEFAULT_COLLECTION,
    QDRANT_STORAGE_PATH,
)
from src.embedding.qdrant_singleton import QdrantClientSingleton
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QueryEngine:
    """
    A class to handle querying the vector database with enhanced search capabilities.
    """
    
    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        storage_path: Optional[Union[str, Path]] = QDRANT_STORAGE_PATH,
    ):
        """
        Initialize the QueryEngine with an existing Qdrant collection.
        
        Args:
            collection_name: Name of the Qdrant collection
            storage_path: Path where Qdrant data is stored. If None, uses in-memory storage.
        """
        self.collection_name = collection_name
        self.model = SentenceTransformer(DEFAULT_MODEL)
        self.client = QdrantClientSingleton(storage_path).get_client()
        logger.info(f"Connected to Qdrant collection: {collection_name}")

    def _correct_spelling(self, text: str) -> str:
        """Correct common spelling mistakes in the query."""
        text = text.lower()
        corrections = {
            'mathamatics': 'mathematics',
            'sience': 'science',
            'bio sience': 'bioscience',
            'computing': 'computer science',
            'it': 'information technology',
            'cs': 'computer science',
            'bsc': 'bachelor of science'
        }
        for wrong, right in corrections.items():
            text = text.replace(wrong, right)
        return text

    def _improve_query(self, query: str) -> str:
        """Enhance the search query with spelling correction and expansion."""
        query = self._correct_spelling(query)
        query_terms = query.lower().split()
        
        # Add relevant terms based on query content
        if 'subject' in query_terms or 'course' in query_terms:
            query += " modules curriculum"
        if 'information' in query_terms or 'detail' in query_terms:
            query += " description overview"
        if 'degree' in query_terms or 'program' in query_terms:
            query += " qualification study"
            
        return query.strip()

    def search(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: Optional[float] = None,
        filter_condition: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search the vector database with enhanced query processing.
        
        Args:
            query: The search query
            top_k: Maximum number of results to return
            score_threshold: Minimum similarity score (0-1) for results
            filter_condition: Optional filter conditions (e.g., {"source": "handbook"})
            
        Returns:
            List of search results with scores and metadata
        """
        if not query or not query.strip():
            logger.warning("Empty search query provided")
            return []
        """
        Search the vector database with enhanced query processing.
        
        Args:
            query: The search query
            top_k: Maximum number of results to return
            score_threshold: Minimum similarity score (0-1) for results
            filter_condition: Optional filter conditions (e.g., {"source": "handbook"})
            
        Returns:
            List of search results with scores and metadata
        """
        if not query or not query.strip():
            logger.warning("Empty search query provided")
            return []
            
        try:
            # Process and improve the query
            processed_query = self._improve_query(query)
            logger.debug(f"Original query: '{query}' -> Processed: '{processed_query}'")
            
            # Generate query embedding
            query_embedding = self.model.encode(
                processed_query,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()
            
            # Prepare filter conditions if provided
            must_conditions = []
            if filter_condition:
                for field, value in filter_condition.items():
                    must_conditions.append(
                        FieldCondition(
                            key=f"metadata.{field}",
                            match=MatchValue(value=value)
                        )
                    )
            
            query_filter = Filter(must=must_conditions) if must_conditions else None
            
            # Execute search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k * 2,  # Get more results for deduplication
                query_filter=query_filter,
                score_threshold=score_threshold or 0.0
            )
            
            # Process and deduplicate results
            seen_texts = set()
            results = []
            
            for hit in search_results:
                text = hit.payload.get('text', '').strip()
                if not text or text in seen_texts:
                    continue
                    
                seen_texts.add(text)
                results.append({
                    'id': hit.id,
                    'score': float(hit.score),
                    'text': text,
                    'metadata': hit.payload.get('metadata', {})
                })
                
                if len(results) >= top_k:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}", exc_info=True)
            return []

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection."""
        try:
            collection = self.client.get_collection(self.collection_name)
            return {
                'name': collection.name,
                'vectors_count': collection.vectors_count,
                'status': collection.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}


def test_queries():
    """Test the query engine with sample queries."""
    import time
    
    print("Testing Query Engine...")
    engine = QueryEngine()
    
    test_cases = [
        "Subjects in IT degree",
        "Bio Science department information",
        "Duration of Computer Science program",
        "Admission requirements for Applied Mathematics"
    ]
    
    for query in test_cases:
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print(f"{'='*80}")
        
        start_time = time.time()
        results = engine.search(query, top_k=2)
        elapsed = (time.time() - start_time) * 1000
        
        if not results:
            print("No results found.")
            continue
            
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} (Score: {result['score']:.4f}, {elapsed:.2f}ms) ---")
            print(f"Source: {result['metadata'].get('source', 'N/A')}")
            print(f"Page: {result['metadata'].get('page', 'N/A')}")
            print(f"Text: {result['text'][:300]}{'...' if len(result['text']) > 300 else ''}")


if __name__ == "__main__":
    test_queries()