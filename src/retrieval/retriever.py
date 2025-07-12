import logging
import sys
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.embedding.config import DEFAULT_MODEL
from src.embedding.qdrant_singleton import QdrantClientSingleton

# Configuration
DEFAULT_COLLECTION = "handbook_chunks"
DEFAULT_QDRANT_PATH = str(Path(__file__).parent.parent.parent / "qdrant_handbook")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('retriever.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Container for search results with formatted output methods."""
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]
    
    def format(self, max_length: int = 500) -> str:
        """Format the search result into a readable string."""
        text = self.text[:max_length]
        if len(self.text) > max_length:
            text += "..."
            
        return (
            f"Relevance: {self.score:.1%}\n"
            f"Source: {self.metadata.get('source', 'N/A')}\n"
            f"Page: {self.metadata.get('page', 'N/A')}\n"
            f"Content: {text}"
        )

class QueryEngine:
    """
    A class to handle querying the vector database with enhanced search capabilities.
    Includes spelling correction, query expansion, and result formatting.
    """
    
    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        storage_path: Union[str, Path] = DEFAULT_QDRANT_PATH,
        model_name: str = DEFAULT_MODEL
    ):
        """
        Initialize the QueryEngine with an existing Qdrant collection.
        
        Args:
            collection_name: Name of the Qdrant collection
            storage_path: Path where Qdrant data is stored
            model_name: Name of the sentence transformer model to use
        """
        self.collection_name = collection_name
        self.model = SentenceTransformer(model_name)
        self.storage_path = Path(storage_path)
        
        try:
            self.client = QdrantClientSingleton(self.storage_path).get_client()
            self._verify_collection()
            logger.info(f"Connected to Qdrant collection: {collection_name} at {storage_path}")
        except Exception as e:
            logger.error(f"Failed to initialize QueryEngine: {str(e)}")
            raise
    
    def _verify_collection(self) -> bool:
        """Verify that the collection exists and is accessible."""
        try:
            collections = self.client.get_collections()
            collection_names = {c.name for c in collections.collections}
            if self.collection_name not in collection_names:
                raise ValueError(
                    f"Collection '{self.collection_name}' not found. "
                    f"Available collections: {', '.join(collection_names) or 'None'}"
                )
            return True
        except Exception as e:
            logger.error(f"Collection verification failed: {str(e)}")
            raise
    
    def _correct_spelling(self, text: str) -> str:
        """
        Correct common spelling mistakes in the query.
        
        Args:
            text: Input text to correct
            
        Returns:
            str: Text with common spelling corrections applied
        """
        if not text or not isinstance(text, str):
            return ""
            
        text = text.lower()
        corrections = {
            'mathamatics': 'mathematics',
            'sience': 'science',
            'bio sience': 'bioscience',
            'computing': 'computer science',
            'it': 'information technology',
            'cs': 'computer science',
            'bsc': 'bachelor of science',
            'enviornment': 'environment',
            'enviroment': 'environment',
            'degreee': 'degree',
            'programme': 'program',
            'requirment': 'requirement',
            'admissionn': 'admission',
            'qualificationn': 'qualification',
            'cource': 'course'
        }
        
        # Split into words but preserve original spacing
        words = text.split()
        corrected_words = []
        
        for word in words:
            corrected = corrections.get(word.lower(), word)
            corrected_words.append(corrected)
        
        return ' '.join(corrected_words)
    
    def _improve_query(self, query: str) -> str:
        """
        Enhance the search query with minimal modifications.
        
        Args:
            query: Original search query
            
        Returns:
            str: Improved query with basic spelling corrections
        """
        # First correct spelling
        query = self._correct_spelling(query)
        return query

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.3,  # Lowered threshold to get more results
        filter_condition: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Search the vector database with enhanced query processing.
        
        Args:
            query: The search query string
            top_k: Maximum number of results to return (default: 5)
            score_threshold: Minimum similarity score (0-1) for results (default: 0.3)
            filter_condition: Optional filter conditions (e.g., {"department": "cs"})
            **kwargs: Additional search parameters
            
        Returns:
            List[SearchResult]: List of search results with scores and metadata
        """
        if not query or not isinstance(query, str) or not query.strip():
            logger.warning("Invalid or empty search query provided")
            return []
            
        try:
            # Use the original query without aggressive processing
            logger.info(f"Searching for: '{query}'")
            
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Prepare filter conditions
            query_filter = self._build_filter_condition(filter_condition)
            
            # Execute search with error handling
            search_results = self._execute_search(
                query_embedding=query_embedding,
                top_k=top_k * 2,  # Get more results for deduplication
                score_threshold=score_threshold,
                query_filter=query_filter,
                **kwargs
            )
            
            # Process and deduplicate results
            results = self._process_search_results(search_results, top_k)
            
            # If no results, try again with a lower threshold
            if not results and score_threshold > 0.1:
                logger.debug("No results found, trying with lower threshold")
                search_results = self._execute_search(
                    query_embedding=query_embedding,
                    top_k=top_k * 2,
                    score_threshold=0.1,  # Very low threshold
                    query_filter=query_filter,
                    **kwargs
                )
                results = self._process_search_results(search_results, top_k)
                
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}", exc_info=True)
            return []
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text."""
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise
    
    def _build_filter_condition(self, filter_condition: Optional[Dict[str, Any]] = None) -> Optional[Filter]:
        """Build Qdrant filter condition from dictionary."""
        if not filter_condition:
            return None
            
        must_conditions = []
        
        for field, value in filter_condition.items():
            if not value:
                continue
                
            if isinstance(value, (list, set, tuple)):
                # Handle multiple values with 'or' condition
                conditions = [
                    FieldCondition(
                        key=f"metadata.{field}",
                        match=MatchValue(value=v)
                    )
                    for v in value if v
                ]
                if conditions:
                    must_conditions.extend(conditions)
            else:
                must_conditions.append(
                    FieldCondition(
                        key=f"metadata.{field}",
                        match=MatchValue(value=value)
                    )
                )
        
        return Filter(must=must_conditions) if must_conditions else None
    
    def _execute_search(
        self,
        query_embedding: List[float],
        top_k: int,
        score_threshold: float,
        query_filter: Optional[Filter] = None,
        **kwargs
    ) -> List[Any]:
        """Execute the search against Qdrant."""
        try:
            # First try with query_points API (newer versions of Qdrant)
            try:
                search_results = self.client.query_points(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,  # Try without the tuple first
                    limit=top_k * 2,
                    query_filter=query_filter,
                    score_threshold=score_threshold,
                    **kwargs
                )
                logger.debug(f"Found {len(search_results)} results using query_points")
                return search_results
            except Exception as e:
                logger.debug(f"query_points with direct vector failed, trying with field name: {str(e)}")
                
                # If that fails, try with the field name
                try:
                    search_results = self.client.query_points(
                        collection_name=self.collection_name,
                        query_vector=("text", query_embedding),  # Try with field name
                        limit=top_k * 2,
                        query_filter=query_filter,
                        score_threshold=score_threshold,
                        **kwargs
                    )
                    logger.debug(f"Found {len(search_results)} results using query_points with field name")
                    return search_results
                except Exception as e2:
                    logger.debug(f"query_points with field name failed, falling back to search: {str(e2)}")
            
            # Fall back to the older search method
            try:
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=top_k * 2,  # Get more results for deduplication
                    query_filter=query_filter,
                    score_threshold=score_threshold,
                    **kwargs
                )
                logger.debug(f"Found {len(search_results)} results using search")
                return search_results
            except Exception as e3:
                logger.error(f"Search API failed: {str(e3)}", exc_info=True)
                return []
                
        except Exception as e:
            logger.error(f"Search execution failed: {str(e)}", exc_info=True)
            return []
    
    def _process_search_results(
        self,
        search_results: List[Any],
        top_k: int
    ) -> List[SearchResult]:
        """Process and deduplicate search results."""
        seen_texts = set()
        results = []
        
        if not search_results:
            logger.debug("No search results returned from Qdrant")
            return results
            
        logger.debug(f"Processing {len(search_results)} raw search results")
        
        for hit in search_results:
            try:
                # Handle different Qdrant result formats
                payload = {}
                score = 0.0
                point_id = 'unknown'
                
                # Extract payload, score, and ID based on the result format
                if hasattr(hit, 'payload'):  # Qdrant PointStruct
                    payload = hit.payload or {}
                    score = getattr(hit, 'score', 0.0)
                    point_id = getattr(hit, 'id', 'unknown')
                elif hasattr(hit, 'version'):  # ScoredPoint format
                    payload = getattr(hit, 'payload', {}) or {}
                    score = getattr(hit, 'score', 0.0)
                    point_id = getattr(hit, 'id', 'unknown')
                elif isinstance(hit, dict):  # Dictionary format
                    payload = hit.get('payload', {}) or {}
                    score = hit.get('score', 0.0)
                    point_id = hit.get('id', 'unknown')
                else:
                    logger.debug(f"Skipping result with unexpected format: {type(hit)}")
                    continue
                
                # Log the structure for debugging
                logger.debug(f"Processing result - ID: {point_id}, Score: {score}, Payload keys: {list(payload.keys())}")
                
                # Extract content and metadata
                content = ''
                if isinstance(payload, dict):
                    content = str(payload.get('content', '') or '').strip()
                    # If no 'content' key, try to find the first string value
                    if not content:
                        for v in payload.values():
                            if isinstance(v, str) and v.strip():
                                content = v.strip()
                                break
                
                if not content:
                    logger.debug("Skipping result with empty content")
                    continue
                
                # Simple deduplication
                content_hash = hash(content[:200])  # Use first 200 chars for dedupe
                if content_hash in seen_texts:
                    logger.debug("Skipping duplicate content")
                    continue
                    
                seen_texts.add(content_hash)
                
                # Extract metadata (all non-content fields)
                metadata = {}
                if hasattr(payload, 'items'):  # If payload is a dict-like object
                    metadata = {
                        str(k): v 
                        for k, v in payload.items() 
                        if k != 'content' and v is not None
                    }
                
                # Create SearchResult object
                result = SearchResult(
                    id=str(point_id),
                    score=float(score) if score is not None else 0.0,
                    text=content,
                    metadata=metadata
                )
                
                results.append(result)
                logger.debug(f"Added result with score {score:.3f}: {content[:100]}...")
                
                if len(results) >= top_k:
                    break
                    
            except Exception as e:
                logger.error(f"Error processing search result: {str(e)}", exc_info=True)
                continue
        
        logger.info(f"Returning {len(results)} processed results")
        return results

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the current collection.
        
        Returns:
            Dict containing collection metadata including vector count and configuration.
        """
        try:
            collection = self.client.get_collection(self.collection_name)
            
            # Get collection statistics
            stats = self.client.count(
                collection_name=self.collection_name,
                exact=True
            )
            
            # Get a sample of points to check content
            sample_points = []
            try:
                # Scroll through the collection to get sample points
                records, _ = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=3,
                    with_payload=True,
                    with_vectors=False
                )
                sample_points = records
            except Exception as e:
                logger.warning(f"Could not fetch sample points: {str(e)}")
            
            # Build basic info
            info = {
                'collection_name': self.collection_name,
                'points_count': stats.count,
                'storage_path': str(self.storage_path.absolute()),
                'sample_points': []
            }
            
            # Process sample points with proper error handling
            for point in sample_points:
                try:
                    point_info = {
                        'id': str(point.id) if hasattr(point, 'id') else 'unknown',
                        'payload_keys': list(point.payload.keys()) if hasattr(point, 'payload') else [],
                        'has_vector': hasattr(point, 'vector') and bool(point.vector)
                    }
                    info['sample_points'].append(point_info)
                except Exception as e:
                    logger.warning(f"Error processing point: {str(e)}")
                    continue
            
            # Add vector config if available
            if hasattr(collection.config, 'params') and hasattr(collection.config.params, 'vectors'):
                if hasattr(collection.config.params.vectors, 'size'):
                    info['dimensions'] = collection.config.params.vectors.size
                if hasattr(collection.config.params.vectors, 'distance'):
                    info['distance'] = collection.config.params.vectors.distance.name
            
            # Add status if available
            if hasattr(collection, 'status'):
                info['status'] = collection.status
                
            # Add vector count if available
            if hasattr(collection, 'vectors_count'):
                info['vectors_count'] = collection.vectors_count
                
            return info
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}", exc_info=True)
            return {'error': str(e)}


def test_queries():
    """Test the query engine with sample queries."""
    import time
    from typing import List, Dict, Any
    
    def run_test_case(engine: QueryEngine, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Run a single test case and return timing and results."""
        # First print collection info
        if not hasattr(run_test_case, 'collection_info_printed'):
            print("\n" + "="*50)
            print("COLLECTION INFORMATION")
            print("="*50)
            info = engine.get_collection_info()
            for key, value in info.items():
                if key != 'sample_points':
                    print(f"{key}: {value}")
            
            print("\nSample documents:")
            for i, point in enumerate(info.get('sample_points', []), 1):
                print(f"\nDocument {i}:")
                print(f"  ID: {point['id']}")
                print(f"  Payload keys: {', '.join(point['payload_keys'])}")
                print(f"  Has vector: {point['has_vector']}")
            
            run_test_case.collection_info_printed = True
        
        # Run the search
        print("\n" + "="*50)
        print(f"SEARCHING FOR: {query}")
        print("="*50)
        
        start_time = time.time()
        results = engine.search(query, top_k=top_k, score_threshold=0.1)  # Lower threshold for testing
        elapsed_ms = (time.time() - start_time) * 1000
        
        return {
            'query': query,
            'results': results,
            'time_ms': elapsed_ms,
            'result_count': len(results)
        }
    
    def print_test_results(test_result: Dict[str, Any]):
        """Print formatted test results."""
        print(f"\n{'='*100}")
        print(f"QUERY: {test_result['query']}")
        print(f"Time: {test_result['time_ms']:.2f}ms | Results: {test_result['result_count']}")
        print("-" * 100)
        
        if not test_result['results']:
            print("No results found.")
            return
            
        for i, result in enumerate(test_result['results'], 1):
            print(f"\n--- RESULT {i} (Relevance: {result.score:.1%}) ---")
            print(result.format())
    
    print("\n" + "="*50)
    print("UNIVERSITY HANDBOOK RETRIEVAL SYSTEM")
    print("="*50)
    
    try:
        # Initialize the query engine
        print("\nInitializing Query Engine...")
        engine = QueryEngine()
        
        # Display collection info
        try:
            info = engine.get_collection_info()
            print(f"\nConnected to collection: {info.get('name', 'N/A')}")
            print(f"Documents: {info.get('points_count', 0):,}")
            print(f"Dimensions: {info.get('dimensions', 'N/A')}")
        except Exception as e:
            print(f"\nWarning: Could not get collection info: {str(e)}")
        
        # Define test cases
        test_cases = [
            "Admission requirements for Environmental Science higher diploma",
            "List of modules in the first year Environmental Science higher diploma",
            "How to apply for Environmental Science higher diploma",
            "Examination regulations and grading system",
            
            
            
            
            
        ]
        
        # Run test cases
        print(f"\nRunning {len(test_cases)} test queries...\n")
        all_results = []
        
        for query in test_cases:
            result = run_test_case(engine, query)
            all_results.append(result)
            print_test_results(result)
        
        # Print summary
        print("\n" + "="*100)
        print("TEST SUMMARY")
        print("="*100)
        
        avg_time = sum(r['time_ms'] for r in all_results) / len(all_results)
        total_results = sum(r['result_count'] for r in all_results)
        
        print(f"\nTotal Queries: {len(all_results)}")
        print(f"Average Query Time: {avg_time:.2f}ms")
        print(f"Total Results Found: {total_results}")
        
        # Show top performing queries
        print("\nTop Performing Queries (fastest):")
        for i, r in enumerate(sorted(all_results, key=lambda x: x['time_ms'])[:3], 1):
            print(f"{i}. '{r['query']}' - {r['time_ms']:.2f}ms ({r['result_count']} results)")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        return 1
    
    return 0


def interactive_search():
    """Run an interactive search session."""
    print("\n" + "="*50)
    print("UNIVERSITY HANDBOOK SEARCH")
    print("Type 'exit' or 'quit' to end the session")
    print("="*50 + "\n")
    
    try:
        engine = QueryEngine()
        
        while True:
            try:
                query = input("\nEnter your query: ").strip()
                
                if query.lower() in ('exit', 'quit', 'q'):
                    print("\nGoodbye!")
                    break
                    
                if not query:
                    print("Please enter a search query.")
                    continue
                
                start_time = time.time()
                results = engine.search(query, top_k=3)
                elapsed_ms = (time.time() - start_time) * 1000
                
                print(f"\nFound {len(results)} results in {elapsed_ms:.2f}ms")
                
                if not results:
                    print("No relevant information found. Try rephrasing your query.")
                    continue
                
                for i, result in enumerate(results, 1):
                    print(f"\n--- RESULT {i} (Relevance: {result.score:.1%}) ---")
                    print(result.format())
                
                print("\n" + "-"*50)
                print("Tip: Try being more specific or use different keywords for better results.")
                
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                continue
                
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="University Handbook Search System")
    parser.add_argument(
        "--interactive", "-i", 
        action="store_true",
        help="Run in interactive mode"
    )
    args = parser.parse_args()
    
    if args.interactive:
        sys.exit(interactive_search())
    else:
        sys.exit(test_queries())