import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Generator
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from .config import (
    DEFAULT_COLLECTION,
    DEFAULT_MODEL,
    VECTOR_SIZE,
    DEFAULT_BATCH_SIZE,
    get_qdrant_config
)
from .qdrant_singleton import QdrantClientSingleton

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextEmbedder:
    """
    A class to handle text embedding generation and storage using Sentence Transformers and Qdrant.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        collection_name: str = DEFAULT_COLLECTION,
        storage_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the TextEmbedder.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            collection_name: Name of the Qdrant collection
            storage_path: Path to store Qdrant data. If None, uses in-memory storage.
        """
        self.model = SentenceTransformer(model_name)
        self.collection_name = collection_name
        self.vector_size = VECTOR_SIZE
        
        # Initialize Qdrant client
        self.client = QdrantClientSingleton(storage_path).get_client()
        
        # Create collection if it doesn't exist
        self._initialize_collection()
    
    def _initialize_collection(self) -> None:
        """Initialize the Qdrant collection if it doesn't exist."""
        try:
            collections = self.client.get_collections()
            collection_names = [collection.name for collection in collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created new collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            raise
    
    def _chunk_text(self, text: str, max_length: int = 500, min_chunk_size: int = 100) -> List[str]:
        """
        Split text into meaningful chunks while preserving sentence boundaries.
        
        Args:
            text: Input text to be chunked
            max_length: Maximum number of words per chunk
            min_chunk_size: Minimum number of words per chunk (except possibly the last one)
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
            
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # First, try to split by paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        
        for para in paragraphs:
            words = para.split()
            
            # If paragraph is small enough, add as is
            if len(words) <= max_length:
                chunks.append(para)
                continue
                
            # Otherwise, try to split at sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                sentence_words = sentence.split()
                sentence_length = len(sentence_words)
                
                # If sentence is too long, we need to split it
                if sentence_length > max_length:
                    # Flush current chunk if not empty
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = []
                        current_length = 0
                    
                    # Split the long sentence into chunks
                    for i in range(0, len(sentence_words), max_length):
                        chunk = ' '.join(sentence_words[i:i + max_length])
                        chunks.append(chunk)
                    continue
                    
                # If adding this sentence would make the chunk too long, start a new chunk
                if current_length + sentence_length > max_length and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                    
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_length += sentence_length
            
            # Add the last chunk if it meets minimum size or is the only chunk
            if current_chunk and (current_length >= min_chunk_size or not chunks):
                chunks.append(' '.join(current_chunk))
        
        # If we still have chunks that are too small, merge them
        if len(chunks) > 1:
            merged_chunks = []
            current = chunks[0].split()
            
            for chunk in chunks[1:]:
                chunk_words = chunk.split()
                # If current chunk is small, try to merge with next
                if len(current) < min_chunk_size and len(current) + len(chunk_words) <= max_length:
                    current.extend(chunk_words)
                else:
                    merged_chunks.append(' '.join(current))
                    current = chunk_words
            
            # Add the last chunk
            if current:
                merged_chunks.append(' '.join(current))
                
            chunks = merged_chunks
            
        return chunks

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
            
        try:
            # Filter out empty strings and normalize whitespace
            texts = [' '.join(t.split()) for t in texts if t and t.strip()]
            if not texts:
                return []
                
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 10,
                normalize_embeddings=True
            )
            return [embedding.tolist() for embedding in embeddings]
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def _prepare_documents(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = "text",
        metadata_field: str = "metadata"
    ) -> List[Dict[str, Any]]:
        """Prepare and chunk documents for embedding."""
        prepared_docs = []
        
        for doc_idx, doc in enumerate(documents):
            if not isinstance(doc, dict):
                logger.warning(f"Skipping non-dict document at index {doc_idx}")
                continue
                
            text = doc.get(text_field, '').strip()
            if not text:
                logger.warning(f"Document {doc_idx} has no text content")
                continue
                
            # Get base metadata
            metadata = doc.get(metadata_field, {})
            if not isinstance(metadata, dict):
                metadata = {}
                
            # Add document-level metadata
            metadata.update({
                'doc_id': metadata.get('doc_id', doc_idx),
                'doc_type': metadata.get('doc_type', 'document'),
                'timestamp': metadata.get('timestamp', str(datetime.utcnow()))
            })
            
            # Chunk the text
            chunks = self._chunk_text(text)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_id': chunk_idx,
                    'total_chunks': len(chunks),
                    'is_chunk': len(chunks) > 1
                })
                
                prepared_docs.append({
                    'text': chunk,
                    'metadata': chunk_metadata
                })
                
        return prepared_docs

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = DEFAULT_BATCH_SIZE,
        text_field: str = "text",
        metadata_field: str = "metadata",
        chunk_size: int = 500
    ) -> Dict[str, int]:
        """
        Add documents to the vector store with embedding and chunking.
        
        Args:
            documents: List of document dictionaries
            batch_size: Number of documents to process in each batch
            text_field: Key in document containing the text to embed
            metadata_field: Key in document containing metadata
            chunk_size: Maximum number of words per chunk
            
        Returns:
            Dictionary with statistics about the operation
        """
        if not documents:
            logger.warning("No documents provided to add")
            return {"total_documents": 0, "total_chunks": 0, "batches_processed": 0}
            
        # Prepare all documents with chunking
        prepared_docs = self._prepare_documents(
            documents,
            text_field=text_field,
            metadata_field=metadata_field
        )
        
        total_chunks = len(prepared_docs)
        if not total_chunks:
            logger.warning("No valid text chunks found in documents")
            return {"total_documents": 0, "total_chunks": 0, "batches_processed": 0}
            
        logger.info(f"Processing {len(documents)} documents into {total_chunks} chunks in batches of {batch_size}")
        
        # Process in batches
        successful_batches = 0
        for i in range(0, total_chunks, batch_size):
            batch = prepared_docs[i:i + batch_size]
            batch_texts = [doc['text'] for doc in batch]
            batch_metadatas = [doc['metadata'] for doc in batch]
            
            try:
                # Generate embeddings for the batch
                embeddings = self.embed_texts(batch_texts)
                if not embeddings:
                    logger.warning(f"No embeddings generated for batch {i//batch_size + 1}")
                    continue
                    
                # Prepare points for Qdrant
                points = []
                for idx, (text, vector, metadata) in enumerate(zip(batch_texts, embeddings, batch_metadatas)):
                    point_id = i + idx
                    points.append(
                        PointStruct(
                            id=point_id,
                            vector=vector,
                            payload={
                                'text': text,
                                'metadata': metadata or {}
                            }
                        )
                    )
                
                # Store in Qdrant
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True
                )
                
                successful_batches += 1
                logger.info(
                    f"Processed batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size} "
                    f"({len(points)} chunks)"
                )
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                continue
                
        return {
            'total_documents': len(documents),
            'total_chunks': total_chunks,
            'batches_processed': successful_batches
        }
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_condition: Optional[Dict] = None,
        score_threshold: Optional[float] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents to the query with enhanced filtering and scoring.
        
        Args:
            query: The search query text
            top_k: Maximum number of results to return
            filter_condition: Optional filter conditions (e.g., {"doc_type": "section"})
            score_threshold: Minimum similarity score (0-1) for results
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of search results with scores and metadata
        """
        if not query or not query.strip():
            logger.warning("Empty search query provided")
            return []
            
        try:
            # Generate query embedding
            query_embedding = self.embed_texts([query.strip()])
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []
                
            query_embedding = query_embedding[0]
            
            # Prepare filter conditions
            must_conditions = []
            if filter_condition:
                for field, value in filter_condition.items():
                    if isinstance(value, (list, tuple, set)):
                        must_conditions.append(
                            FieldCondition(
                                key=f"metadata.{field}",
                                match=MatchValue(value=value)
                            )
                        )
                    else:
                        must_conditions.append(
                            FieldCondition(
                                key=f"metadata.{field}",
                                match=MatchValue(value=value)
                            )
                        )
            
            # Build the query filter
            query_filter = Filter(must=must_conditions) if must_conditions else None
            
            # Execute search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=min(top_k * 2, 100),  # Get more results for post-filtering
                query_filter=query_filter,
                score_threshold=score_threshold or 0.0
            )
            
            # Process and deduplicate results
            seen_docs = set()
            results = []
            
            for hit in search_results:
                doc_id = hit.payload.get('metadata', {}).get('doc_id')
                chunk_id = hit.payload.get('metadata', {}).get('chunk_id')
                
                # Skip if we've seen this document (unless it's a different chunk)
                if doc_id is not None and chunk_id is not None:
                    doc_key = f"{doc_id}_{chunk_id}"
                    if doc_key in seen_docs:
                        continue
                    seen_docs.add(doc_key)
                
                result = {
                    'id': hit.id,
                    'score': float(hit.score),
                    'text': hit.payload.get('text', '')
                }
                
                if include_metadata:
                    result['metadata'] = hit.payload.get('metadata', {})
                
                results.append(result)
                
                # Stop if we have enough results
                if len(results) >= top_k:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}", exc_info=True)
            return []
    
    def delete_collection(self) -> bool:
        """Delete the current collection."""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False


def load_document_chunks(chunks_path: str) -> list[dict]:
    """
    Load document chunks from a JSON file.
    
    Args:
        chunks_path: Path to the JSON file containing document chunks
        
    Returns:
        List of document chunks in the format expected by TextEmbedder
    """
    import json
    import logging
    from pathlib import Path
    from typing import Union, List, Dict, Any
    
    logger = logging.getLogger(__name__)
    
    chunks_path = Path(chunks_path)
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found at {chunks_path}")
    
    logger.info(f"Loading chunks from {chunks_path}")
    with open(chunks_path, 'r', encoding='utf-8') as f:
        try:
            chunks_data = json.load(f)
            if not isinstance(chunks_data, list):
                logger.warning(f"Expected a list of chunks, got {type(chunks_data)}. Wrapping in a list.")
                chunks_data = [chunks_data]  # In case the root is a single chunk
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {chunks_path}: {e}")
            raise
    
    if not chunks_data:
        logger.warning("No chunks found in the input file")
        return []
    
    # Log sample of the first chunk for debugging
    logger.info(f"First chunk sample: {json.dumps(chunks_data[0], indent=2)[:500]}...")
    
    # Convert chunks to the format expected by TextEmbedder
    documents = []
    empty_chunks = 0
    
    for i, chunk in enumerate(chunks_data):
        try:
            # Handle Qdrant export format (has 'payload' with 'content')
            if isinstance(chunk, dict) and 'payload' in chunk and isinstance(chunk['payload'], dict):
                payload = chunk['payload']
                text = payload.get('content', '')
                metadata = {k: v for k, v in payload.items() if k != 'content'}
                
                # Add any additional metadata from the root level
                metadata.update({k: v for k, v in chunk.items() if k not in ['payload', 'vector']})
                
                # Ensure required metadata fields exist
                metadata.setdefault('source', 'unknown')
                metadata.setdefault('page', 0)
                metadata['chunk_id'] = metadata.get('chunk_id', i)
                metadata['total_chunks'] = metadata.get('total_chunks', len(chunks_data))
                
            # Handle standard chunk format
            elif isinstance(chunk, dict):
                # If chunk is a dict, try to extract text and metadata
                text = chunk.get('text', '')
                if not text and 'content' in chunk:
                    text = chunk['content']  # Some chunkers use 'content' instead of 'text'
                
                # Extract metadata
                metadata = chunk.get('metadata', {})
                if not metadata and any(k != 'text' and k != 'content' for k in chunk.keys()):
                    # If no explicit metadata but has other keys, use them as metadata
                    metadata = {k: v for k, v in chunk.items() if k not in ['text', 'content']}
                
                # Ensure required metadata fields exist
                metadata.setdefault('source', chunk.get('source', 'unknown'))
                metadata.setdefault('page', chunk.get('page', 0))
                metadata['chunk_id'] = chunk.get('chunk_id', i)
                metadata['total_chunks'] = chunk.get('total_chunks', len(chunks_data))
                
            # Handle string chunks
            elif isinstance(chunk, str):
                text = chunk
                metadata = {
                    'source': 'unknown',
                    'page': 0,
                    'chunk_id': i,
                    'total_chunks': len(chunks_data)
                }
            else:
                logger.warning(f"Unexpected chunk type {type(chunk)} at index {i}")
                continue
            
            # Clean and validate text
            if not text or not isinstance(text, str) or not text.strip():
                empty_chunks += 1
                logger.debug(f"Empty or invalid text in chunk {i}")
                continue
                
            documents.append({
                "text": text.strip(),
                "metadata": metadata
            })
            
        except Exception as e:
            logger.error(f"Error processing chunk {i}: {e}", exc_info=True)
            continue
    
    if empty_chunks > 0:
        logger.warning(f"Skipped {empty_chunks} chunks with empty or invalid text")
    
    logger.info(f"Successfully loaded {len(documents)} valid chunks out of {len(chunks_data)} total")
    return documents

def main():
    """
    Main function to demonstrate TextEmbedder with actual document chunks.
    """
    import time
    from pathlib import Path
    
    # Path to the chunks file
    project_root = Path(__file__).parent.parent.parent
    chunks_path = project_root / "data" / "processed" / "chunks" / "handbook_chunks.json"
    
    try:
        # Load actual document chunks
        print(f"Loading document chunks from {chunks_path}...")
        documents = load_document_chunks(chunks_path)
        print(f"Loaded {len(documents)} document chunks")
        
        # Initialize the embedder with a collection for the handbook
        print("\nInitializing TextEmbedder...")
        storage_path = "./qdrant_handbook"
        collection_name = "handbook_chunks"
        
        # First, create a client to clean up any existing collection
        try:
            temp_client = QdrantClient(path=storage_path)
            collections = temp_client.get_collections()
            if collection_name in [c.name for c in collections.collections]:
                print(f"Cleaning up existing collection: {collection_name}")
                temp_client.delete_collection(collection_name=collection_name)
        except Exception as e:
            print(f"Warning: Could not clean up existing collection: {e}")
        finally:
            if 'temp_client' in locals():
                temp_client.close()
        
        # Now initialize our embedder
        embedder = TextEmbedder(
            model_name="all-MiniLM-L6-v2",
            collection_name=collection_name,
            storage_path=storage_path
        )
        
        try:
            # Add documents
            print("\nAdding documents to the vector store...")
            result = embedder.add_documents(documents)
            print(f"Added {result['total_documents']} documents as {result['total_chunks']} chunks")
            
            # Give Qdrant a moment to index
            print("\nWaiting for indexing to complete...")
            time.sleep(2)
            
            # Example searches
            test_queries = [
                "Subjects belongs to applied mathamatics and computing",
                "Bio Sience department information",
                "What are the subjects in IT degree",
                "What are the subjects in Bio Sience degree"
            ]
            
            for query in test_queries:
                print(f"\n{'='*80}")
                print(f"SEARCH RESULTS FOR: '{query}'")
                print(f"{'='*80}")
                
                # Basic search
                print("\nBasic search results:")
                try:
                    results = embedder.search(query, top_k=2)
                    if not results:
                        print("No results found")
                    else:
                        for i, result in enumerate(results, 1):
                            print(f"\nResult {i} (Score: {result['score']:.4f}):")
                            print(f"Source: {result['metadata'].get('source', 'N/A')}")
                            print(f"Page: {result['metadata'].get('page', 'N/A')}")
                            print(f"Text: {result['text']}")
                except Exception as e:
                    print(f"Error during search: {e}")
            
            # Show metadata filtering
            print("\n" + "="*80)
            print("METADATA FILTERING EXAMPLE")
            print("="*80)
            print("\nSearching for chunks from a specific source:")
            try:
                # Get the first source for demonstration
                sample_source = documents[0]['metadata'].get('source', '') if documents else ""
                if sample_source:
                    results = embedder.search(
                        "",  # Empty query to get any matching documents
                        filter_condition={"source": sample_source},
                        top_k=2
                    )
                    if not results:
                        print(f"No results found for source: {sample_source}")
                    else:
                        print(f"\nFound {len(results)} chunks from source: {sample_source}")
                        for i, result in enumerate(results, 1):
                            print(f"\nResult {i} (Score: {result['score']:.4f}):")
                            print(f"Source: {result['metadata'].get('source', 'N/A')}")
                            print(f"Page: {result['metadata'].get('page', 'N/A')}")
                            print(f"Text: {result['text']}")
                else:
                    print("No source information available in the chunks")
            except Exception as e:
                print(f"Error during filtered search: {e}")
        
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Clean up
            user_input = input("\nDo you want to delete the collection? (y/n): ")
            if user_input.lower() == 'y':
                print("Cleaning up collection...")
                embedder.delete_collection()
            else:
                print(f"\nCollection '{collection_name}' was not deleted.")
                print(f"You can access it later with collection name: {collection_name}")
                print(f"Storage path: {storage_path}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up (comment this out if you want to keep the collection)
        user_input = input("\nDo you want to delete the test collection? (y/n): ")
        if user_input.lower() == 'y':
            print("Cleaning up test collection...")
            embedder.delete_collection()
        else:
            print(f"Test collection '{collection_name}' was not deleted.")
            print(f"You can access it later with collection name: {collection_name}")
            print(f"Storage path: {storage_path}")


if __name__ == "__main__":
    main()