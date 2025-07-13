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
        chunk_size: int = 500,
        skip_chunking: bool = False
    ) -> Dict[str, int]:
        """
        Add documents to the vector store with optional chunking.
        
        Args:
            documents: List of document dictionaries
            batch_size: Number of documents to process in each batch
            text_field: Key in document containing the text to embed
            metadata_field: Key in document containing metadata
            chunk_size: Maximum number of words per chunk (used when skip_chunking=False)
            skip_chunking: If True, assumes documents are already properly chunked
            
        Returns:
            Dictionary with statistics about the operation
        """
        if not documents:
            logger.warning("No documents provided to add")
            return {"total_documents": 0, "total_chunks": 0, "batches_processed": 0}
            
        # Prepare documents with or without chunking
        if skip_chunking:
            logger.info("Using pre-chunked documents (chunking skipped)")
            prepared_docs = []
            for doc in documents:
                if not isinstance(doc, dict):
                    logger.warning(f"Skipping non-dict document: {doc}")
                    continue
                    
                # Extract text and metadata
                text = doc.get('text', doc.get('content', '')).strip()
                if not text:
                    logger.warning("Document has no text content")
                    continue
                    
                # Get or create metadata
                metadata = doc.get('metadata', {})
                if not isinstance(metadata, dict):
                    metadata = {}
                    
                # Preserve any top-level fields as metadata
                metadata.update({
                    k: v for k, v in doc.items() 
                    if k not in ['text', 'content', 'metadata'] and v is not None
                })
                
                prepared_docs.append({
                    'text': text,
                    'metadata': metadata
                })
        else:
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
    Load document chunks from a JSON or JSONL file, with special handling for Qdrant export format.
    
    Args:
        chunks_path: Path to the JSON/JSONL file containing document chunks
        
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
    
    # Read the file content first to detect format
    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
        # Try to parse as JSON array first
        if content.startswith('[') and content.endswith(']'):
            try:
                chunks_data = json.loads(content)
                if not isinstance(chunks_data, list):
                    chunks_data = [chunks_data]
            except json.JSONDecodeError:
                # Fall through to JSONL parsing
                chunks_data = [json.loads(line) for line in content.splitlines() if line.strip()]
        else:
            # Parse as JSONL
            chunks_data = [json.loads(line) for line in content.splitlines() if line.strip()]
            
    except Exception as e:
        logger.error(f"Error parsing file {chunks_path}: {e}")
        raise
    
    if not chunks_data:
        logger.warning("No chunks found in the input file")
        return []
    
    # Log sample of the first chunk for debugging
    sample = json.dumps(chunks_data[0], indent=2, ensure_ascii=False)
    logger.info(f"First chunk sample: {sample[:500]}...")
    
    # Convert chunks to the format expected by TextEmbedder
    documents = []
    empty_chunks = 0
    
    for i, chunk in enumerate(chunks_data):
        try:
            # Skip None or empty chunks
            if not chunk:
                empty_chunks += 1
                continue
                
            # Handle Qdrant export format (has 'payload' with 'content' or 'text')
            if isinstance(chunk, dict) and 'payload' in chunk and isinstance(chunk['payload'], dict):
                payload = chunk['payload']
                # Try both 'content' and 'text' fields
                text = payload.get('content', payload.get('text', '')).strip()
                
                # Extract metadata from payload first, excluding content/text
                metadata = {k: v for k, v in payload.items() 
                          if k not in ['content', 'text'] and v is not None}
                
                # Add any additional metadata from the root level
                metadata.update({k: v for k, v in chunk.items() 
                              if k not in ['payload', 'vector'] and v is not None})
                
                # Ensure required metadata fields exist with sensible defaults
                metadata.setdefault('source', metadata.get('source', 'unknown'))
                metadata.setdefault('page', metadata.get('page', 0))
                metadata['chunk_id'] = metadata.get('chunk_id', i)
                metadata['total_chunks'] = metadata.get('total_chunks', len(chunks_data))
                
            # Handle standard chunk format (either direct or with metadata field)
            elif isinstance(chunk, dict):
                # Extract text from either 'text' or 'content' field
                text = chunk.get('text', chunk.get('content', '')).strip()
                
                # Extract metadata from 'metadata' field or use the chunk itself
                metadata = chunk.get('metadata', {})
                if not isinstance(metadata, dict):
                    metadata = {}
                    
                # Add any non-standard fields as metadata (except text/content)
                metadata.update({k: v for k, v in chunk.items() 
                              if k not in ['text', 'content', 'metadata'] 
                              and v is not None})
                
                # Set default values for required fields
                metadata.setdefault('source', 'unknown')
                metadata.setdefault('page', 0)
                metadata['chunk_id'] = chunk.get('chunk_id', i)
                metadata['total_chunks'] = chunk.get('total_chunks', len(chunks_data))
                
            # Handle string chunks (should be rare with Qdrant exports)
            elif isinstance(chunk, str):
                text = chunk.strip()
                metadata = {
                    'source': 'unknown',
                    'page': 0,
                    'chunk_id': i,
                    'total_chunks': len(chunks_data)
                }
            else:
                logger.warning(f"Unexpected chunk type {type(chunk)} at index {i}")
                continue
            
            # Skip empty text chunks
            if not text:
                empty_chunks += 1
                logger.debug(f"Empty text in chunk {i}")
                continue
                
            documents.append({
                "text": text,
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
    Main function to process document chunks and create embeddings.
    """
    from pathlib import Path
    
    # Path to the chunks file
    project_root = Path(__file__).parent.parent.parent
    chunks_path = project_root / "data" / "chunks" / "qdrant_points.jsonl"
    
    try:
        # Load document chunks
        print(f"Loading document chunks from {chunks_path}...")
        documents = load_document_chunks(chunks_path)
        print(f"Loaded {len(documents)} document chunks")
        
        # Initialize the embedder
        print("\nInitializing TextEmbedder...")
        storage_path = "./qdrant_handbook"
        collection_name = "handbook_chunks"
        
        # Clean up any existing collection
        try:
            temp_client = QdrantClient(path=storage_path)
            collections = temp_client.get_collections()
            if collection_name in [c.name for c in collections.collections]:
                print(f"Removing existing collection: {collection_name}")
                temp_client.delete_collection(collection_name=collection_name)
        except Exception as e:
            print(f"Warning: Could not clean up existing collection: {e}")
        finally:
            if 'temp_client' in locals():
                temp_client.close()
        
        # Initialize the embedder
        embedder = TextEmbedder(
            model_name="all-MiniLM-L6-v2",
            collection_name=collection_name,
            storage_path=storage_path
        )
        
        # Add documents with skip_chunking=True for pre-chunked data
        print("\nAdding documents to the vector store...")
        result = embedder.add_documents(
            documents,
            skip_chunking=True  # Skip chunking since data is already chunked
        )
        
        print(f"\nSuccessfully processed {result['total_chunks']} chunks")
        print(f"Collection '{collection_name}' is ready for use")
        print(f"Storage location: {Path(storage_path).resolve()}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())