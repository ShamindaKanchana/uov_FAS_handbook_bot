#!/usr/bin/env python3
"""
Script to process handbook chunks, generate embeddings, and store them in Qdrant.
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct, Distance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # Efficient model for semantic search
VECTOR_SIZE = 384  # Dimension of the embeddings
BATCH_SIZE = 32    # Batch size for embedding generation
COLLECTION_NAME = "handbook_chunks"

class HandbookEmbedder:
    """Class to handle embedding generation and storage for handbook chunks."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL, collection_name: str = COLLECTION_NAME):
        """Initialize the embedder with model and collection name."""
        self.model = SentenceTransformer(model_name)
        self.collection_name = collection_name
        self.vector_size = VECTOR_SIZE
        
        # Initialize Qdrant client (using local SQLite for persistence)
        self.qdrant = QdrantClient(path="qdrant_handbook")
        
        # Create collection if it doesn't exist
        self._setup_collection()
    
    def _setup_collection(self):
        """Create the Qdrant collection if it doesn't exist."""
        collections = self.qdrant.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        
        if self.collection_name not in collection_names:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def load_chunks(self, file_path: str) -> List[Dict[str, Any]]:
        """Load chunks from a JSONL file."""
        chunks = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        return chunks
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for a list of text chunks."""
        # Extract text content for embedding
        texts = [chunk['payload']['content'] for chunk in chunks]
        
        # Generate embeddings in batches
        embeddings = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Generating embeddings"):
            batch_texts = texts[i:i+BATCH_SIZE]
            batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['vector'] = embedding.tolist()
        
        return chunks
    
    def store_in_qdrant(self, chunks: List[Dict[str, Any]]):
        """Store chunks with embeddings in Qdrant."""
        points = []
        for chunk in tqdm(chunks, desc="Preparing points for Qdrant"):
            if 'vector' not in chunk:
                logger.warning(f"Skipping chunk without vector: {chunk.get('id', 'unknown')}")
                continue
                
            point = PointStruct(
                id=chunk['id'],
                vector=chunk['vector'],
                payload=chunk['payload']
            )
            points.append(point)
        
        # Upload to Qdrant
        if points:
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Stored {len(points)} vectors in Qdrant collection '{self.collection_name}'")

def main():
    """Main function to process handbook chunks and store embeddings."""
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    chunks_path = project_root / "data" / "chunks" / "qdrant_points.jsonl"
    
    if not chunks_path.exists():
        logger.error(f"Chunks file not found at {chunks_path}")
        return
    
    try:
        # Initialize embedder
        embedder = HandbookEmbedder()
        
        # Load chunks
        logger.info(f"Loading chunks from {chunks_path}")
        chunks = embedder.load_chunks(chunks_path)
        logger.info(f"Loaded {len(chunks)} chunks")
        
        # Generate and store embeddings
        logger.info("Generating embeddings...")
        chunks_with_embeddings = embedder.generate_embeddings(chunks)
        
        # Store in Qdrant
        logger.info(f"Storing embeddings in Qdrant collection '{COLLECTION_NAME}'...")
        embedder.store_in_qdrant(chunks_with_embeddings)
        
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error processing handbook embeddings: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
