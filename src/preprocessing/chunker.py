import json
import logging
import time
from typing import Dict, List, Optional, Any, Union, Generator
import os
import re
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """
    Represents a chunk of document content with metadata for vector storage.
    """
    id: str
    content: str
    metadata: Dict[str, Any]

class HandbookChunker:
    """
    Chunks the handbook JSON data into smaller pieces for vector storage.
    
    The chunker processes the hierarchical structure of the handbook and creates
    meaningful chunks of text with appropriate metadata for retrieval.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, *, skip_empty_sections: bool = True):
        """
        Initialize the chunker with chunking parameters.
        
        Args:
            chunk_size: Maximum size of each chunk in characters (500-2000)
            chunk_overlap: Number of characters to overlap between chunks (10-25% of chunk_size)
            skip_empty_sections: Whether to skip sections with empty content
        """
        # Validate parameters
        if not 500 <= chunk_size <= 2000:
            raise ValueError("chunk_size should be between 500 and 2000")
        if not (0 <= chunk_overlap <= chunk_size // 2):
            raise ValueError("chunk_overlap should be between 0 and half of chunk_size")
            
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, chunk_size // 2)
        self.skip_empty_sections = skip_empty_sections
        self.max_iterations = 10000  # Safety limit for chunking loop
        self.skip_empty_sections = skip_empty_sections
    
    def load_json(self, file_path: str) -> List[Dict]:
        """
        Load JSON data from a file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of document entries
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Expected a JSON array of document entries")
            return data
    
    def create_chunks(self, data: List[Dict]) -> List[DocumentChunk]:
        """
        Create chunks from the input data.
        
        Args:
            data: List of document entries
            
        Returns:
            List of document chunks
        """
        chunks = []
        
        # Process each document entry
        for entry in tqdm(data, desc="Processing entries"):
            try:
                # Skip if the entry is empty or has no content
                if not entry or 'content' not in entry or not entry['content'].strip():
                    if self.skip_empty_sections:
                        continue
                    
                # Create metadata dictionary
                metadata = {
                    'section_number': entry.get('section_number', ''),
                    'section_name': entry.get('section_name', ''),
                    'title': entry.get('title', ''),
                    'subtitle': entry.get('subtitle', ''),
                    'page': str(entry.get('page', '')),
                    'source': 'uov_handbook',
                    'chunk_id': str(uuid.uuid4())
                }
                
                # Clean and prepare content
                content = entry['content'].strip()
                
                # Add title and subtitle to content for better context
                if metadata['title']:
                    content = f"{metadata['title']}\n" + content
                if metadata['subtitle']:
                    content = f"{metadata['subtitle']}\n" + content
                
                # Create chunk
                chunk = DocumentChunk(
                    id=metadata['chunk_id'],
                    content=content,
                    metadata=metadata
                )
                
                chunks.append(chunk)
                
            except Exception as e:
                logger.error(f"Error processing entry: {str(e)}")
                continue
                
        return chunks
    
    def chunks_to_qdrant_format(self, chunks: List[DocumentChunk]) -> List[Dict]:
        """
        Convert document chunks to Qdrant point format.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            List of dictionaries in Qdrant point format
        """
        points = []
        for chunk in chunks:
            point = {
                'id': chunk.id,
                'vector': None,  # Will be filled in by the vectorizer
                'payload': {
                    'content': chunk.content,
                    **chunk.metadata
                }
            }
            points.append(point)
        return points
        
    def save_chunks_to_jsonl(self, chunks: List[DocumentChunk], output_path: str) -> None:
        """
        Save chunks to a JSONL file.
        
        Args:
            chunks: List of document chunks
            output_path: Path to save the JSONL file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                # Convert chunk to dict and ensure all values are JSON-serializable
                chunk_dict = {
                    'id': chunk.id,
                    'content': chunk.content,
                    'metadata': chunk.metadata
                }
                f.write(json.dumps(chunk_dict, ensure_ascii=False) + '\n')


def process_handbook(input_path: str, output_dir: str = None) -> List[Dict]:
    """
    Process a handbook JSON file and prepare it for vector storage.
    
    Args:
        input_path: Path to the input JSON file
        output_dir: Directory to save the processed output (default: 'data/chunks')
        
    Returns:
        List of document chunks in Qdrant point format
        
    Raises:
        FileNotFoundError: If the input file doesn't exist
        json.JSONDecodeError: If the input file is not valid JSON
        Exception: For other processing errors
    """
    start_time = time.time()
    logger.info(f"Starting to process handbook: {input_path}")
    
    try:
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(input_path))), 'chunks')
        
        # Validate input file
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        # Initialize chunker with default parameters
        chunker = HandbookChunker()
        
        # Load the data
        logger.info("Loading JSON data...")
        data = chunker.load_json(str(input_path))
        
        if not data:
            logger.warning("No data found in the input file")
            return []
            
        logger.info(f"Found {len(data)} entries to process")
        
        # Process the data
        logger.info("Starting chunking process...")
        chunks = chunker.create_chunks(data)
        
        if not chunks:
            logger.warning("No chunks were created from the input data")
            return []
            
        logger.info(f"Created {len(chunks)} chunks in total")
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save chunks in JSONL format
        output_path = output_dir / 'handbook_chunks.jsonl'
        chunker.save_chunks_to_jsonl(chunks, str(output_path))
        
        # Also save in Qdrant point format for compatibility
        qdrant_points = chunker.chunks_to_qdrant_format(chunks)
        qdrant_output_path = output_dir / 'qdrant_points.jsonl'
        with open(qdrant_output_path, 'w', encoding='utf-8') as f:
            for point in qdrant_points:
                f.write(json.dumps(point, ensure_ascii=False) + '\n')
        
        # Log completion
        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")
        logger.info(f"Saved Qdrant-compatible points to {qdrant_output_path}")
        
        return qdrant_points
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {input_path}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error processing handbook: {str(e)}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process handbook JSON into chunks for vector storage.')
    parser.add_argument('input_file', type=str, help='Path to the input JSON file')
    parser.add_argument('--output-dir', type=str, default=None, 
                       help='Directory to save the output files (default: data/chunks)')
    
    args = parser.parse_args()
    process_handbook(args.input_file, args.output_dir)
