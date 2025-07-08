import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
import os
import re
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path

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
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the chunker with chunking parameters.
        
        Args:
            chunk_size: Maximum size of each chunk in characters (500-2000)
            chunk_overlap: Number of characters to overlap between chunks (10-25% of chunk_size)
        """
        # Validate parameters
        if not 500 <= chunk_size <= 2000:
            raise ValueError("chunk_size should be between 500 and 2000")
        if not (0 <= chunk_overlap <= chunk_size // 2):
            raise ValueError("chunk_overlap should be between 0 and half of chunk_size")
            
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, chunk_size // 2)
        self.max_iterations = 10000  # Safety limit for chunking loop
    
    def load_json(self, file_path: str) -> Dict:
        """Load JSON data from a file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_chunks(self, data: Dict) -> List[DocumentChunk]:
        """
        Create chunks from the handbook data.
        
        Args:
            data: The loaded JSON data from the handbook
            
        Returns:
            List of document chunks ready for vector storage
        """
        chunks = []
        
        # Process each section in the handbook
        for section in data.get('sections', []):
            section_chunks = self._process_section(section)
            chunks.extend(section_chunks)
            
        return chunks
    
    def _process_section(self, section: Dict, parent_metadata: Optional[Dict] = None) -> List[DocumentChunk]:
        """
        Process a section and its subsections recursively.
        
        Args:
            section: The section to process
            parent_metadata: Metadata from parent sections
            
        Returns:
            List of document chunks from this section and its children
        """
        if parent_metadata is None:
            parent_metadata = {}
            
        # Create metadata for this section
        metadata = {
            'section_number': section.get('section_number', ''),
            'title': section.get('title', ''),
            'page_start': section.get('page_start', 0),
            'source': 'handbook',
            **parent_metadata
        }
        
        chunks = []
        
        # Process the section's own content
        content = section.get('content', '').strip()
        if content:
            # Create chunks from the content
            content_chunks = self._split_into_chunks(content, metadata)
            chunks.extend(content_chunks)
        
        # Process subsections
        for subsection in section.get('subsections', []):
            subsection_chunks = self._process_section(subsection, metadata)
            chunks.extend(subsection_chunks)
            
        return chunks
    
    def _find_meaningful_boundary(self, text: str, start: int, max_end: int) -> int:
        """
        Find the most meaningful boundary point for chunking.
        Prioritizes sentence endings, paragraph breaks, and other natural boundaries.
        
        Args:
            text: The full text
            start: Start position of current chunk
            max_end: Maximum end position (chunk_size)
            
        Returns:
            Position to end the current chunk
        """
        # If we're near the end of the text, just return the end
        if max_end >= len(text) - 1:
            return len(text)
            
        # Look for paragraph breaks first (double newlines)
        para_break = text.rfind('\n\n', start, max_end)
        if para_break > start and (max_end - para_break) < 100:  # If close to a paragraph break
            return para_break + 2
            
        # Look for sentence boundaries in the last 40% of the chunk
        look_behind = max(start, max_end - int(self.chunk_size * 0.4))
        
        # Try different boundary markers in order of preference
        boundaries = [
            ('. ', 2),    # Sentence end with space
            ('! ', 2),    # Exclamation with space
            ('? ', 2),    # Question with space
            ('.\n', 2),  # Sentence at end of line
            ('; ', 2),    # Semicolon with space
            (': ', 2),    # Colon with space
            ('\n', 1),   # Single newline
            (', ', 2),    # Comma with space
            (' ', 1)      # Space as last resort
        ]
        
        for boundary, offset in boundaries:
            boundary_pos = text.rfind(boundary, look_behind, max_end)
            # Ensure we don't create chunks that are too small
            if boundary_pos > start + (self.chunk_size // 2):
                # Check if this is a false positive (e.g., abbreviations)
                if boundary in ['. ', '! ', '? ']:
                    # Skip common abbreviations
                    prev_word = text[max(0, boundary_pos-10):boundary_pos].split()[-1] if boundary_pos > 10 else ''
                    if any(prev_word.lower().endswith(abbr) for abbr in ['dr', 'mr', 'mrs', 'ms', 'phd', 'etc', 'fig', 'no']):
                        continue
                return boundary_pos + offset
        
        # If no good boundary found, try to find a space near the max_end
        space_pos = text.rfind(' ', max(start, max_end - 50), max_end)
        if space_pos > start:
            return space_pos + 1
            
        # If we can't find a good boundary, return the max_end
        return max_end
    
    def _clean_chunk_text(self, text: str) -> str:
        """Clean and format chunk text."""
        # Replace multiple spaces/newlines with a single space
        text = ' '.join(text.split())
        # Ensure proper spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,!?;:])(\S)', r'\1 \2', text)  # Add space after punctuation
        # Fix common OCR/formatting issues
        text = text.replace('(cid:136)', '•')  # Convert special bullet points
        text = re.sub(r'\s*\n\s*', ' ', text)  # Normalize line breaks
        return text.strip()

    def _split_into_chunks(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        """
        Split text into meaningful chunks with proper boundaries.
        
        Args:
            text: The text to split
            metadata: Metadata to include with each chunk
            
        Returns:
            List of document chunks with proper boundaries
        """
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid text provided for chunking")
            return []
            
        chunks = []
        start = 0
        text_length = len(text)
        min_chunk_size = 50  # Minimum size for a chunk to be considered valid
        iteration = 0
        last_progress = 0
        
        logger.info(f"Starting chunking of text with length: {text_length} characters")
        
        try:
            while (start < text_length - min_chunk_size and 
                   iteration < self.max_iterations):
                iteration += 1
                
                # Log progress every 10% or every 100 iterations
                progress = (start / text_length) * 100
                if progress >= last_progress + 10 or iteration % 100 == 0:
                    logger.info(f"Progress: {progress:.1f}% - Created {len(chunks)} chunks so far")
                    last_progress = int(progress // 10) * 10
                
                # Calculate the maximum end position for this chunk
                max_end = min(start + self.chunk_size, text_length)
                
                # Find a meaningful boundary
                try:
                    end = self._find_meaningful_boundary(text, start, max_end)
                    if end <= start:  # Ensure we're making progress
                        logger.warning(f"No progress made in chunking. Start: {start}, End: {end}")
                        end = start + min(100, text_length - start - 1)
                        if end <= start:  # Still no progress, abort
                            logger.error("Cannot make progress in chunking. Aborting.")
                            break
                except Exception as e:
                    logger.error(f"Error finding boundary: {str(e)}")
                    break
                
                # Get the chunk text and clean it up
                chunk_text = self._clean_chunk_text(text[start:end])
                
                # Only add if we have meaningful content
                if len(chunk_text) >= min_chunk_size:
                    # Ensure the chunk ends with proper punctuation if needed
                    if end < text_length and not chunk_text[-1] in {'.', '!', '?', ';', ':'}:
                        chunk_text = chunk_text.rstrip(',') + '.'
                    
                    chunk_id = str(uuid.uuid4())
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        'chunk_number': len(chunks) + 1,
                        'char_start': start,
                        'char_end': end
                    })
                    
                    chunks.append(DocumentChunk(
                        id=chunk_id,
                        content=chunk_text,
                        metadata=chunk_metadata
                    ))
                
                # Calculate overlap start position (25% of chunk size or 100 chars, whichever is smaller)
                overlap_size = min(max(self.chunk_size // 4, 100), self.chunk_overlap)
                overlap_start = max(start, end - overlap_size)
                
                # Find the nearest sentence start in the overlap region
                try:
                    sentence_start = text.rfind('. ', overlap_start, end)
                    if sentence_start > start and (end - sentence_start) < (overlap_size * 1.5):
                        new_start = sentence_start + 2  # Move past the period and space
                    else:
                        # If no good sentence start, just use the overlap start
                        new_start = overlap_start
                    
                    # Ensure we're making progress
                    if new_start <= start:
                        logger.warning(f"No progress made in chunk position. Forcing progress. Old: {start}, New: {new_start}")
                        new_start = start + min(100, text_length - start - 1)
                        if new_start <= start:  # Still no progress, abort
                            logger.error("Cannot make progress in chunk position. Aborting.")
                            break
                            
                    start = new_start
                    
                except Exception as e:
                    logger.error(f"Error calculating next chunk start: {str(e)}")
                    start = end  # Move forward to prevent getting stuck
                
                # Safety check to prevent infinite loops
                if start >= text_length - min_chunk_size:
                    logger.info("Reached end of text")
                    break
                    
        except Exception as e:
            logger.error(f"Unexpected error during chunking: {str(e)}")
            raise
            
        logger.info(f"Finished chunking. Created {len(chunks)} chunks total.")
        return chunks
    
    def chunks_to_qdrant_format(self, chunks: List[DocumentChunk]) -> List[Dict]:
        """
        Convert document chunks to Qdrant-compatible format.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            List of dictionaries in Qdrant format
        """
        qdrant_points = []
        
        for chunk in chunks:
            qdrant_point = {
                'id': chunk.id,
                'vector': None,  # Will be filled in by the embedding model
                'payload': {
                    'content': chunk.content,
                    **chunk.metadata
                }
            }
            qdrant_points.append(qdrant_point)
            
        return qdrant_points


def process_handbook(input_path: str, output_dir: str = None) -> List[Dict]:
    """
    Process a handbook JSON file and prepare it for Qdrant.
    
    Args:
        input_path: Path to the input JSON file
        output_dir: Directory to save the processed output (optional)
        
    Returns:
        List of Qdrant-compatible points
        
    Raises:
        FileNotFoundError: If the input file doesn't exist
        json.JSONDecodeError: If the input file is not valid JSON
        Exception: For other processing errors
    """
    start_time = time.time()
    logger.info(f"Starting to process handbook: {input_path}")
    
    try:
        # Validate input file
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        # Initialize chunker with default parameters
        chunker = HandbookChunker()
        
        # Load the data
        logger.info("Loading JSON data...")
        data = chunker.load_json(str(input_path))
        
        if not data or 'sections' not in data or not data['sections']:
            logger.warning("No sections found in the input data")
            return []
            
        logger.info(f"Found {len(data['sections'])} sections to process")
        
        # Process the data
        logger.info("Starting chunking process...")
        chunks = chunker.create_chunks(data)
        
        if not chunks:
            logger.warning("No chunks were created from the input data")
            return []
            
        logger.info(f"Created {len(chunks)} chunks in total")
        
        # Convert to Qdrant format
        logger.info("Converting to Qdrant format...")
        qdrant_points = chunker.chunks_to_qdrant_format(chunks)
        
        # Save to output directory if specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save full output
            output_path = output_dir / 'handbook_chunks.json'
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(qdrant_points, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved {len(qdrant_points)} chunks to {output_path}")
            except IOError as e:
                logger.error(f"Failed to save output file: {str(e)}")
                raise
        
        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        
        return qdrant_points
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {input_path}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error processing handbook: {str(e)}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process handbook JSON for Qdrant vector storage')
    parser.add_argument('input', help='Path to input JSON file')
    parser.add_argument('-o', '--output', help='Output directory for processed chunks', default='./output')
    
    args = parser.parse_args()
    process_handbook(args.input, args.output)
