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
        Skips empty content and prevents duplicate chunks.
        
        Args:
            section: The section to process
            parent_metadata: Metadata from parent sections
            
        Returns:
            List of document chunks from this section and its children
        """
        if not section or not isinstance(section, dict):
            return []
            
        if parent_metadata is None:
            parent_metadata = {}
            
        # Skip sections with empty or invalid content
        content = section.get('content', '')
        if not content or not isinstance(content, str) or not content.strip():
            content = None
        else:
            content = content.strip()
            
        # Skip if no content and no subsections
        if not content and not section.get('subsections'):
            return []
            
        # Create metadata for this section
        metadata = {
            'section_number': str(section.get('section_number', '')).strip(),
            'title': str(section.get('title', '')).strip(),
            'page_start': int(section.get('page_start', 0)),
            'source': 'handbook',
            **parent_metadata
        }
        
        chunks = []
        seen_chunks = set()  # Track seen content to prevent duplicates
        
        # Process the section's own content if it exists
        if content:
            # Clean the content first
            content = self._clean_chunk_text(content)
            if content:  # Only process if we have valid content after cleaning
                # Create chunks from the content
                content_chunks = self._split_into_chunks(content, metadata)
                # Filter out empty or duplicate chunks
                for chunk in content_chunks:
                    chunk_text = chunk.content.strip()
                    if chunk_text and chunk_text not in seen_chunks:
                        seen_chunks.add(chunk_text)
                        chunks.append(chunk)
        
        # Process subsections
        for subsection in section.get('subsections', []):
            if subsection:  # Only process non-empty subsections
                subsection_chunks = self._process_section(subsection, metadata)
                # Filter out empty or duplicate chunks
                for chunk in subsection_chunks:
                    chunk_text = chunk.content.strip()
                    if chunk_text and chunk_text not in seen_chunks:
                        seen_chunks.add(chunk_text)
                        chunks.append(chunk)
            
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
        if not text or max_end >= len(text) - 1:
            return len(text) if text else start
            
        # Look for paragraph breaks first (double newlines)
        para_break = text.rfind('\n\n', start, max_end)
        if para_break > start and (max_end - para_break) < 100:
            return para_break + 2
            
        # Look for sentence boundaries in the last 30% of the chunk
        look_behind = max(start, max_end - int(self.chunk_size * 0.3))
        
        # Common sentence endings with proper spacing
        sentence_enders = ['. ', '! ', '? ', '".', '!"', '?"', '.\n', '!\n', '?\n']
        
        # Check for sentence boundaries first
        for ender in sentence_enders:
            pos = text.rfind(ender, look_behind, max_end)
            if pos > start + (self.chunk_size // 2):
                # Skip common abbreviations and decimal points
                if ender in ['. ', '! ', '? ']:
                    prev_word = text[max(0, pos-10):pos].split()[-1].lower() if pos > 10 else ''
                    if any(prev_word.endswith(abbr) for abbr in ['dr', 'mr', 'mrs', 'ms', 'phd', 'etc', 'fig', 'no', 'vol', 'inc']):
                        continue
                    # Skip decimal points in numbers
                    if pos > 0 and text[pos-1].isdigit():
                        continue
                return pos + len(ender)
        
        # Try other meaningful boundaries
        boundaries = [
            ('\n', 1),           # Newline
            ('; ', 2),           # Semicolon
            (': ', 2),           # Colon
            (', ', 2),           # Comma
            (' - ', 3),          # Dash
            (' (', 2),           # Parenthesis
            (') ', 2),           # Close parenthesis
            (' and ', 5),        # Word boundary
            (' or ', 4),         # Word boundary
            (' ', 1)             # Space as last resort
        ]
        
        for boundary, offset in boundaries:
            boundary_pos = text.rfind(boundary, look_behind, max_end)
            if boundary_pos > start + (self.chunk_size // 3):  # More flexible minimum chunk size
                return boundary_pos + offset
        
        # If no good boundary found, try to find a space near the max_end
        space_pos = text.rfind(' ', max(start, max_end - 30), max_end)
        if space_pos > start + (self.chunk_size // 4):  # Ensure minimum chunk size
            return space_pos + 1
            
        # If we can't find a good boundary, return the max_end
        return max_end
    
    def _clean_chunk_text(self, text: str) -> str:
        """
        Clean and format chunk text while preserving meaningful structure.
        Handles special characters, OCR artifacts, and formatting issues.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned and formatted text, or empty string if text is invalid
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Replace common OCR/formatting issues
        replacements = {
            # Common OCR artifacts
            '(cid:136)': '•',  # Bullet points
            '(cid:13)': '',    # Special characters
            '\u2022': '•',    # Different bullet point format
            '\u25cf': '•',    # Another bullet point format
            '\u2013': '-',    # En dash
            '\u2014': '--',   # Em dash
            '\u201c': '"',   # Left double quote
            '\u201d': '"',   # Right double quote
            '\u2018': "'",   # Left single quote
            '\u2019': "'"    # Right single quote
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        # Normalize whitespace and newlines (preserve paragraphs)
        text = re.sub(r'[\r\n]+', '\n', text)  # Normalize newlines
        text = re.sub(r'[\t\f\v ]+', ' ', text)  # Normalize other whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
        
        # Fix common punctuation and spacing issues
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,!?;:])([A-Za-z])', r'\1 \2', text)  # Add space after punctuation
        
        # Fix common OCR errors (split words, merged words with numbers, etc.)
        text = re.sub(r'\b([A-Z])\s+([A-Z])\b', r'\1\2', text)  # Fix split words
        text = re.sub(r'([a-z])([0-9])', r'\1 \2', text)  # Add space between letter and number
        text = re.sub(r'([0-9])([A-Za-z])', r'\1 \2', text)  # Add space between number and letter
        
        # Remove any remaining control characters except newlines
        text = re.sub(r'[\x00-\x09\x0B-\x1F\x7F]', '', text)
        
        return text.strip()
        
        # Ensure sentences end with proper punctuation
        if text and text[-1] not in {'.', '!', '?', ';', ':'}:
            text = text.rstrip() + '.'
            
        return text.strip()

    def _is_meaningful_chunk(self, text: str) -> bool:
        """
        Check if a chunk of text is meaningful enough to keep.
        
        Args:
            text: The text to evaluate
            
        Returns:
            bool: True if the text is meaningful, False otherwise
        """
        if not text or len(text.strip()) < 50:  # Minimum length
            return False
            
        # Check if the text contains at least one complete sentence
        if not re.search(r'[.!?]\s+[A-Z]', text):
            return False
            
        # Check for common patterns that indicate incomplete chunks
        if re.search(r'\b(and|or|but|however|therefore|because|so|if|then|when|where|which|that|who|whom|whose)\s*$', 
                    text, re.IGNORECASE):
            return False
            
        return True
        
    def _find_optimal_boundary(self, text: str, start: int, max_end: int) -> int:
        """
        Find the best boundary point to split the text.
        
        Args:
            text: Full text
            start: Starting position for this chunk
            max_end: Maximum end position
            
        Returns:
            int: Best end position for the current chunk
        """
        # If we're near the end of the text, return the end
        if max_end >= len(text) - 1:
            return len(text)
            
        # First, try to find a paragraph break
        para_break = text.rfind('\n\n', start, max_end)
        if para_break > start and (max_end - para_break) < 150:  # If close to a paragraph break
            return para_break + 2
            
        # Look for sentence boundaries in the last 30% of the chunk
        look_behind = max(start, max_end - int(self.chunk_size * 0.3))
        
        # Try different boundary markers in order of preference
        boundaries = [
            ('. ', 2),     # Sentence end with space
            ('! ', 2),     # Exclamation with space
            ('? ', 2),     # Question with space
            ('\n', 1),    # Newline
            ('; ', 2),     # Semicolon with space
            (': ', 2),     # Colon with space
            (', ', 2),     # Comma with space
            (' ', 1)       # Space as last resort
        ]
        
        for boundary, offset in boundaries:
            boundary_pos = text.rfind(boundary, look_behind, max_end)
            if boundary_pos > start + (self.chunk_size // 2):  # Ensure reasonable chunk size
                return boundary_pos + offset
                
        # If no good boundary found, try to find a space near max_end
        space_pos = text.rfind(' ', max(start, max_end - 50), max_end)
        if space_pos > start:
            return space_pos + 1
            
        # If we can't find a good boundary, return the max_end
        return max_end
        
    def _split_into_chunks(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        """
        Split text into meaningful, complete chunks with proper boundaries.
        
        Args:
            text: The text to split
            metadata: Metadata to include with each chunk
            
        Returns:
            List of document chunks with proper boundaries
        """
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid text provided for chunking")
            return []
            
        # Clean the entire text first
        text = self._clean_chunk_text(text)
        if not text:
            return []
            
        chunks = []
        start = 0
        text_length = len(text)
        min_chunk_size = 100  # Minimum size for a chunk to be considered meaningful
        
        logger.info(f"Starting chunking of text with length: {text_length} characters")
        
        try:
            iteration = 0
            while start < text_length and iteration < self.max_iterations:
                iteration += 1
                
                # Calculate the maximum end position for this chunk
                max_end = min(start + self.chunk_size, text_length)
                
                # Find the best boundary
                end = self._find_meaningful_boundary(text, start, max_end)
                
                # Ensure we make progress
                if end <= start:
                    logger.warning(f"No progress made, forcing forward progress. Start: {start}, End: {end}")
                    start = min(start + 100, text_length)
                    continue
                
                # Get the chunk text and clean it
                chunk_text = text[start:end].strip()
                
                # Only add if we have meaningful content
                if self._is_meaningful_chunk(chunk_text):
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        'chunk_number': len(chunks) + 1,
                        'char_start': start,
                        'char_end': end,
                        'is_complete': True,
                        'word_count': len(chunk_text.split())
                    })
                    
                    chunks.append(DocumentChunk(
                        id=str(uuid.uuid4()),
                        content=chunk_text,
                        metadata=chunk_metadata
                    ))
                
                # Move start position, accounting for overlap
                next_start = end - self.chunk_overlap
                if next_start <= start:  # Ensure we make progress
                    next_start = start + (self.chunk_size // 2)
                start = next_start
                
                # If we're near the end, check if we should include the remaining text
                if start >= text_length - min_chunk_size:
                    remaining_text = text[start:].strip()
                    if len(remaining_text) >= min_chunk_size // 2:
                        chunks.append(DocumentChunk(
                            id=str(uuid.uuid4()),
                            content=remaining_text,
                            metadata={
                                **metadata,
                                'chunk_number': len(chunks) + 1,
                                'char_start': start,
                                'char_end': text_length,
                                'is_complete': True,
                                'word_count': len(remaining_text.split())
                            }
                        ))
                    break
                    
        except Exception as e:
            logger.error(f"Error during chunking: {str(e)}")
            raise
            
        logger.info(f"Finished chunking. Created {len(chunks)} chunks.")
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
