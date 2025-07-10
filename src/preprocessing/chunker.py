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
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, *, skip_empty_sections: bool = True):
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
        self.skip_empty_sections = skip_empty_sections
    
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
        # Optional filtering & sorting
        sections = data.get('sections', [])
        if self.skip_empty_sections:
            sections = [s for s in sections if s.get('content') or s.get('subsections')]

        # Sort sections intelligently (numeric → alphabetic → others)
        sections = sorted(sections, key=self._sort_key_for_section)

        chunks: List[DocumentChunk] = []
        for section in sections:
            chunks.extend(self._process_section(section))

        return chunks
    
    def _extract_title_components(self, title: str) -> tuple[str, str, list[str]]:
        """
        Extract main title, subtitle, and any additional components from a title.
        Handles various formats like:
        - "1.2.3 Title: Subtitle (Additional Info)"
        - "Title - Subtitle"
        - "Title (Code): Subtitle"
        - "Title: Subtitle - Additional Info"
        """
        title = str(title).strip()
        if not title:
            return '', '', []
            
        # Initialize components
        main_title = title
        subtitle = ''
        additional = []
        
        # Try to extract subtitle after colon first (most common pattern)
        if ':' in title:
            parts = title.split(':', 1)
            main_title = parts[0].strip()
            remaining = parts[1].strip()
            
            # Check if there's a parenthetical after the colon
            if '(' in remaining and ')' in remaining.split('(')[1]:
                sub_parts = re.split(r'(\(.*?\))', remaining, 1)
                subtitle = sub_parts[0].strip()
                if len(sub_parts) > 1:
                    additional.append(sub_parts[1].strip('() '))
            else:
                subtitle = remaining
        # Try dash separator
        elif ' - ' in title:
            parts = title.split(' - ', 1)
            main_title = parts[0].strip()
            subtitle = parts[1].strip()
        # Try en/em dash
        elif '–' in title or '—' in title:
            parts = re.split(r'[–—]', title, 1)
            main_title = parts[0].strip()
            if len(parts) > 1:
                subtitle = parts[1].strip()
        
        # Extract any parentheticals from main title
        if '(' in main_title and ')' in main_title:
            paren_matches = re.findall(r'\(([^)]+)\)', main_title)
            if paren_matches:
                main_title = re.sub(r'\s*\([^)]+\)', '', main_title).strip()
                additional.extend([m.strip() for m in paren_matches if m.strip()])
        
        # Clean up any remaining punctuation
        main_title = main_title.strip(' :.-')
        subtitle = subtitle.strip(' :.-')
        
        return main_title, subtitle, additional

    def _build_hierarchical_path(self, hierarchy: List[Dict], current_section: Dict) -> str:
        """Build a hierarchical path from the section hierarchy."""
        path_parts = []
        
        # Add parent sections
        for h in hierarchy:
            if h.get('section_number'):
                path_parts.append(h['section_number'])
            if h.get('title'):
                path_parts.append(h['title'])
        
        # Add current section
        if current_section.get('section_number'):
            path_parts.append(current_section['section_number'])
        if current_section.get('title'):
            path_parts.append(current_section['title'])
        
        return ' > '.join(p for p in path_parts if p)

    def _process_section(self, section: Dict, parent_metadata: Optional[Dict] = None, 
                        hierarchy: List[Dict] = None) -> List[DocumentChunk]:
        """
        Process a section and its subsections recursively with improved handling of academic content.
        """
        if not section or not isinstance(section, dict):
            return []
            
        parent_metadata = parent_metadata or {}
        hierarchy = hierarchy or []
        
        # Extract section components
        section_number = str(section.get('section_number', '')).strip()
        title = str(section.get('title', '')).strip()
        page_start = int(section.get('page_start', 0))
        
        # Extract title components
        main_title, subtitle, additional = self._extract_title_components(title)
        # Fallbacks for empty titles to ensure meaningful hierarchy/metadata
        if not main_title:
            if subtitle:
                main_title = subtitle
                subtitle = ''
            elif section_number:
                main_title = f"Section {section_number}"
        
        # Create current section info
        current_section = {
            'section_number': section_number,
            'title': main_title,
            'subtitle': subtitle,
            'additional': additional,
            'page_start': page_start
        }
        
        # Build metadata
        hierarchy_titles: List[str] = [h['title'] for h in hierarchy] if hierarchy else []
        if main_title:
            hierarchy_titles.append(main_title)

        section_metadata = {
            'section_number': section_number,
            'title': main_title,
            'subtitle': subtitle,  # Include subtitle in metadata
            'additional_components': additional,  # Include additional components
            'page_start': section.get('page_start', 0),
            'source': 'handbook',
            'content_type': 'section' if main_title else 'unnamed_section',
            'has_subcontent': bool(section.get('subsections')),
            # Full hierarchical chain (root → … → current)
            'hierarchy': hierarchy_titles,
            # Readable path string for UI / debugging e.g. "Environmental Science > Department of Bio-science"
            'section_path': ' > '.join(hierarchy_titles)
        }
        
        # Add course code if this looks like a course section
        if re.match(r'^[A-Z]{2,4}\s*\d{3,4}[A-Z]?$', main_title):
            section_metadata['course_code'] = main_title
            section_metadata['content_type'] = 'course_description'
        
        # Ensure subtitle is included in the hierarchy with all components
        current_section = {
            'section_number': section_number,
            'title': main_title,
            'subtitle': subtitle,
            'additional': additional
        }
        # Update hierarchy for subsections
        current_hierarchy = hierarchy.copy()
        if main_title:  # Only add to hierarchy if we have a title
            current_hierarchy.append(current_section)
        
        chunks = []
        seen_chunks = set()
        
        # Process the section's content
        content = section.get('content', '')
        if content and isinstance(content, str) and content.strip():
            # Clean content first to handle special cases
            cleaned_content = self._clean_chunk_text(content.strip())
            if not cleaned_content:
                return chunks
            
            # Build context-aware content
            context_lines = []
            for level, ancestor in enumerate(current_hierarchy, 1):
                indent = '  ' * (level - 1)
                line_parts = []
                
                # Add section number if available
                if ancestor.get('section_number'):
                    line_parts.append(ancestor['section_number'])
                
                # Add main title
                if ancestor.get('title'):
                    line_parts.append(ancestor['title'])
                
                # Add subtitle if exists
                if ancestor.get('subtitle'):
                    line_parts[-1] += f": {ancestor['subtitle']}"
                
                # Add additional components in parentheses
                if ancestor.get('additional'):
                    line_parts[-1] += f" ({', '.join(ancestor['additional'])})"
                
                if line_parts:
                    context_lines.append(f"{indent}{' '.join(line_parts)}")
            
            # Combine context with content
            context = '\n'.join(context_lines)
            full_content = f"{context}\n\n{cleaned_content}"
            
            # Split into chunks
            content_chunks = self._split_into_chunks(full_content, section_metadata)
            for chunk in content_chunks:
                chunk_text = chunk.content.strip()
                if chunk_text and chunk_text not in seen_chunks:
                    # Enhance chunk metadata
                    chunk.metadata.update({
                        'word_count': len(chunk_text.split()),
                        'char_count': len(chunk_text),
                        'avg_word_length': sum(len(word) for word in chunk_text.split()) / max(1, len(chunk_text.split())),
                        'starts_with_section': chunk_text.startswith(section_number) if section_number else False,
                        'contains_table': bool(re.search(r'\|.*\|', chunk_text)) or bool(re.search(r'\+[-]+\+', chunk_text))
                    })
                    seen_chunks.add(chunk_text)
                    chunks.append(chunk)
        
        # Process subsections
        for subsection in self._sort_subsections(section.get('subsections', [])):
            if subsection:
                subsection_chunks = self._process_section(
                    subsection, 
                    section_metadata,
                    current_hierarchy
                )
                for chunk in subsection_chunks:
                    chunk_text = chunk.content.strip()
                    if chunk_text and chunk_text not in seen_chunks:
                        seen_chunks.add(chunk_text)
                        chunks.append(chunk)
        
        return chunks
    
    def _find_meaningful_boundary(self, text: str, start: int, max_end: int) -> int:
        """
        Find the most meaningful boundary point for chunking.
        Prioritizes sentence endings, paragraph breaks, nested titles, and other natural boundaries.
        Ensures chunks start with capital letters and maintain minimum meaningful size.
        
        Args:
            text: The full text
            start: Start position of current chunk
            max_end: Maximum end position (chunk_size)
            
        Returns:
            Position to end the current chunk
        """
        if not text or max_end >= len(text) - 1:
            return len(text) if text else start

        # Minimum chunk size (in characters) to avoid tiny chunks
        MIN_CHUNK_SIZE = 150
        
        # Look for paragraph breaks first (double newlines)
        para_break = text.rfind('\n\n', start, max_end)
        if para_break > start and (max_end - para_break) < 100:
            return para_break + 2
            
        # Look for section headers with or without subtitles (e.g., '3.2.1 Title: Subtitle' or 'Title: Subtitle')
        section_header = self._find_section_header(text, start, max_end)
        if section_header > start + MIN_CHUNK_SIZE:
            return section_header
            
        # Look for colon endings that might indicate a subtitle or section header
        colon_pos = text.rfind(':', start, max_end)
        if colon_pos > start + MIN_CHUNK_SIZE and colon_pos > text.rfind('\n', start, colon_pos):
            # Check if this is likely a subtitle by looking at the next line
            next_newline = text.find('\n', colon_pos, max_end)
            if next_newline != -1 and (next_newline - colon_pos) < 100:  # Reasonable subtitle length
                return colon_pos + 1
            
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
                
                # Ensure next character starts a new sentence (capital letter or end of text)
                next_pos = pos + len(ender)
                if next_pos >= len(text) or text[next_pos].isupper() or text[next_pos].isspace():
                    return next_pos
        
        # Try other meaningful boundaries with improved handling
        boundaries = [
            ('\n', 1),           # Newline
            ('; ', 2),           # Semicolon
            (': ', 2),           # Colon (often indicates a title or definition)
            (', ', 2),           # Comma
            (' - ', 3),          # Dash (often used for parenthetical statements)
            (' (', 2),           # Parenthesis
            (') ', 2),           # Close parenthesis
            (' and ', 5),        # Word boundary
            (' or ', 4),         # Word boundary
            (' ', 1)             # Space as last resort
        ]
        
        for boundary, offset in boundaries:
            boundary_pos = text.rfind(boundary, look_behind, max_end)
            # Ensure minimum chunk size and check if boundary is meaningful
            if boundary_pos > start + MIN_CHUNK_SIZE:
                # For colons, only use if it's likely a title/definition
                if boundary == ': ' and not self._is_likely_title(text, boundary_pos):
                    continue
                return boundary_pos + offset
        
        # If we have a minimum chunk size, try to extend to reach it
        if (max_end - start) < MIN_CHUNK_SIZE and max_end < len(text):
            next_space = text.find(' ', max_end)
            if next_space > 0 and (next_space - start) <= self.chunk_size * 1.2:  # Allow slight overflow
                return next_space + 1
        
        # Ensure we're not creating tiny chunks
        if (max_end - start) < MIN_CHUNK_SIZE and max_end < len(text):
            return min(start + MIN_CHUNK_SIZE, len(text))
            
        return max_end
    
    def _find_section_header(self, text: str, start: int, max_end: int) -> int:
        """
        Find section headers in the format 'X.Y.Z Title' or 'Title:'
        """
        # Look for section numbers (e.g., '3.2.1 ')
        section_pattern = r'\n\d+(?:\.\d+)*\s+[A-Z]'
        for match in re.finditer(section_pattern, text[start:max_end]):
            return start + match.start() + 1  # +1 to include the newline
            
        # Look for bold text (assuming it's marked with ** or __)
        bold_pattern = r'\*\*([^*]+)\*\*|__([^_]+)__'
        for match in re.finditer(bold_pattern, text[start:max_end]):
            return start + match.start()
            
        # Look for lines that end with a colon (potential titles)
        colon_pos = text.rfind(':', start, max_end)
        if colon_pos > start and colon_pos < max_end - 1:
            # Check if this is a title by looking at surrounding text
            if self._is_likely_title(text, colon_pos):
                return colon_pos + 1
                
        return -1
    
    def _is_likely_title(self, text: str, pos: int) -> bool:
        """
        Check if text around pos is likely a title/header
        """
        # Check if the line is short (likely a title)
        line_start = text.rfind('\n', 0, pos) + 1
        line_end = text.find('\n', pos)
        if line_end == -1:
            line_end = len(text)
            
        line = text[line_start:line_end].strip()
        
        # Titles are usually short and often end with a colon
        if len(line) > 100 or ' ' not in line:
            return False
            
        # Check if next line starts with capital letter (new paragraph)
        if line_end + 1 < len(text) and text[line_end + 1].isupper():
            return True
            
        # Check if previous line is empty (common before titles)
        if line_start > 1 and text[line_start-2:line_start].strip() == '':
            return True
            
        return False
    
    def _clean_chunk_text(self, text: str) -> str:
        """
        Clean and format chunk text with enhanced handling of academic content.
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Replace common OCR/formatting issues with academic context awareness
        replacements = {
            # Common OCR artifacts
            r'\(cid:\d+\)': '',  # Remove all (cid:XXX) patterns
            '\u2022': '•',        # Bullet points
            '\u25cf': '•',        # Another bullet point format
            '\u2013': '-',        # En dash
            '\u2014': ' -- ',     # Em dash with spaces
            '\u201c': '"',       # Left double quote
            '\u201d': '"',       # Right double quote
            '\u2018': "'",       # Left single quote
            '\u2019': "'",       # Right single quote
            '\u00a0': ' ',        # Non-breaking space
            '\u200b': '',         # Zero-width space
            
            # Fix common OCR errors
            'ﬁ': 'fi',
            'ﬂ': 'fl',
            'ﬀ': 'ff',
            'ﬃ': 'ffi',
            'ﬄ': 'ffl',
            '’': "'",
            '`': "'",
            '“': '"',
            '”': '"',
            '„': '"',
            '´': "'",
            '‘': "'",
            '‚': ',',
            '–': '-',
            '—': '--',
            '…': '...',
            
            # Common academic abbreviations
            'i.e.': 'that is',
            'e.g.': 'for example',
            'etc.': 'and so on',
            'et al.': 'and others',
            
            # Fix hyphenation at line breaks
            '-\n': '',
            '\n': ' ',
            '  ': ' ',
            
            # Fix spacing around dashes
            ' - ': ' -- ',
            '—': ' -- ',
            '–': ' -- ',
            
            # Fix spacing around slashes
            ' / ': '/',
            '/ ': '/',
            ' /': '/',
            
            # Fix spacing around parentheses and brackets
            '( ': '(',
            ' )': ')',
            '[ ': '[',
            ' ]': ']',
            
            # Fix spacing around quotes
            ' "': '"',
            '" ': '"',
            " '": "'",
            "' ": "'"
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        # Apply replacements
        for pattern, replacement in replacements.items():
            if isinstance(replacement, str):
                text = text.replace(pattern, replacement)
            else:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Normalize whitespace and newlines (preserve paragraphs)
        text = re.sub(r'[\r\n]+', '\n', text)  # Normalize newlines
        text = re.sub(r'[\t\f\v\u00A0]+', ' ', text)  # Normalize other whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
        
        # Fix common punctuation and spacing issues
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,!?;:])([A-Za-z])', r'\1 \2', text)  # Add space after punctuation
        
        # Fix academic-specific formatting
        text = re.sub(r'\b([A-Z][a-z]*\s+){2,5}\([A-Z]{2,4}\s*\d{3,4}[A-Z]?\)', 
                     lambda m: m.group(0).replace(' ', '_'), text)  # Preserve course codes with spaces
        
        # Fix common OCR errors (split words, merged words with numbers, etc.)
        text = re.sub(r'\b([A-Z])\s+([A-Z])\b', r'\1\2', text)  # Fix split words
        text = re.sub(r'([a-z])([0-9])', r'\1 \2', text)  # Add space between letter and number
        text = re.sub(r'([0-9])([A-Za-z])', r'\1 \2', text)  # Add space between number and letter
        
        # Fix common OCR errors in words
        text = re.sub(r'\b([A-Z])\s+([a-z])', lambda m: m.group(1).lower() + m.group(2), text)
        
        # Clean up any remaining control characters
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        
        # Normalize quotes and apostrophes
        text = re.sub(r'[\u2018\u2019\u02BC\u02BB]', "'", text)  # Single quotes
        text = re.sub(r'[\u201C\u201D\u201E\u201F]', '"', text)  # Double quotes
        
        # Fix spacing around dashes used as separators
        text = re.sub(r'\s*[-–—]\s*', ' -- ', text)
        
        # Fix spacing around slashes in common academic terms
        text = re.sub(r'\b([A-Za-z]+)\s*/\s*([A-Za-z]+)\b', r'\1/\2', text)  # e.g., and/or
        
        # Normalize multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Fix sentence spacing
        text = re.sub(r'\.(\s*[A-Z])', lambda m: '. ' + m.group(1).lstrip(), text)
        
        # Capitalize first letter of each sentence
        sentences = re.split(r'([.!?]\s+)', text)
        for i in range(0, len(sentences)-1, 2):
            if i+1 < len(sentences):
                sentences[i+1] = sentences[i+1][0].upper() + sentences[i+1][1:]
        text = ''.join(sentences)
        
        # Ensure proper spacing after section numbers (e.g., "1.1" -> "1.1 ")
        text = re.sub(r'(\d+)(\.\d+)+(?=[^\d.])', r'\g<0> ', text)
        
        # Ensure sentences end with proper punctuation
        text = text.strip()
        if text and text[-1] not in {'.', '!', '?', ';', ':'}:
            # Only add period if the text is a complete sentence
            if len(text.split()) > 3 and text[0].isupper():
                text += '.'
        
        # Clean up any double punctuation
        text = re.sub(r'([.,!?;:])\1+', r'\1', text)
        text = re.sub(r'([.,!?;:])[.,!?;:]+(\s|$)', r'\1\2', text)
        
        return text.strip()

    def _sort_key_for_section(self, section: Dict) -> tuple:
        """Return a tuple key for consistent section ordering."""
        sn = str(section.get('section_number', '')).strip()
        if sn.isdigit():
            return (0, int(sn))
        # Alphanumeric like 'A', 'B', 'C', or '1.2.3'
        num_part = re.findall(r'^\d+', sn)
        if num_part:
            try:
                return (0, float(num_part[0]))
            except ValueError:
                pass
        return (1, sn)

    def _sort_subsections(self, subsections: List[Dict]) -> List[Dict]:
        """Sort subsections using same logic as top-level."""
        return sorted(subsections, key=self._sort_key_for_section)

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
                    # Create a deep copy of metadata to avoid modifying the original
                    chunk_metadata = {
                        **metadata,  # Include all original metadata
                        'chunk_number': len(chunks) + 1,
                        'char_start': start,
                        'char_end': end,
                        'is_complete': True,
                        'word_count': len(chunk_text.split())
                    }
                    
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


def process_handbook(input_path: str, output_dir: str = None, output_format: str = "jsonl") -> List[Dict]:
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
            
            if output_format.lower() == "jsonl":
                output_path = output_dir / 'handbook_chunks.jsonl'
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        for point in qdrant_points:
                            f.write(json.dumps(point, ensure_ascii=False) + "\n")
                    logger.info(f"Saved {len(qdrant_points)} chunks to {output_path} in JSONL format")
                except IOError as e:
                    logger.error(f"Failed to save output file: {str(e)}")
                    raise
            else:
                # default JSON list
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
    parser.add_argument('--format', choices=['json', 'jsonl'], default='jsonl', help='Output file format (jsonl recommended for large datasets)')
    
    args = parser.parse_args()
    process_handbook(args.input, args.output, args.format)
