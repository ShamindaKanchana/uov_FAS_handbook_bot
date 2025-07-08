import pdfplumber
from pathlib import Path
from typing import List, Dict, Any

class SectionDetector:
    def __init__(self, pdf_path: str):
        """
        Initialize Section Detector
        
        Args:
            pdf_path: Path to the PDF file
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at {self.pdf_path}")

    def detect_sections(self) -> List[Dict[str, Any]]:
        """
        Detect sections in the PDF based on font size and style.
        
        Returns:
            List of dictionaries containing section information
        """
        sections = []
        current_section = None
        
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    # Extract words with formatting information
                    words = page.extract_words(
                        keep_blank_chars=True,
                        extra_attrs=["size", "fontname"]  # Removed 'bold' from here
                    )
                    
                    for word in words:
                        # Get font size, default to 0 if not available
                        font_size = word.get('size', 0)
                        
                        # Detect headings by font size (adjust threshold as needed)
                        is_heading = (
                            font_size >= 20.0 or  # Large font size
                            word['text'].isupper() and len(word['text']) > 3  # UPPERCASE words
                        )
                        
                        if is_heading:
                            # Save previous section if exists
                            if current_section and current_section["content"].strip():
                                sections.append(current_section)
                            
                            # Start new section
                            current_section = {
                                "title": word['text'].strip(),
                                "content": "",
                                "page": page_num,
                                "font_size": font_size,
                                "font_name": word.get('fontname', 'unknown')
                            }
                        elif current_section:
                            # Add to current section's content
                            current_section["content"] += word['text'] + " "
                    
                    # Add section break between pages if content continues
                    if current_section and current_section["content"].strip():
                        current_section["content"] += "\n"
                        
                except Exception as e:
                    print(f"Warning: Error processing page {page_num}: {str(e)}")
                    continue
        
        # Add the last section
        if current_section and current_section["content"].strip():
            sections.append(current_section)
            
        return sections
# Example usage
if __name__ == "__main__":
    try:
        # Update this path to your PDF
        detector = SectionDetector("data/raw/Hand-Book-Fas-2021-2022.pdf")
        sections = detector.detect_sections()
        
        # Print first 5 sections as example
        for i, section in enumerate(sections[:5], 1):
            print(f"\nSection {i}: {section['title']}")
            print(f"Page: {section['page']}")
            print(f"Preview: {section['content'][:100]}...")
            
    except Exception as e:
        print(f"Error: {str(e)}")