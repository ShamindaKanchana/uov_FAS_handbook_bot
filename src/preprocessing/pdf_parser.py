import pdfplumber
from pathlib import Path
import sys
import json
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import re

class PDFParser:
    def __init__(self, pdf_path: str):
        """
        Initialize PDF Parser with enhanced text extraction capabilities.
        
        Args:
            pdf_path: Relative or absolute path to the PDF file
        """
        # Get the project root directory (src's parent)
        project_root = Path(__file__).parent.parent.parent
        # Resolve the full path to the PDF
        self.pdf_path = str(project_root / pdf_path)
        print(f"Looking for PDF at: {self.pdf_path}")
        
        # Verify the file exists
        if not Path(self.pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found at {self.pdf_path}")

    def extract_structured_content(self, output_path: str = None) -> Dict[str, Any]:
        """
        Extract structured content from the PDF and optionally save to JSON.
        
        Args:
            output_path: Path to save the JSON file. If None, returns the data.
            
        Returns:
            Dictionary with structured content
        """
        print("Starting PDF content extraction...")
        print(f"PDF Path: {self.pdf_path}")
        # Define the section hierarchy from the TOC
        section_hierarchy = {
            "1": {"title": "General Information", "subsections": {
                "1.1": "Introduction",
                "1.2": "Officers of the University of Vavuniya",
                "1.3": "Academic Staff of the Library",
                "1.4": "Executive Staff",
                "1.5": "University Medical Officer",
                "1.6": "Staff of the Faculty of Applied Science",
                "1.6.1": "Office of the Dean",
                "1.6.2": "Department of Bio-science",
                "1.6.3": "Department of Physical Science",
                "1.7": "Faculty Quality Assurance Cell (FQAC)"
            }},
            "2": {"title": "Degree Programme – Department of Bio-science", "subsections": {
                "2.1": "The Structure of the Programme",
                "2.1.1": "The Title of the Degree Programme",
                "2.1.2": "Admission",
                "2.1.3": "Medium of Instruction",
                "2.1.4": "Program Overview",
                "2.1.5": "Credit Valued Course Unit System",
                "2.1.6": "Volume of Learning",
                "2.1.7": "Opting for General Degree",
                "2.2": "Degree Programme Objectives and Graduate Profile",
                "2.2.1": "Programme Objectives",
                "2.2.2": "Graduate Profile",
                "2.3": "Evaluation System of the Degree Programme",
                "2.3.1": "Evaluation Methods",
                "2.3.2": "Grading system and Grade Point Average (GPA)",
                "2.3.3": "Examination Process",
                "2.3.4": "Award of Degree",
                "2.3.5": "Award of Classes",
                "2.3.6": "Award of Diploma/ Higher Diploma",
                "2.3.7": "Effective Date of the Degree",
                "2.4": "Curriculum Layout"
            }},
            "3": {"title": "Degree Programmes - Department of Physical Science", "subsections": {
                "3.1": "Structure of the Degree Programmes",
                "3.1.1": "The Names of the Degree Programmes",
                "3.1.2": "Admission",
                "3.1.3": "Medium of Instruction",
                "3.1.4": "Programmes Overview",
                "3.1.5": "Credit Valued Course Unit System",
                "3.1.6": "Selection to the Honours Degree Programmes",
                "3.1.7": "Opting for Bachelors Degree",
                "3.2": "Degree Programmes Objectives and Graduate Profiles",
                "3.2.1": "Programmes Objectives",
                "3.2.2": "Graduate Profiles",
                "3.2.3": "Career Prospects",
                "3.3": "Evaluation System of the Degree Programmes",
                "3.3.1": "Evaluation Methods",
                "3.3.2": "Grading System and Grade Point Average",
                "3.3.3": "Examination Process",
                "3.3.4": "Award of Degrees",
                "3.3.5": "Award of Classes",
                "3.3.6": "Award of Diploma/Higher Diploma",
                "3.3.7": "Effective Date of the Degree",
                "3.4": "Curriculum Layout"
            }},
            "4": {"title": "Examination Rules", "subsections": {}},
            "5": {"title": "Services and Facilities", "subsections": {}},
            "A": {"title": "AMC and Computer Science", "subsections": {}},
            "B": {"title": "Environmental Science", "subsections": {}},
            "C": {"title": "Information Technology", "subsections": {}}
        }

        try:
            content_data = {
                "sections": [],
                "metadata": {
                    "source": str(self.pdf_path),
                    "extracted_at": datetime.now().isoformat(),
                    "total_pages": 0
                }
            }

            with pdfplumber.open(self.pdf_path) as pdf:
                content_data["metadata"]["total_pages"] = len(pdf.pages)
                current_section = None
                current_subsection = None
                current_content = []
                current_page_num = 0

                # Flatten section patterns for matching and add debug info
                section_patterns = []
                print("\nSection patterns to look for:")
                print("-" * 50)
                for sec_num, sec_data in section_hierarchy.items():
                    section_patterns.append((sec_num, sec_data["title"]))
                    print(f"{sec_num}: {sec_data['title']}")
                    for sub_num, sub_title in sec_data["subsections"].items():
                        section_patterns.append((sub_num, sub_title))
                        print(f"  {sub_num}: {sub_title}")
                print("-" * 50)
                
                # Sort by section number for proper hierarchy
                section_patterns.sort(key=lambda x: [int(n) if n.isdigit() else ord(n) for n in re.split(r'(\d+)', x[0]) if n])
                
                print(f"Total section patterns: {len(section_patterns)}")

                for page in pdf.pages:
                    current_page_num += 1
                    text = page.extract_text() or ""
                    if not text.strip():
                        continue

                    lines = [line.strip() for line in text.split('\n') if line.strip()]

                    for line in lines:
                        # Check for section headers with more flexible matching
                        section_found = None
                        for sec_num, title in section_patterns:
                            # Normalize both the line and patterns for comparison
                            normalized_line = line.strip().lower()
                            normalized_title = title.lower()
                            
                            # Try multiple possible formats for section headers
                            patterns = [
                                f"{sec_num} {title}",  # "1.1 Title"
                                f"{sec_num}.{title}",  # "1.1.Title"
                                f"{sec_num}. {title}", # "1.1. Title"
                                f"{sec_num} {normalized_title}",  # case-insensitive
                                f"{sec_num}.{normalized_title}",  # case-insensitive with dot
                                f"{sec_num}. {normalized_title}", # case-insensitive with dot and space
                                normalized_title  # Just the normalized title by itself
                            ]
                            
                            # Check if any pattern matches (case-insensitive)
                            for pattern in patterns:
                                pattern = pattern.strip().lower()
                                if (normalized_line == pattern or 
                                    normalized_line.startswith(pattern + ' ') or
                                    normalized_line.endswith(' ' + pattern) or
                                    f' {pattern} ' in f' {normalized_line} '):
                                    section_found = (sec_num, title)
                                    break
                                    
                            if section_found:
                                # Print debug info
                                print(f"Found section {sec_num}: {title}")
                                break

                        if section_found:
                            sec_num, title = section_found
                            level = len(sec_num.split('.')) if '.' in sec_num else 1

                            # Save previous content
                            if current_subsection and current_content:
                                current_subsection["content"] = self._clean_content("\n".join(current_content))
                                current_content = []

                            # Update current section
                            if level == 1:  # Main section
                                current_section = {
                                    "section_number": sec_num,
                                    "title": title,
                                    "subsections": [],
                                    "page_start": current_page_num
                                }
                                content_data["sections"].append(current_section)
                                current_subsection = None
                            else:  # Subsection
                                current_subsection = {
                                    "section_number": sec_num,
                                    "title": title,
                                    "content": "",
                                    "page_start": current_page_num
                                }
                                if current_section:
                                    current_section["subsections"].append(current_subsection)
                        else:
                            # Add to current content if we're in a section
                            if current_subsection is not None:
                                current_content.append(line)
                            elif current_section and not current_section.get("subsections"):
                                # Handle case where main section has direct content
                                current_content.append(line)

                # Save the last section's content
                if current_subsection and current_content:
                    current_subsection["content"] = self._clean_content("\n".join(current_content))
                elif current_section and current_content:
                    current_section["content"] = self._clean_content("\n".join(current_content))

            # Save to file if output path is provided
            if output_path:
                output_path = str(Path(__file__).parent.parent.parent / output_path)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(content_data, f, indent=2, ensure_ascii=False)
                print(f"Content saved to {output_path}")

            return content_data

        except Exception as e:
            error_msg = f"Error extracting content: {str(e)}"
            print(error_msg)
            return {"error": error_msg, "error_type": type(e).__name__}

    @staticmethod
    def _clean_content(text: str) -> str:
        """Clean and normalize extracted content."""
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r'\s+', ' ', text.strip())
        # Fix common OCR/PDF extraction artifacts
        text = re.sub(r'-\s+', '', text)  # Handle hyphenated words
        return text

    def check_font_info(self, page_num: int = 0) -> dict:
        """
        Check available font information in the PDF.
        
        Args:
            page_num: Page number to check (0-based)
            
        Returns:
            Dictionary with font information
        """
        try:
            print(f"Opening PDF file: {self.pdf_path}")
            with pdfplumber.open(self.pdf_path) as pdf:
                print(f"Total pages in PDF: {len(pdf.pages)}")
                
                if page_num >= len(pdf.pages):
                    return {"error": f"Page {page_num} not found (total pages: {len(pdf.pages)})"}
                    
                page = pdf.pages[page_num]
                print(f"Processing page {page_num + 1}")
                
                # Try to extract text first
                text = page.extract_text()
                print(f"Extracted {len(text or '')} characters of text")
                
                # Check for character data
                chars = page.chars if hasattr(page, 'chars') else []
                print(f"Found {len(chars)} character objects")
                
                if not chars:
                    return {
                        "pdf_path": self.pdf_path,
                        "page": page_num + 1,
                        "has_char_data": False,
                        "text_available": bool(text),
                        "text_preview": (text or "")[:200] + ("..." if len(text or "") > 200 else ""),
                        "message": "No character-level data available"
                    }
                    
                # Get sample of font information
                sample_char = {k: chars[0].get(k) for k in chars[0].keys()}
                font_sizes = list(set(char['size'] for char in chars if 'size' in char))
                font_names = list(set(char['fontname'] for char in chars if 'fontname' in char))
                
                return {
                    "pdf_path": self.pdf_path,
                    "page": page_num + 1,
                    "total_pages": len(pdf.pages),
                    "has_char_data": True,
                    "text_available": bool(text),
                    "text_preview": (text or "")[:200] + ("..." if len(text or "") > 200 else ""),
                    "sample_character": sample_char,
                    "font_sizes": sorted(font_sizes),
                    "font_names": sorted(font_names),
                    "total_characters": len(chars)
                }
                
        except Exception as e:
            return {
                "pdf_path": self.pdf_path,
                "error": str(e),
                "error_type": type(e).__name__
            }

def main() -> int:
    """
    Main function to run the PDF parser from command line.
    
    Returns:
        int: Exit code (0 for success, non-zero for errors)
        
    Usage: 
        python -m src.preprocessing.pdf_parser [PDF_PATH] [--output-dir DIR] [--check-fonts]
    """
    import argparse
    from datetime import datetime

    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Extract structured content from PDF')
    parser.add_argument('pdf_path', 
                      help='Path to the PDF file (relative to project root)')
    parser.add_argument('--output-dir', 
                      default='data/processed',
                      help='Output directory for extracted content (default: data/processed)')
    parser.add_argument('--check-fonts', 
                      action='store_true',
                      help='Check font information before extraction')
    
    try:
        args = parser.parse_args()
    except Exception as e:
        print(f"❌ Error parsing arguments: {e}")
        return 1

    try:
        print(f"\n{'='*50}")
        print("PDF Parser - University Handbook Extractor")
        print(f"{'='*50}\n")
        
        # Resolve output directory
        project_root = Path(__file__).parent.parent.parent
        output_dir = (project_root / args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize parser with error handling
        try:
            parser = PDFParser(args.pdf_path)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return 1
            
        # Generate output filename
        pdf_name = Path(args.pdf_path).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"{pdf_name}_extracted_{timestamp}.json"
        
        # Optional font check
        if args.check_fonts:
            print("\n--- Font Information ---")
            try:
                font_info = parser.check_font_info(0)
                print(json.dumps(font_info, indent=2, default=str))
            except Exception as e:
                print(f"⚠️  Warning: Could not get font info: {e}")
            print("-" * 50 + "\n")
        
        # Extract content
        print(f"Extracting content from: {args.pdf_path}")
        try:
            result = parser.extract_structured_content(str(output_file))
            
            if "error" in result:
                print(f"\n❌ Error: {result['error']}")
                if 'error_type' in result:
                    print(f"Error Type: {result['error_type']}")
                return 1
                
            section_count = len(result.get('sections', []))
            print(f"\n✅ Successfully extracted {section_count} sections")
            print(f"   Output saved to: {output_file.relative_to(project_root)}")
            return 0
            
        except KeyboardInterrupt:
            print("\n⚠️  Operation cancelled by user")
            return 1
        
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
            