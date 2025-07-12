import json
import re
from pathlib import Path

def clean_json_file(input_path: str, output_path: str = None) -> bool:
    """
    Clean a JSON file by fixing line breaks in strings and removing invalid control characters.
    
    Args:
        input_path: Path to the input JSON file
        output_path: Path to save the cleaned JSON file (if None, overwrites input)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the file content
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove line breaks within strings (but keep actual line breaks for JSON structure)
        lines = content.splitlines()
        cleaned_lines = []
        in_string = False
        current_line = ""
        
        for line in lines:
            for char in line:
                if char == '"':
                    in_string = not in_string
                    current_line += char
                elif char == '\n' and in_string:
                    # Skip newlines within strings
                    current_line += ' '
                else:
                    current_line += char
            
            # If we're not in a string, add the line and reset
            if not in_string:
                cleaned_lines.append(current_line)
                current_line = ""
            else:
                # If we're still in a string, add a space instead of newline
                current_line += ' '
        
        # Join the cleaned lines
        cleaned_content = '\n'.join(cleaned_lines)
        
        # Remove any remaining control characters except for \t, \n, \r
        cleaned_content = ''.join(char for char in cleaned_content if ord(char) >= 32 or char in '\t\n\r')
        
        # Try to parse the JSON to validate it
        json.loads(cleaned_content)
        
        # If we get here, the JSON is valid
        output_path = output_path or input_path
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
            
        return True
        
    except Exception as e:
        print(f"Error cleaning JSON file: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean a JSON file with line breaks in strings')
    parser.add_argument('input', help='Path to input JSON file')
    parser.add_argument('-o', '--output', help='Path to save cleaned JSON file (default: overwrite input)', default=None)
    
    args = parser.parse_args()
    
    if clean_json_file(args.input, args.output):
        print(f"Successfully cleaned JSON file: {args.output or args.input}")
    else:
        print("Failed to clean JSON file")
