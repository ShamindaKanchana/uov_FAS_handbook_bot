#!/usr/bin/env python3
"""
Script to fix the malformed manual_extract.json file.
Handles line breaks in strings and ensures valid JSON output.
"""
import json
import re
from pathlib import Path

def fix_json_file(input_path: str, output_path: str) -> bool:
    """
    Fix a malformed JSON file with line breaks in strings.
    
    Args:
        input_path: Path to the input JSON file
        output_path: Path to save the fixed JSON file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the entire file as text
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove all newlines and extra whitespace
        content = ' '.join(content.split())
        
        # Fix the JSON structure - this is a custom fix for the specific format
        # The file appears to be a single JSON object with line breaks in strings
        # Let's try to parse it as a single object first
        try:
            # Try to parse as a single JSON object
            data = json.loads(content)
            
            # If successful, write it back with proper formatting
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            print(f"Successfully fixed and saved to {output_path}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse as single JSON object: {e}")
            
            # If that fails, try to process it as a list of JSON objects
            print("Attempting to process as a list of JSON objects...")
            
            # Split by closing brace followed by whitespace and an opening brace
            objects = re.split(r'}\s*{', content)
            
            # Add back the braces we split on
            if len(objects) > 1:
                objects = [objects[0] + '}'] + \
                         ['{' + obj + '}' for obj in objects[1:-1]] + \
                         ['{' + objects[-1]]
            
            # Try to parse each object
            parsed_objects = []
            for i, obj_str in enumerate(objects, 1):
                try:
                    obj = json.loads(obj_str)
                    parsed_objects.append(obj)
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse object {i}: {e}")
            
            if parsed_objects:
                # Write the array of objects
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(parsed_objects, f, indent=2, ensure_ascii=False)
                
                print(f"Successfully parsed and saved {len(parsed_objects)} objects to {output_path}")
                return True
            else:
                print("No valid JSON objects found in the file.")
                return False
                
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix malformed JSON file with line breaks in strings')
    parser.add_argument('input', help='Path to input JSON file')
    parser.add_argument('-o', '--output', help='Path to save fixed JSON file', 
                      default='fixed_output.json')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        exit(1)
    
    success = fix_json_file(args.input, args.output)
    
    if success:
        print("File fixed successfully!")
        print(f"Output saved to: {args.output}")
    else:
        print("Failed to fix the file. Please check the input format.")
        exit(1)
