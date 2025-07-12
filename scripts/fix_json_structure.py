#!/usr/bin/env python3
"""
Script to fix the structure of manual_extract.json by handling line breaks in strings.
"""
import json
import re
from pathlib import Path

def fix_json_structure(input_path: str, output_path: str) -> bool:
    """
    Fix the structure of a JSON file with line breaks in strings.
    
    Args:
        input_path: Path to the input JSON file
        output_path: Path to save the fixed JSON file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the file line by line
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Process each line to fix line breaks in strings
        fixed_lines = []
        current_key = None
        current_value = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line starts a new key-value pair
            if '"' in line and ':' in line and '"' in line.split(':', 1)[0]:
                # If we were building a value, save it first
                if current_key and current_value:
                    fixed_lines.append(f'    "{current_key}": "{" ".join(current_value)}",')
                    current_value = []
                
                # Start a new key-value pair
                key_part, value_part = line.split(':', 1)
                current_key = key_part.strip().strip('"')
                value_part = value_part.strip()
                
                # If the value is a string, start collecting it
                if value_part.startswith('"') and value_part.endswith('"'):
                    fixed_lines.append(f'    "{current_key}": {value_part},')
                    current_key = None
                elif value_part.startswith('"'):
                    current_value.append(value_part[1:].strip())
                else:
                    current_value.append(value_part.strip())
            elif current_key and current_value:
                # Continue collecting the current value
                current_value.append(line.strip())
        
        # Add the last key-value pair if it exists
        if current_key and current_value:
            fixed_lines.append(f'    "{current_key}": "{" ".join(current_value)}"')
        
        # Create the fixed JSON content
        fixed_content = '{\n' + '\n'.join(fixed_lines) + '\n}'
        
        # Try to parse the fixed content to validate it
        try:
            json.loads(fixed_content)
            
            # Save the fixed content
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            print(f"Successfully fixed and saved to {output_path}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Failed to validate fixed JSON: {e}")
            print("The fixed content might still be useful. Saving anyway...")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print(f"Saved potentially fixed content to {output_path}")
            return False
            
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix JSON structure with line breaks in strings')
    parser.add_argument('input', help='Path to input JSON file')
    parser.add_argument('-o', '--output', help='Path to save fixed JSON file', 
                      default='fixed_output.json')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        exit(1)
    
    success = fix_json_structure(args.input, args.output)
    
    if success:
        print("File fixed successfully!")
        print(f"Output saved to: {args.output}")
    else:
        print("There were some issues fixing the file. The output may need manual review.")
    
    exit(0 if success else 1)
