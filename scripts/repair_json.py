#!/usr/bin/env python3
"""
Script to repair a malformed JSON file with line breaks in strings.
This script handles the specific format of the manual_extract.json file.
"""
import json
import re
from pathlib import Path

def repair_json(input_path: str, output_path: str) -> bool:
    """
    Repair a JSON file with line breaks in strings.
    
    Args:
        input_path: Path to the input JSON file
        output_path: Path to save the repaired JSON file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the entire file as text
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split the content into lines and process each line
        lines = content.split('\n')
        
        # Initialize variables to store the JSON objects
        objects = []
        current_obj = {}
        current_key = None
        
        for line in lines:
            line = line.strip()
            if not line or line == '{' or line == '}':
                continue
                
            # Check if this is a key-value pair
            if ':' in line:
                # If we have a current key, finalize its value
                if current_key and current_key not in current_obj:
                    current_obj[current_key] = current_value.strip() if 'current_value' in locals() else ""
                
                # Start a new key-value pair
                key_part, value_part = line.split(':', 1)
                current_key = key_part.strip().strip('"')
                value_part = value_part.strip()
                
                # If the value is a complete string, add it to the object
                if value_part.startswith('"') and value_part.endswith('"'):
                    current_obj[current_key] = value_part[1:-1].strip()
                    current_key = None
                else:
                    # Start collecting the value
                    current_value = value_part[1:] if value_part.startswith('"') else value_part
            elif current_key:
                # Continue the current value
                if 'current_value' not in locals():
                    current_value = line
                else:
                    current_value += ' ' + line
        
        # Add the last object if it exists
        if current_key and current_key not in current_obj and 'current_value' in locals():
            current_obj[current_key] = current_value.strip()
        
        # Write the fixed JSON to the output file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([current_obj], f, indent=2, ensure_ascii=False)
        
        print(f"Successfully repaired JSON and saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error repairing JSON: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Repair a malformed JSON file with line breaks in strings')
    parser.add_argument('input', help='Path to input JSON file')
    parser.add_argument('-o', '--output', help='Path to save repaired JSON file', 
                      default='repaired_output.json')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        exit(1)
    
    success = repair_json(args.input, args.output)
    
    if success:
        print("JSON file repaired successfully!")
        print(f"Output saved to: {args.output}")
    else:
        print("Failed to repair the JSON file.")
    
    exit(0 if success else 1)
