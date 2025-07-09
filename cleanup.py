#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

def cleanup_qdrant_locks():
    """Remove Qdrant lock files to resolve 'already accessed' errors."""
    qdrant_path = Path("./qdrant_handbook")
    
    if not qdrant_path.exists():
        print("Qdrant directory not found.")
        return
    
    # Find and remove lock files
    lock_files = list(qdrant_path.rglob("*.lock")) + list(qdrant_path.rglob("lock"))
    
    if not lock_files:
        print("No lock files found.")
        return
    
    for lock_file in lock_files:
        try:
            lock_file.unlink()
            print(f"Removed lock file: {lock_file}")
        except Exception as e:
            print(f"Error removing {lock_file}: {e}")

if __name__ == "__main__":
    print("Cleaning up Qdrant lock files...")
    cleanup_qdrant_locks()
    print("Cleanup complete. Try running your query tool again.")
