from pathlib import Path
from typing import Dict, Any

# Base directory for the project
BASE_DIR = Path(__file__).parent.parent.parent

# Default Qdrant collection name
DEFAULT_COLLECTION = "uov_fas_handbook"

# Default embedding model
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Default vector size for the chosen model
VECTOR_SIZE = 384

# Default batch size for processing
DEFAULT_BATCH_SIZE = 100

# Directory to persist Qdrant data (relative to project root)
QDRANT_STORAGE_PATH = BASE_DIR / "database" / "qdrant"

# Default distance metric for vector search
DEFAULT_DISTANCE = "COSINE"

# Default configuration for Qdrant collection
COLLECTION_CONFIG = {
    "vectors": {
        "size": VECTOR_SIZE,
        "distance": DEFAULT_DISTANCE,
    },
    "optimizers_config": {
        "default_segment_number": 2,
    },
    "hnsw_config": {
        "payload_m": 16,
        "ef_construct": 100,
    },
}

def get_qdrant_config() -> Dict[str, Any]:
    """Return the Qdrant collection configuration."""
    return COLLECTION_CONFIG
