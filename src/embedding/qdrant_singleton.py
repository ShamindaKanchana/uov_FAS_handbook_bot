from pathlib import Path
from typing import Optional
from qdrant_client import QdrantClient


class QdrantClientSingleton:
    _instance = None
    _client = None
    _path = None

    def __new__(cls, path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super(QdrantClientSingleton, cls).__new__(cls)
            cls._path = Path(path) if path else None
            cls._initialize_client()
        return cls._instance

    @classmethod
    def _initialize_client(cls):
        """Initialize the Qdrant client with the specified path or in-memory."""
        if cls._path:
            cls._path.mkdir(parents=True, exist_ok=True)
            cls._client = QdrantClient(
                path=str(cls._path),
                prefer_grpc=False,
                force_disable_check_same_thread=True
            )
        else:
            cls._client = QdrantClient(":memory:")

    @classmethod
    def get_client(cls) -> QdrantClient:
        """Get the Qdrant client instance."""
        if cls._client is None:
            cls._initialize_client()
        return cls._client

    @classmethod
    def reset(cls):
        """Reset the singleton instance (mainly for testing)."""
        cls._instance = None
        cls._client = None
        cls._path = None