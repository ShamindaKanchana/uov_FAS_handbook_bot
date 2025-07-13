import os
import atexit
import signal
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from functools import lru_cache

from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

class QdrantClientSingleton:
    _instances = {}
    
    def __new__(cls, path: Optional[os.PathLike] = None):
        path_str = str(Path(path).resolve()) if path else ':memory:'
        
        if path_str not in cls._instances:
            instance = super(QdrantClientSingleton, cls).__new__(cls)
            instance._path = path_str
            instance._client = None
            instance._initialized = False
            cls._instances[path_str] = instance
            atexit.register(instance._cleanup)
            
            # Register signal handlers for clean exit
            signal.signal(signal.SIGINT, instance._signal_handler)
            signal.signal(signal.SIGTERM, instance._signal_handler)
            
            # Initialize the client
            instance._initialize_client()
            instance._initialized = True
            
        return cls._instances[path_str]
    
    def _initialize_client(self):
        if self._client is None and not self._initialized:
            try:
                logger.info(f"Initializing Qdrant client at: {self._path}")
                self._client = QdrantClient(
                    path=self._path if self._path != ':memory:' else None,
                    prefer_grpc=True,
                    force_disable_check_same_thread=True
                )
                logger.info("Qdrant client initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Qdrant client: {e}")
                self._cleanup()
                raise
    
    def get_client(self):
        if self._client is None:
            self._initialize_client()
        return self._client
    
    def _cleanup(self):
        if self._client is not None:
            try:
                logger.info("Closing Qdrant client...")
                self._client.close()
                logger.info("Qdrant client closed successfully.")
            except Exception as e:
                logger.error(f"Error closing Qdrant client: {e}")
            finally:
                self._client = None
                
        # Clean up lock files if using file-based storage
        if self._path and self._path != ':memory:':
            lock_file = Path(self._path) / ".lock"
            if lock_file.exists():
                try:
                    lock_file.unlink()
                    logger.info(f"Removed lock file: {lock_file}")
                except Exception as e:
                    logger.warning(f"Could not remove lock file {lock_file}: {e}")
        
        # Remove from instances dictionary
        if self._path in self.__class__._instances:
            del self.__class__._instances[self._path]
    
    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, cleaning up...")
        self._cleanup()
        exit(0)
    
    def __del__(self):
        self._cleanup()

# Global function to get a Qdrant client
def get_qdrant_client(path: Optional[os.PathLike] = None):
    """Get a Qdrant client instance with the specified path."""
    return QdrantClientSingleton(path).get_client()