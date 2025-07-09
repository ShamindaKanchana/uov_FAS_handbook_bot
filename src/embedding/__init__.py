"""
Embedding module for the UoV FAS Handbook Bot.

This module provides functionality for generating text embeddings and managing
vector storage using Qdrant.
"""

from .qdrant_singleton import QdrantClientSingleton
from .embedder import TextEmbedder

__all__ = ['QdrantClientSingleton', 'TextEmbedder']
