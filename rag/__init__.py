"""
RAG (Retrieval-Augmented Generation) module for DeepCode

This module provides intelligent document indexing and querying capabilities
for research paper analysis and code reproduction workflows.

Core Components:
- RAGManager: Main interface for RAG operations
- RAGConfig: Configuration management for RAG settings
- QueryGenerator: LLM-powered query generation
"""

from .rag_manager import RAGManager
from .config import RAGConfig
from .query_generator import QueryGenerator

__all__ = ["RAGManager", "RAGConfig", "QueryGenerator"]
