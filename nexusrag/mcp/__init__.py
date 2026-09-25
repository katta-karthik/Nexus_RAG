"""
MCP module for NexusRAG.
Exposes lightweight document operations for external MCP clients.
"""

from .server import NexusRAGMCPServer

__all__ = ["NexusRAGMCPServer"]
