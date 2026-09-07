"""Adapter: un traduttore sottile per ogni strumento RAG, tutti con la stessa interfaccia."""

from adapters.base import AdapterAnswer, Document, RAGAdapter, load_documents

__all__ = ["AdapterAnswer", "Document", "RAGAdapter", "load_documents"]
