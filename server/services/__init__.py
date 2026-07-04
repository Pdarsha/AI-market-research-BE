"""Service layer for the RAG document pipeline.

Each service has a single responsibility and is loosely coupled to the others:

* :mod:`server.services.ocr_service` — text extraction only.
* :mod:`server.services.chunking_service` — chunk creation only.
* :mod:`server.services.embedding_service` — OpenAI embedding generation only.
* :mod:`server.services.document_processor` — orchestrates ingestion.
* :mod:`server.services.rag_service` — orchestrates retrieval.

Module-level singleton accessors (``get_*_service``) are provided so request
handlers can share one instance without wiring dependencies manually.
"""

from server.services.chunking_service import Chunk, ChunkingService, get_chunking_service
from server.services.document_processor import (
    DocumentProcessor,
    get_document_processor,
)
from server.services.embedding_service import EmbeddingService, get_embedding_service
from server.services.ocr_service import OCRService, get_ocr_service
from server.services.rag_service import RAGService, get_rag_service

__all__ = [
    "Chunk",
    "ChunkingService",
    "get_chunking_service",
    "DocumentProcessor",
    "get_document_processor",
    "EmbeddingService",
    "get_embedding_service",
    "OCRService",
    "get_ocr_service",
    "RAGService",
    "get_rag_service",
]