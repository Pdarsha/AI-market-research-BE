"""Service layer for the RAG document pipeline.

Each service has a single responsibility and is loosely coupled to the others:

* :mod:`server.services.ocr_service` — text extraction only.
* :mod:`server.services.chunking_service` — chunk creation only.
* :mod:`server.services.embedding_service` — OpenAI embedding generation only.
* :mod:`server.services.rerank_service` — cross-encoder re-ranking only.
* :mod:`server.services.document_processor` — orchestrates ingestion.
* :mod:`server.services.rag_service` — orchestrates retrieval (search + re-rank).

Module-level singleton accessors (``get_*_service``) are provided so request
handlers can share one instance without wiring dependencies manually.
"""

from server.services.chunking_service import (
    Chunk,
    ChunkingService,
    get_chunking_service,
)
from server.services.document_processor import (
    DocumentProcessor,
    get_document_processor,
)
from server.services.embedding_service import EmbeddingService, get_embedding_service
from server.services.financial_news_mcp import FinancialNewsMCPTool, financial_news
from server.services.ocr_service import OCRService, get_ocr_service
from server.services.rag_service import RAGService, get_rag_service
from server.services.rerank_service import RerankService, get_rerank_service
from server.services.sec_edgar_mcp import SECEDGARClient, sec_company_facts
from server.services.roic_ai_mcp import RoicAIMCPTool, roic_ai
from server.services.prompt_template_registry import (
    PromptTemplate,
    PromptTemplateRegistry,
    get_prompt_template_registry,
)
from server.services.skill_suggestion_registry import (
    SkillSuggestion,
    SkillSuggestionRegistry,
    get_skill_suggestion_registry,
)
from server.services.audit_logger import AuditLogger

__all__ = [
    "Chunk",
    "ChunkingService",
    "get_chunking_service",
    "DocumentProcessor",
    "get_document_processor",
    "EmbeddingService",
    "get_embedding_service",
    "FinancialNewsMCPTool",
    "financial_news",
    "OCRService",
    "get_ocr_service",
    "RAGService",
    "get_rag_service",
    "RerankService",
    "get_rerank_service",
    "SECEDGARClient",
    "sec_company_facts",
    "RoicAIMCPTool",
    "roic_ai",
    "PromptTemplate",
    "PromptTemplateRegistry",
    "get_prompt_template_registry",
    "SkillSuggestion",
    "SkillSuggestionRegistry",
    "get_skill_suggestion_registry",
    "AuditLogger",
]

