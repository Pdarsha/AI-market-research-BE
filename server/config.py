import os
from pathlib import Path
from typing import List, Optional, Union
from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    debug: bool = os.environ.get("DEBUG")
    FASTAPI_CONFIG: str = os.environ.get("FASTAPI_CONFIG")

    SQLALCHEMY_DATABASE_URI: str = os.getenv("SQLALCHEMY_DATABASE_URI")
    MAIL_PORT: str = os.getenv("MAIL_PORT")
    MAIL_SERVER: str = os.getenv("MAIL_SERVER")

    AUTH_TOKEN: str = os.getenv("AUTH_TOKEN")
    GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "")

    MAIL_USERNAME: str = os.getenv("MAIL_USERNAME")
    MAIL_PASSWORD: str = os.getenv("MAIL_PASSWORD")
    MAIL_FROM: str = os.getenv("MAIL_FROM")

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    SERPAPI_KEY: str = os.getenv("SERPAPI_KEY")
    ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    ALPHA_VANTAGE_NEWS_CACHE_TTL: int = int(
        os.getenv("ALPHA_VANTAGE_NEWS_CACHE_TTL", "600")
    )
    ALPHA_VANTAGE_NEWS_MAX_RESULTS: int = int(
        os.getenv("ALPHA_VANTAGE_NEWS_MAX_RESULTS", "10")
    )
    SEC_EDGAR_USER_AGENT: str = os.getenv(
        "SEC_EDGAR_USER_AGENT", "CapitalEdgeAI/1.0 (support@example.com)"
    )
    SEC_EDGAR_MAPPING_CACHE_TTL: int = int(
        os.getenv("SEC_EDGAR_MAPPING_CACHE_TTL", "86400")
    )
    SEC_EDGAR_FACTS_CACHE_TTL: int = int(
        os.getenv("SEC_EDGAR_FACTS_CACHE_TTL", "43200")
    )
    ROIC_API_KEY: str = os.getenv("ROIC_API_KEY", "")
    ROIC_CACHE_TTL: int = int(os.getenv("ROIC_CACHE_TTL", "3600"))
    COMPANIES_HOUSE_API_KEY: str = os.getenv("COMPANIES_HOUSE_API_KEY", "")
    COMPANIES_HOUSE_BASE_URL: str = os.getenv(
        "COMPANIES_HOUSE_BASE_URL", "https://api.company-information.service.gov.uk"
    )
    COMPANIES_HOUSE_CACHE_TTL: int = int(
        os.getenv("COMPANIES_HOUSE_CACHE_TTL", "3600")
    )

    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4.1-mini")
    THINK_LONGER_MODEL: str = os.getenv("THINK_LONGER_MODEL", "gpt-5.6-terra")
    THINK_LONGER_REASONING_EFFORT: str = os.getenv("THINK_LONGER_REASONING_EFFORT", "medium")
    THINK_LONGER_FALLBACK_MODEL: str = os.getenv("THINK_LONGER_FALLBACK_MODEL", os.getenv("LLM_MODEL", "gpt-4.1-mini"))
    THINK_LONGER_DAILY_LIMIT: int = int(os.getenv("THINK_LONGER_DAILY_LIMIT", "10"))
    
    DEEP_RESEARCH_MODEL: str = os.getenv("DEEP_RESEARCH_MODEL", "gpt-5.6-terra")
    DEEP_RESEARCH_REASONING_EFFORT: str = os.getenv("DEEP_RESEARCH_REASONING_EFFORT", "high")

    # -- OpenAI embeddings -------------------------------------------------
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL") or None

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
    EMBEDDING_MAX_RETRIES: int = int(os.getenv("EMBEDDING_MAX_RETRIES", "3"))

    # -- Chunking / retrieval ---------------------------------------------
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))

    # -- OCR ---------------------------------------------------------------
    OCR_ENABLED: bool = os.getenv("OCR_ENABLED", "true").lower() == "true"
    OCR_ENGINE: str = os.getenv("OCR_ENGINE", "tesseract").strip().lower()
    OCR_PREPROCESS: bool = os.getenv("OCR_PREPROCESS", "true").lower() == "true"
    OCR_PDF_RENDER_DPI: int = int(os.getenv("OCR_PDF_RENDER_DPI", "200"))
    OCR_MAX_WORKERS: int = int(os.getenv("OCR_MAX_WORKERS", "0"))
    OCR_REC_BATCH: int = int(os.getenv("OCR_REC_BATCH", "6"))
    OCR_CLS_BATCH: int = int(os.getenv("OCR_CLS_BATCH", "24"))
    OCR_MEM_ARENA: bool = os.getenv("OCR_MEM_ARENA", "false").lower() == "true"

    # -- Re-ranking (two-stage retrieval) --------------------------------
    RAG_RERANK_ENABLED: bool = os.getenv("RAG_RERANK_ENABLED", "true").lower() == "true"
    RAG_RERANK_CANDIDATES: int = int(os.getenv("RAG_RERANK_CANDIDATES", "20"))
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "gpt-4.1-mini")
    RAG_MAX_CONTEXT_CHUNKS: int = int(os.getenv("RAG_MAX_CONTEXT_CHUNKS", "12"))
    RAG_HNSW_EF_SEARCH: int = int(os.getenv("RAG_HNSW_EF_SEARCH", "200"))

    # -- Document viewer / citations --------------------------------------
    SIGNED_URL_TTL: int = int(os.getenv("SIGNED_URL_TTL", "3600"))

    # -- Web search --------------------------------------------------------
    WEB_SEARCH_ENGINE: str = os.getenv("WEB_SEARCH_ENGINE", "google")
    WEB_SEARCH_RESULTS: int = int(os.getenv("WEB_SEARCH_RESULTS", "5"))
    WEB_SEARCH_COUNTRY: str = os.getenv("WEB_SEARCH_COUNTRY", "in")
    WEB_SEARCH_LANGUAGE: str = os.getenv("WEB_SEARCH_LANGUAGE", "en")

    # -- PDF Export Branding -----------------------------------------------
    PDF_COMPANY_NAME: str = os.getenv("PDF_COMPANY_NAME", "CapitalEdge AI")
    PDF_LOGO_PATH: str = os.getenv(
        "PDF_LOGO_PATH",
        str(Path(__file__).resolve().parent.parent / "media" / "logo" / "logo.png"),
    )
    PDF_WATERMARK_TEXT: str = os.getenv("PDF_WATERMARK_TEXT", "CapitalEdge AI")
    PDF_HEADER_HEIGHT: float = float(os.getenv("PDF_HEADER_HEIGHT", "60"))
    PDF_FOOTER_HEIGHT: float = float(os.getenv("PDF_FOOTER_HEIGHT", "35"))
    PDF_WATERMARK_OPACITY: float = float(os.getenv("PDF_WATERMARK_OPACITY", "0.08"))
    PDF_LOGO_HEIGHT: float = float(os.getenv("PDF_LOGO_HEIGHT", "22"))
    PDF_LOGO_MARGIN: float = float(os.getenv("PDF_LOGO_MARGIN", "40"))
    PDF_LOGO_DPI: int = int(os.getenv("PDF_LOGO_DPI", "300"))
    PDF_MARGIN: float = float(os.getenv("PDF_MARGIN", "40"))

    BACKEND_CORS_ORIGINS: List = []

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)


settings = Settings()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
WORKSPACE_DOCS_DIR = STATIC_DIR / "workspace_docs"
AVATARS_DIR = STATIC_DIR / "avatars"


def create_storage_directories():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    AVATARS_DIR.mkdir(parents=True, exist_ok=True)
