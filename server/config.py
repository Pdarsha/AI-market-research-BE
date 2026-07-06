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

    LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-4.1-mini")

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

    # -- Re-ranking (two-stage retrieval) --------------------------------
    RAG_RERANK_ENABLED: bool = os.getenv("RAG_RERANK_ENABLED", "true").lower() == "true"
    RAG_RERANK_CANDIDATES: int = int(os.getenv("RAG_RERANK_CANDIDATES", "20"))
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "openai/gpt-4.1-mini")

    # -- Web search --------------------------------------------------------
    WEB_SEARCH_ENGINE: str = os.getenv("WEB_SEARCH_ENGINE", "google")
    WEB_SEARCH_RESULTS: int = int(os.getenv("WEB_SEARCH_RESULTS", "5"))
    WEB_SEARCH_COUNTRY: str = os.getenv("WEB_SEARCH_COUNTRY", "in")
    WEB_SEARCH_LANGUAGE: str = os.getenv("WEB_SEARCH_LANGUAGE", "en")

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


def create_storage_directories():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
