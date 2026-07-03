from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from server.config import settings
from server.db.base_class import Base

load_dotenv()


engine = create_engine(settings.SQLALCHEMY_DATABASE_URI)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def ensure_vector_extension() -> None:
    """Enable pgvector before any table with a VECTOR column is created.

    ``document_chunks.embedding`` is a ``VECTOR`` column, which requires the
    ``vector`` extension to exist first. Both ``create_all`` (startup) and the
    Alembic migration depend on this. No-op on non-PostgreSQL backends.
    """
    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


ensure_vector_extension()

Base.metadata.create_all(engine)
