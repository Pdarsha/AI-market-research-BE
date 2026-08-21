from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from server.config import settings
from server.db.base_class import Base

load_dotenv()


db_url = settings.SQLALCHEMY_DATABASE_URI or ""
engine_kwargs = {}
if db_url.startswith("postgresql"):
    engine_kwargs = {
        "pool_size": 20,
        "max_overflow": 30,
        "pool_pre_ping": True,
        "pool_recycle": 300,
    }

engine = create_engine(db_url, **engine_kwargs)
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
