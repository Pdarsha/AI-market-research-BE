import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from server.db.base_class import Base

# Dimensionality of the sentence-transformers/all-MiniLM-L6-v2 embeddings.
EMBEDDING_DIM = 1536


class Workspace(Base):
    id = Column(String, default=lambda: str(uuid.uuid4()), primary_key=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User")
    # Deleting a workspace cascades to its chats (and their messages) and to its
    # uploaded files. DB-side ondelete=CASCADE + passive_deletes avoids loading rows.
    chats = relationship(
        "ChatSession",
        back_populates="workspace",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    docs = relationship(
        "Document",
        back_populates="workspace",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Document(Base):
    id = Column(String, default=lambda: str(uuid.uuid4()), primary_key=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    # A file always belongs to a workspace (enforced at the API layer). Nullable
    # at the DB level only so legacy rows created before this refactor survive.
    workspace_id = Column(
        String,
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        index=True,
        nullable=True,
    )
    file_name = Column(String)
    file_format = Column(String)
    storage_path: Mapped[str] = mapped_column(String(500), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User")
    workspace = relationship("Workspace", back_populates="docs")
    # Chunks (with their pgvector embeddings) belong to a document. Deleting the
    # document cascades to them at the DB level; passive_deletes avoids loading rows.
    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class DocumentChunk(Base):
    """A single embedded chunk of a document, stored in PostgreSQL/pgvector.

    Replaces the former ChromaDB collection ``embedded``. The metadata that used
    to live on the Chroma record (``user_id``, ``workspace_id``, ``file_name`` …)
    is denormalized here so retrieval can filter by owner/workspace directly, and
    the ``embedding`` column holds the 1536-dim vector for similarity search.
    Foreign keys use ``ON DELETE CASCADE`` so deleting a document (or its
    workspace) also removes the associated embeddings.

    Citation metadata fields enable the frontend Citation View to highlight
    retrieved content within the original document.
    """

    id = Column(String, default=lambda: str(uuid.uuid4()), primary_key=True)
    document_id = Column(
        String,
        ForeignKey("documents.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    workspace_id = Column(
        String,
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        index=True,
        nullable=True,
    )
    file_name = Column(String)
    chunk_number = Column(Integer)
    total_chunks = Column(Integer)
    content = Column(Text)
    embedding = Column(Vector(EMBEDDING_DIM))

    # Citation metadata: text offsets within the original chunk for highlighting
    start_offset = Column(Integer, nullable=True)
    end_offset = Column(Integer, nullable=True)

    # Citation metadata: document structure information
    page_number = Column(Integer, nullable=True)
    section_title = Column(String, nullable=True)

    # Citation metadata: references to original document location
    document_path = Column(String, nullable=True)
    document_url = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")
    user = relationship("User")
    workspace = relationship("Workspace")


# HNSW index for approximate-nearest-neighbour cosine similarity search over the
# embeddings. Defined here so ``create_all`` builds it, mirroring the Alembic
# migration. Only materializes on PostgreSQL (pgvector); ignored elsewhere.
Index(
    "ix_document_chunks_embedding",
    DocumentChunk.embedding,
    postgresql_using="hnsw",
    postgresql_ops={"embedding": "vector_cosine_ops"},
)
