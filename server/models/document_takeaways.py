import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, Index, func
from sqlalchemy.orm import relationship

from server.db.base_class import Base

class DocumentTakeawayGrid(Base):
    __tablename__ = "document_takeaway_grids"

    id = Column(String, default=lambda: str(uuid.uuid4()), primary_key=True)
    workspace_id = Column(String, ForeignKey("workspaces.id", ondelete="CASCADE"), index=True, nullable=False)
    name = Column(String, nullable=False)
    status = Column(String, default="draft", nullable=False) # draft, processing, completed, completed_with_errors, failed
    created_by = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    columns = relationship("DocumentTakeawayGridColumn", back_populates="grid", cascade="all, delete-orphan", order_by="DocumentTakeawayGridColumn.display_order")
    rows = relationship("DocumentTakeawayGridRow", back_populates="grid", cascade="all, delete-orphan", order_by="DocumentTakeawayGridRow.display_order")

class DocumentTakeawayGridColumn(Base):
    __tablename__ = "document_takeaway_grid_columns"

    id = Column(String, default=lambda: str(uuid.uuid4()), primary_key=True)
    grid_id = Column(String, ForeignKey("document_takeaway_grids.id", ondelete="CASCADE"), index=True, nullable=False)
    name = Column(String, nullable=False)
    prompt = Column(Text, nullable=True)
    column_type = Column(String, nullable=False) # document, ai_generated
    display_order = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    grid = relationship("DocumentTakeawayGrid", back_populates="columns")
    cells = relationship("DocumentTakeawayGridCell", back_populates="column", cascade="all, delete-orphan")

    __table_args__ = (
        Index("uix_grid_id_name_lower", "grid_id", func.lower(name), unique=True),
    )

class DocumentTakeawayGridRow(Base):
    __tablename__ = "document_takeaway_grid_rows"

    id = Column(String, default=lambda: str(uuid.uuid4()), primary_key=True)
    grid_id = Column(String, ForeignKey("document_takeaway_grids.id", ondelete="CASCADE"), index=True, nullable=False)
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=False)
    display_order = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    grid = relationship("DocumentTakeawayGrid", back_populates="rows")
    cells = relationship("DocumentTakeawayGridCell", back_populates="row", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("grid_id", "document_id", name="uix_grid_id_document_id"),
    )

class DocumentTakeawayGridCell(Base):
    __tablename__ = "document_takeaway_grid_cells"

    id = Column(String, default=lambda: str(uuid.uuid4()), primary_key=True)
    grid_row_id = Column(String, ForeignKey("document_takeaway_grid_rows.id", ondelete="CASCADE"), index=True, nullable=False)
    grid_column_id = Column(String, ForeignKey("document_takeaway_grid_columns.id", ondelete="CASCADE"), index=True, nullable=False)
    value = Column(Text, nullable=True)
    status = Column(String, default="pending", nullable=False) # pending, processing, completed, failed
    error_message = Column(Text, nullable=True)
    generated_at = Column(DateTime, nullable=True)
    model_name = Column(String, nullable=True)
    model_version = Column(String, nullable=True)
    prompt_version = Column(String, nullable=True)
    context_chunk_count = Column(Integer, nullable=True)
    confidence = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    row = relationship("DocumentTakeawayGridRow", back_populates="cells")
    column = relationship("DocumentTakeawayGridColumn", back_populates="cells")

    __table_args__ = (
        UniqueConstraint("grid_row_id", "grid_column_id", name="uix_grid_row_id_grid_column_id"),
    )
