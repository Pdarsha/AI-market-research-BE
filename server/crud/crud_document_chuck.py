from typing import Optional
from sqlalchemy.orm import Session, Query

from server.crud.base import CRUDBase
from server.models.workspace import DocumentChunk
from server.schemas.workspace import DocumentChunkCreate, DocumentChunkUpdate


class CRUDDocumentChunk(
    CRUDBase[DocumentChunk, DocumentChunkCreate, DocumentChunkUpdate]
):
    def create(self, db: Session, *, obj_in: DocumentChunkCreate) -> DocumentChunk:
        return super().create(db, obj_in=obj_in)

    def update(
        self, db: Session, *, db_obj: DocumentChunk, obj_in: DocumentChunkUpdate
    ) -> DocumentChunk:
        return super().update(db, db_obj=db_obj, obj_in=obj_in)

    def remove(self, db: Session, *, id: str) -> Optional[DocumentChunk]:
        return super().remove(db, id=id)

    def get_chunks_by_id(self, db: Session, *, citation_id: str, user_id: str):
        return (
            db.query(DocumentChunk)
            .filter(
                DocumentChunk.id == citation_id,
                DocumentChunk.user_id == str(user_id),
            )
            .first()
        )

    def get_chunks_by_document_id(
        self, db: Session, *, chunk_id: str, document_id: str, user_id: str
    ):
        return (
            db.query(DocumentChunk)
            .filter(
                DocumentChunk.id == chunk_id,
                DocumentChunk.document_id == document_id,
                DocumentChunk.user_id == user_id,
            )
            .first()
        )

    def query_check(
        self,
        db: Session,
        *,
        user_id: str,
        workspace_id: str | None = None,
    ) -> Query:
        if workspace_id:
            return db.query(DocumentChunk).filter(
                (DocumentChunk.workspace_id == workspace_id) | (DocumentChunk.user_id == str(user_id))
            )
        return db.query(DocumentChunk).filter(DocumentChunk.user_id == str(user_id))

    def set_search_parameters(self, db: Session, ef_search: int) -> None:
        from sqlalchemy import text
        try:
            if db.bind is not None and db.bind.dialect.name == "postgresql":
                db.execute(
                    text("SET LOCAL hnsw.ef_search = :ef").bindparams(
                        ef=int(ef_search)
                    )
                )
                try:
                    with db.begin_nested():
                        db.execute(
                            text("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
                        )
                except Exception:
                    pass
        except Exception:
            pass

    def create_chunks(
        self, db: Session, *, chunks: list[DocumentChunkCreate]
    ) -> list[DocumentChunk]:
        db_objs = [
            DocumentChunk(
                document_id=c.document_id,
                user_id=c.user_id,
                workspace_id=c.workspace_id,
                file_name=c.file_name,
                chunk_number=c.chunk_number,
                total_chunks=c.total_chunks,
                content=c.content,
                embedding=c.embedding,
                start_offset=c.start_offset,
                end_offset=c.end_offset,
                page_number=c.page_number,
                section_title=c.section_title,
                document_path=c.document_path,
                document_url=c.document_url,
            )
            for c in chunks
        ]
        db.add_all(db_objs)
        db.commit()
        return db_objs

    def search_nearest_chunks(
        self,
        db: Session,
        *,
        query_embedding: list[float],
        user_id: str,
        workspace_id: Optional[str] = None,
        document_ids: Optional[list[str]] = None,
        candidate_k: int,
    ) -> list[DocumentChunk]:
        distance = self.model.embedding.cosine_distance(query_embedding)

        def _scoped():
            return self.query_check(db, user_id=user_id, workspace_id=workspace_id)

        if document_ids and len(document_ids) > 1:
            import math
            per_doc = max(1, math.ceil(candidate_k / len(document_ids)))
            rows: list[DocumentChunk] = []
            for doc_id in document_ids:
                rows.extend(
                    _scoped()
                    .filter(self.model.document_id == doc_id)
                    .order_by(distance)
                    .limit(per_doc)
                    .all()
                )
            return rows
        else:
            stmt = _scoped()
            if document_ids:
                stmt = stmt.filter(self.model.document_id.in_(document_ids))
            return stmt.order_by(distance).limit(candidate_k).all()


document_chunk = CRUDDocumentChunk(DocumentChunk)
