from typing import Optional
import logging
import uuid
from sqlalchemy.orm import Session

from server.crud.base import CRUDBase
from server.models.workspace import DocumentChatLink
from server.schemas.workspace import DocumentChatLinkCreate, DocumentChatLinkUpdate

logger = logging.getLogger(__name__)


class CRUDDocumentChatLink(
    CRUDBase[DocumentChatLink, DocumentChatLinkCreate, DocumentChatLinkUpdate]
):
    def create(self, db: Session, *, obj_in: DocumentChatLinkCreate) -> DocumentChatLink:
        return super().create(db, obj_in=obj_in)

    def update(
        self,
        db: Session,
        *,
        db_obj: DocumentChatLink,
        obj_in: DocumentChatLinkUpdate,
    ) -> DocumentChatLink:
        return super().update(db, db_obj=db_obj, obj_in=obj_in)

    def remove(self, db: Session, *, id: str) -> Optional[DocumentChatLink]:
        return super().remove(db, id=id)

    def ensure_link(
        self,
        db: Session,
        *,
        document_id: str,
        chat_id: str,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> tuple[DocumentChatLink, bool]:
        """Idempotently ensure a (document_id, chat_id) link exists.

        Returns ``(link, created)`` where *created* is ``True`` when a new row
        was inserted and ``False`` when the link already existed.
        """
        existing = (
            db.query(DocumentChatLink)
            .filter(
                DocumentChatLink.document_id == document_id,
                DocumentChatLink.chat_id == chat_id,
            )
            .first()
        )
        if existing:
            logger.debug(
                "document_chat_link: already exists doc=%s chat=%s",
                document_id,
                chat_id,
            )
            return existing, False

        link = DocumentChatLink(
            id=str(uuid.uuid4()),
            document_id=document_id,
            chat_id=chat_id,
            workspace_id=workspace_id,
            user_id=user_id,
        )
        db.add(link)
        db.commit()
        db.refresh(link)

        logger.info(
            "document_chat_link: created doc=%s chat=%s workspace=%s",
            document_id,
            chat_id,
            workspace_id,
        )
        return link, True

    def ensure_links_for_chat(
        self,
        db: Session,
        *,
        document_ids: list[str],
        chat_id: str,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> dict[str, bool]:
        """Idempotently ensure links exist for every document in *document_ids*.

        Returns a mapping of ``{document_id: created}`` — ``True`` when a new
        row was inserted, ``False`` when the link already existed. All inserts
        are batched in a single commit.
        """
        if not document_ids:
            return {}

        # Fetch all already-existing links for this chat in one query.
        existing_doc_ids = {
            row.document_id
            for row in (
                db.query(DocumentChatLink.document_id)
                .filter(
                    DocumentChatLink.chat_id == chat_id,
                    DocumentChatLink.document_id.in_(document_ids),
                )
                .all()
            )
        }

        created: dict[str, bool] = {}
        for doc_id in document_ids:
            if doc_id in existing_doc_ids:
                logger.debug(
                    "document_chat_link: already exists doc=%s chat=%s", doc_id, chat_id
                )
                created[doc_id] = False
                continue

            link = DocumentChatLink(
                id=str(uuid.uuid4()),
                document_id=doc_id,
                chat_id=chat_id,
                workspace_id=workspace_id,
                user_id=user_id,
            )
            db.add(link)
            created[doc_id] = True
            logger.info(
                "document_chat_link: created doc=%s chat=%s workspace=%s",
                doc_id,
                chat_id,
                workspace_id,
            )

        if any(v for v in created.values()):
            db.commit()

        return created


document_chat_link = CRUDDocumentChatLink(DocumentChatLink)
