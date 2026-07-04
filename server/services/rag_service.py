"""RAG retrieval service.

Orchestrates the query side of the pipeline:

    query -> embed query (OpenAI) -> pgvector cosine search
          -> top-k chunks -> build context string

Retrieval is scoped to the requesting user and, when supplied, a single
workspace. Ordering is done in-database by pgvector's cosine distance operator,
so the nearest chunks come first.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from server.config import settings
from server.models.workspace import DocumentChunk
from server.services.embedding_service import EmbeddingService, get_embedding_service
from server.utils.prompt import ai_assistent, classification_prompt, doc_agent_system_agent

logger = logging.getLogger(__name__)


class RAGService:
    """Embed a query and retrieve the most relevant document chunks."""

    def __init__(self, embedding_service: EmbeddingService) -> None:
        self._embedder = embedding_service

    def retrieve(
        self,
        db: Session,
        *,
        query: str,
        user_id: str,
        top_k: Optional[int] = None,
        document_ids: List[str],
        workspace_id: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Return the top-k nearest :class:`DocumentChunk` rows for ``query``."""
        top_k = top_k or settings.RAG_TOP_K
        query_embedding = self._embedder.embed_text(query)

        stmt = db.query(DocumentChunk).filter(
            DocumentChunk.user_id == str(user_id)
        )
        if workspace_id:
            stmt = stmt.filter(DocumentChunk.workspace_id == str(workspace_id))
        if document_ids:
            stmt = stmt.filter(DocumentChunk.document_id.in_(document_ids))

        rows = (
            stmt.order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
            .limit(top_k)
            .all()
        )
        logger.debug(
            "RAG retrieve: user=%s workspace=%s query_len=%d hits=%d",
            user_id,
            workspace_id,
            len(query),
            len(rows),
        )

        return rows
        # documents = [row.content for row in rows]
        # file_names = list(dict.fromkeys(row.file_name for row in rows))

        # reranked_docs = self.rerank_documents(query=query, documents=documents, top_k=top_k)

        # return reranked_docs, file_names
    
    # def rerank_documents(self, query, documents, top_k=3):
    #     pairs = [(query, doc) for doc in documents]
    #     scores = load_rerank_model().predict(pairs)
    #     ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    #     return [doc for doc, score in ranked[:top_k]]


    @staticmethod
    def build_context(chunks: List[DocumentChunk]) -> str:
        """Concatenate chunk contents into a single context string for the LLM."""
        return "\n\n---\n\n".join(chunk.content for chunk in chunks if chunk.content)

    @staticmethod
    def build_messages(
        *,
        history: List[dict],
        question: str,
        retrieved_context: Optional[list] = None,
    ) -> List[dict]:
        """Prompt generation.

        When ``retrieved_context`` is provided (RAG path) the document system prompt
        grounds the answer in the retrieved chunks; otherwise the plain-assistant
        system prompt is used. Previous chat history is always included for
        conversational continuity, and the current question is appended last.
        """
        if retrieved_context is not None:
            system_prompt = doc_agent_system_agent(retrieved_context)
        else:
            system_prompt = ai_assistent()

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": question})
        return messages

    @staticmethod
    def source_file_names(chunks: List[DocumentChunk]) -> List[str]:
        """Return the distinct source file names behind ``chunks`` (order-stable)."""
        return list(dict.fromkeys(c.file_name for c in chunks if c.file_name))

    def retrieve_context(
        self,
        db: Session,
        *,
        query: str,
        user_id: str,
        top_k: Optional[int] = None,
        document_ids: List[str],
        workspace_id: Optional[str] = None,
    ) -> Tuple[str, List[str]]:
        """Convenience wrapper returning ``(context_string, source_file_names)``."""
        chunks = self.retrieve(
            db,
            query=query,
            user_id=user_id,
            top_k=top_k,
            document_ids=document_ids,
            workspace_id=workspace_id,
        )
        # return chunks, self.source_file_names(chunks)
        return self.build_context(chunks), self.source_file_names(chunks)

    @staticmethod
    def generate_response(client, *, messages: List[dict], model: str) -> str:
        """LLM response generation using streaming, returning the full text.

        Streaming is preserved (``stream=True``) exactly as the existing chat flow;
        the chunks are aggregated into the final response string.
        """
        ai_response = ""
        completion =  client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=1024,
                    stream=True,
                    temperature=0
                )
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                ai_response += chunk.choices[0].delta.content
        return ai_response
    
    @staticmethod
    def classify_requires_document(client, question: str, model: str) -> bool:
        """Ask the LLM whether ``question`` needs the uploaded document(s).

        Sends a lightweight one-word (``YES``/``NO``) routing prompt. On any error the
        call fails safe by returning ``True`` (prefer grounding the answer in the
        documents rather than silently ignoring them).
        """
        try:
            completion =  client.chat.completions.create(
                    model=model,
                    messages=[
                    {"role": "system", "content": classification_prompt()},
                    {"role": "user", "content": f'Question:\n"{question}"\nAnswer:'},
                ],
                max_tokens=1024,
                temperature=0
                )
            answer = (completion.choices[0].message.content or "").strip().upper()
            requires = answer.startswith("Y")
            logger.info(
                "rag.classify: verdict=%s (raw=%r) question_len=%d",
                requires,
                answer,
                len(question),
            )
            return requires
        except Exception:
            logger.exception(
                "rag.classify: classification failed; defaulting to RAG (use documents)"
            )
            return True

@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """Return the shared :class:`RAGService` singleton."""
    return RAGService(embedding_service=get_embedding_service())