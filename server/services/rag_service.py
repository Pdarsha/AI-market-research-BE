"""RAG retrieval service.

Orchestrates the query side of the pipeline with two-stage retrieval:

    query -> embed query (OpenAI) -> pgvector cosine search (candidate pool)
          -> cross-encoder re-rank -> top-k chunks -> build context string

Retrieval is scoped to the requesting user and, when supplied, a single
workspace. Stage 1 ordering is done in-database by pgvector's cosine distance
operator; stage 2 re-scores the candidates with a cross-encoder
(:mod:`server.services.rerank_service`) for sharper relevance before the
top-k are passed to the LLM.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Optional, Tuple

import serpapi
from sqlalchemy.orm import Session

from server.config import settings
from server.models.workspace import DocumentChunk
from server.services.embedding_service import EmbeddingService, get_embedding_service
from server.services.rerank_service import get_rerank_service
from server.utils.prompt import (
    ai_assistent,
    augmented_system_prompt,
    classification_prompt,
    doc_agent_system_agent,
    web_search_classification_prompt,
)

logger = logging.getLogger(__name__)

serp_client = serpapi.Client(api_key=settings.SERPAPI_KEY)

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
        """Return the top-k most relevant :class:`DocumentChunk` rows for ``query``.

        Two-stage retrieval:
          1. **Vector search** — fetch a broad candidate pool ordered by pgvector
             cosine distance (fast, approximate).
          2. **Re-ranking** — a cross-encoder re-scores the candidates against the
             query and the top ``top_k`` are kept (slower, more accurate).

        When re-ranking is disabled (``settings.RAG_RERANK_ENABLED``) the vector
        top-k is returned directly.
        """
        top_k = top_k or settings.RAG_TOP_K
        query_embedding = self._embedder.embed_text(query)

        stmt = db.query(DocumentChunk).filter(
            DocumentChunk.user_id == str(user_id)
        )
        if workspace_id:
            stmt = stmt.filter(DocumentChunk.workspace_id == str(workspace_id))
        if document_ids:
            stmt = stmt.filter(DocumentChunk.document_id.in_(document_ids))

        candidate_k = (
            max(top_k, settings.RAG_RERANK_CANDIDATES)
            if settings.RAG_RERANK_ENABLED
            else top_k
        )
        rows = (
            stmt.order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
            .limit(candidate_k)
            .all()
        )

        if settings.RAG_RERANK_ENABLED and rows:
            reranked = get_rerank_service().rerank(
                query,
                rows,
                top_k=top_k,
                text_getter=lambda chunk: chunk.content or "",
            )
            logger.debug(
                "RAG retrieve: user=%s workspace=%s query_len=%d candidates=%d reranked=%d",
                user_id,
                workspace_id,
                len(query),
                len(rows),
                len(reranked),
            )
            return reranked

        rows = rows[:top_k]
        logger.debug(
            "RAG retrieve: user=%s workspace=%s query_len=%d hits=%d (no rerank)",
            user_id,
            workspace_id,
            len(query),
            len(rows),
        )
        return rows

    @staticmethod
    def build_context(chunks: List[DocumentChunk]) -> str:
        """Concatenate chunk contents into a single context string for the LLM."""
        return "\n\n---\n\n".join(chunk.content for chunk in chunks if chunk.content)

    @staticmethod
    def build_messages(
        *,
        history: List[dict],
        question: str,
        retrieved_context: Optional[str] = None,
        web_context: Optional[str] = None,
    ) -> List[dict]:
        """Prompt generation.

        Grounds the answer in whatever context is available:
          * ``retrieved_context`` — RAG document chunks;
          * ``web_context``       — web search results.
        When either (or both) is present the augmented system prompt is used;
        with neither, the plain-assistant prompt is used. Previous chat history is
        always included for continuity, and the current question is appended last.
        """
        if retrieved_context is not None or web_context is not None:
            system_prompt = augmented_system_prompt(
                document_context=retrieved_context, web_context=web_context
            )
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

    @staticmethod
    def classify_requires_web_search(client, question: str, model: str) -> bool:
        """Ask the LLM whether ``question`` needs a live web search.

        Sends a lightweight one-word (``YES``/``NO``) routing prompt. On any error
        the call fails safe by returning ``False`` (skip the search rather than add
        latency / spurious context).
        """
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": web_search_classification_prompt()},
                    {"role": "user", "content": f'Question:\n"{question}"\nAnswer:'},
                ],
                max_tokens=1024,
                temperature=0,
            )
            answer = (completion.choices[0].message.content or "").strip().upper()
            requires = answer.startswith("Y")
            logger.info(
                "rag.classify_web: verdict=%s (raw=%r) question_len=%d",
                requires,
                answer,
                len(question),
            )
            return requires
        except Exception:
            logger.exception(
                "rag.classify_web: classification failed; defaulting to no web search"
            )
            return False

    @staticmethod
    def _format_scalar_fields(data: dict, skip: Tuple[str, ...]) -> str:
        """Flatten the scalar (str/number) fields of a SerpAPI widget to text.

        Nested objects, images and link metadata (in ``skip``) are ignored so the
        digest stays compact and LLM-friendly.
        """
        parts: List[str] = []
        for key, value in data.items():
            if key in skip:
                continue
            if isinstance(value, (str, int, float)) and str(value).strip():
                label = key.replace("_", " ").capitalize()
                parts.append(f"{label}: {value}")
        return "\n".join(parts)

    @classmethod
    def _extract_direct_answer(cls, results: dict) -> str:
        """Pull the direct-answer widgets (answer box / knowledge graph) out of a
        SerpAPI response. This is where live data (weather, prices, scores,
        definitions) lives — organic results usually only have links."""
        sections: List[str] = []
        _skip = (
            "type", "thumbnail", "link", "displayed_link", "images", "list",
            "table", "source", "serpapi_link", "people_also_search_for",
        )

        answer_box = results.get("answer_box")
        if isinstance(answer_box, dict):
            text = cls._format_scalar_fields(answer_box, skip=_skip)
            if text:
                sections.append("Direct answer:\n" + text)

        knowledge_graph = results.get("knowledge_graph")
        if isinstance(knowledge_graph, dict):
            text = cls._format_scalar_fields(
                knowledge_graph, skip=_skip + ("kgmid", "knowledge_graph_search_link"),
            )
            if text:
                sections.append("Knowledge graph:\n" + text)

        return "\n\n".join(sections)

    @classmethod
    def web_search(
        cls, query: str, max_results: Optional[int] = None
    ) -> Tuple[str, List[dict]]:
        """Run a web search and return ``(context_string, sources)``.

        Uses the configured engine (Google by default) and extracts, in priority
        order: the direct-answer widgets (answer box / knowledge graph — where
        live data such as weather and prices lives) followed by the top organic
        results (title + snippet + link). Best-effort: any failure (or no
        results) returns ``("", [])`` so the chat flow can continue.
        """
        max_results = max_results or settings.WEB_SEARCH_RESULTS
        try:
            params = {
                "engine": settings.WEB_SEARCH_ENGINE,
                "q": query,
                "gl": settings.WEB_SEARCH_COUNTRY,
                "hl": settings.WEB_SEARCH_LANGUAGE,
                "num": max_results,
            }
            results = serp_client.search(params)
        except Exception:
            logger.exception("rag.web_search: search failed for query_len=%d", len(query))
            return "", []

        sections: List[str] = []

        direct = cls._extract_direct_answer(results)
        if direct:
            sections.append(direct)

        organic = (results.get("organic_results") or [])[:max_results]
        lines: List[str] = []
        sources: List[dict] = []
        for index, result in enumerate(organic, start=1):
            title = (result.get("title") or "").strip()
            snippet = (result.get("snippet") or "").strip()
            link = (result.get("link") or "").strip()
            lines.append(f"[{index}] {title}\n{snippet}\nSource: {link}")
            if link:
                sources.append({"title": title, "link": link})
        if lines:
            sections.append("Top results:\n" + "\n\n".join(lines))

        context = "\n\n".join(sections)
        logger.info(
            "rag.web_search: engine=%s query_len=%d direct_answer=%s organic=%d",
            settings.WEB_SEARCH_ENGINE,
            len(query),
            bool(direct),
            len(sources),
        )
        return context, sources


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """Return the shared :class:`RAGService` singleton."""
    return RAGService(embedding_service=get_embedding_service())