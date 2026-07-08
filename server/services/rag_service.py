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

import json
import logging
import re
from functools import lru_cache

import serpapi
from sqlalchemy import text
from sqlalchemy.orm import Session

from server.config import settings
from server.models.workspace import DocumentChunk
from server.services.embedding_service import EmbeddingService, get_embedding_service
from server.services.rerank_service import get_rerank_service
from server.utils.prompt import (
    FOLLOWUP_MARKER,
    ai_assistent,
    augmented_system_prompt,
    classification_prompt,
    followup_instruction_block,
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
        top_k: int | None = None,
        document_ids: list[str],
        workspace_id: str | None = None,
    ) -> list[DocumentChunk]:
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

        if document_ids and len(document_ids) > top_k:
            top_k = min(len(document_ids), settings.RAG_MAX_CONTEXT_CHUNKS)

        query_embedding = self._embedder.embed_text(query)

        self._widen_hnsw_search(db)
        distance = DocumentChunk.embedding.cosine_distance(query_embedding)

        def _scoped():
            q = db.query(DocumentChunk).filter(DocumentChunk.user_id == str(user_id))
            if workspace_id:
                q = q.filter(DocumentChunk.workspace_id == str(workspace_id))
            return q

        candidate_k = (
            max(top_k, settings.RAG_RERANK_CANDIDATES)
            if settings.RAG_RERANK_ENABLED
            else top_k
        )

        if document_ids and len(document_ids) > 1:
            import math

            per_doc = max(1, math.ceil(candidate_k / len(document_ids)))
            rows: list[DocumentChunk] = []
            for doc_id in document_ids:
                rows.extend(
                    _scoped()
                    .filter(DocumentChunk.document_id == doc_id)
                    .order_by(distance)
                    .limit(per_doc)
                    .all()
                )
        else:
            stmt = _scoped()
            if document_ids:
                stmt = stmt.filter(DocumentChunk.document_id.in_(document_ids))
            rows = stmt.order_by(distance).limit(candidate_k).all()

        if not rows:
            return []

        if settings.RAG_RERANK_ENABLED:
            ordered = get_rerank_service().rerank(
                query,
                rows,
                top_k=len(rows),
                text_getter=lambda chunk: chunk.content or "",
            )
        else:
            ordered = rows

        final = self._ensure_document_representation(
            ordered, document_ids=document_ids, top_k=top_k
        )
        logger.debug(
            "RAG retrieve: user=%s workspace=%s attached_docs=%d candidates=%d final=%d",
            user_id,
            workspace_id,
            len(document_ids or []),
            len(rows),
            len(final),
        )
        return final

    @staticmethod
    def _ensure_document_representation(
        ordered_rows: list[DocumentChunk],
        *,
        document_ids: list[str],
        top_k: int,
    ) -> list[DocumentChunk]:
        """Pick the final ``top_k`` chunks, guaranteeing that every attached
        document contributes at least its best chunk (when several are attached).

        Without this, a global top-k lets a many-chunk document fill every slot and
        silently drop a few-chunk document (e.g. an uploaded image) from the answer
        context. Slots left after the per-document floor are filled by global rank.
        """
        if not ordered_rows:
            return []
        if not document_ids or len(document_ids) <= 1:
            return ordered_rows[:top_k]

        rank = {id(row): index for index, row in enumerate(ordered_rows)}
        best_by_doc: dict[str, DocumentChunk] = {}
        for row in ordered_rows:
            best_by_doc.setdefault(row.document_id, row)

        selected: list[DocumentChunk] = []
        selected_ids: set[int] = set()

        for doc_id in document_ids:
            row = best_by_doc.get(doc_id)
            if row is not None and id(row) not in selected_ids:
                selected.append(row)
                selected_ids.add(id(row))

        for row in ordered_rows:
            if len(selected) >= top_k:
                break
            if id(row) not in selected_ids:
                selected.append(row)
                selected_ids.add(id(row))

        selected.sort(key=lambda row: rank[id(row)])
        return selected[:top_k]

    @staticmethod
    def _widen_hnsw_search(db: Session) -> None:
        """Raise ``hnsw.ef_search`` for the current transaction (PostgreSQL only).

        HNSW applies the ``WHERE`` filter (user/workspace/document scope) only
        *after* pulling its ``ef_search`` nearest neighbours from the whole table,
        so a small default window (40) can miss a specific document's chunks when
        other documents sit nearer in vector space — the cause of "first upload
        returns nothing, re-upload works". A wider window restores recall for
        filtered searches. ``iterative_scan`` (pgvector >= 0.8) makes it exact;
        it's attempted in a savepoint so older versions degrade gracefully.
        """
        try:
            if db.bind is None or db.bind.dialect.name != "postgresql":
                return
        except Exception:
            return

        try:
            db.execute(
                text("SET LOCAL hnsw.ef_search = :ef").bindparams(
                    ef=int(settings.RAG_HNSW_EF_SEARCH)
                )
            )
        except Exception:
            logger.debug("rag: could not set hnsw.ef_search", exc_info=True)
            return

        try:
            with db.begin_nested():
                db.execute(
                    text("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
                )
        except Exception:
            logger.debug("rag: hnsw.iterative_scan unavailable", exc_info=True)

    @staticmethod
    def build_context(chunks: list[DocumentChunk]) -> str:
        """Concatenate chunk contents into a single context string for the LLM."""
        return "\n\n---\n\n".join(chunk.content for chunk in chunks if chunk.content)

    @staticmethod
    def build_messages(
        *,
        history: list[dict],
        question: str,
        retrieved_context: str | None = None,
        web_context: str | None = None,
    ) -> list[dict]:
        """Prompt generation.

        Grounds the answer in whatever context is available:
          * ``retrieved_context`` — RAG document chunks;
          * ``web_context``       — web search results.
        When either (or both) is present the augmented system prompt is used;
        with neither, the plain-assistant prompt is used. Previous chat history is
        always included for continuity, and the current question is appended last.

        The follow-up-questions instruction is appended to the system prompt so
        every answer is followed by a parseable list of suggested follow-ups
        (see :meth:`split_answer_and_followups`).
        """
        if retrieved_context is not None or web_context is not None:
            system_prompt = augmented_system_prompt(
                document_context=retrieved_context, web_context=web_context
            )
        else:
            system_prompt = ai_assistent()

        system_prompt += followup_instruction_block()

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": question})
        return messages

    @staticmethod
    def split_answer_and_followups(text: str) -> tuple[str, list[str]]:
        """Split a raw generation into ``(answer, follow_up_questions)``.

        The model is instructed to append ``FOLLOWUP_MARKER`` followed by a JSON
        array of questions. This strips that block from the answer and parses the
        questions. If the marker is missing or the block is unparseable, the full
        text is returned as the answer with an empty follow-up list — never
        placeholder text.
        """
        if not text or FOLLOWUP_MARKER not in text:
            return (text or "").strip(), []

        answer, _, tail = text.partition(FOLLOWUP_MARKER)

        questions: list[str] = []
        match = re.search(r"\[.*\]", tail, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    questions = [
                        str(q).strip() for q in parsed if str(q).strip()
                    ]
            except (ValueError, TypeError):
                questions = []

        return answer.strip(), questions[:4]

    @staticmethod
    def source_file_names(chunks: list[DocumentChunk]) -> list[str]:
        """Return the distinct source file names behind ``chunks`` (order-stable)."""
        return list(dict.fromkeys(c.file_name for c in chunks if c.file_name))

    def retrieve_context(
        self,
        db: Session,
        *,
        query: str,
        user_id: str,
        top_k: int | None = None,
        document_ids: list[str],
        workspace_id: str | None = None,
    ) -> tuple[str, list[str]]:
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
    def generate_response(client, *, messages: list[dict], model: str) -> str:
        """LLM response generation using streaming, returning the full text.

        Streaming is preserved (``stream=True``) exactly as the existing chat flow;
        the chunks are aggregated into the final response string.
        """
        ai_response = ""
        completion = client.chat.completions.create(
            model=model, messages=messages, max_tokens=1024, stream=True, temperature=0
        )
        for chunk in completion:
            if (
                chunk.choices
                and chunk.choices[0].delta
                and chunk.choices[0].delta.content
            ):
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
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": classification_prompt()},
                    {"role": "user", "content": f'Question:\n"{question}"\nAnswer:'},
                ],
                max_tokens=1024,
                temperature=0,
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
    def _format_scalar_fields(data: dict, skip: tuple[str, ...]) -> str:
        """Flatten the scalar (str/number) fields of a SerpAPI widget to text.

        Nested objects, images and link metadata (in ``skip``) are ignored so the
        digest stays compact and LLM-friendly.
        """
        parts: list[str] = []
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
        sections: list[str] = []
        _skip = (
            "type",
            "thumbnail",
            "link",
            "displayed_link",
            "images",
            "list",
            "table",
            "source",
            "serpapi_link",
            "people_also_search_for",
        )

        answer_box = results.get("answer_box")
        if isinstance(answer_box, dict):
            text = cls._format_scalar_fields(answer_box, skip=_skip)
            if text:
                sections.append("Direct answer:\n" + text)

        knowledge_graph = results.get("knowledge_graph")
        if isinstance(knowledge_graph, dict):
            text = cls._format_scalar_fields(
                knowledge_graph,
                skip=_skip + ("kgmid", "knowledge_graph_search_link"),
            )
            if text:
                sections.append("Knowledge graph:\n" + text)

        return "\n\n".join(sections)

    @classmethod
    def web_search(
        cls, query: str, max_results: int | None = None
    ) -> tuple[str, list[dict]]:
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
            logger.exception(
                "rag.web_search: search failed for query_len=%d", len(query)
            )
            return "", []

        sections: list[str] = []

        direct = cls._extract_direct_answer(results)
        if direct:
            sections.append(direct)

        organic = (results.get("organic_results") or [])[:max_results]
        lines: list[str] = []
        sources: list[dict] = []
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
