"""Cross-encoder re-ranking service.

Second stage of a two-stage retrieval pipeline. The vector search (stage 1)
retrieves a broad candidate pool by embedding cosine similarity, which is fast
but only approximates relevance. This service (stage 2) re-scores each candidate
against the query with a cross-encoder — which reads the query and passage
*together* — and returns the most relevant ones.

The model is loaded lazily and cached as a process-wide singleton. Loading or
scoring failures degrade gracefully: the caller gets the original (vector-ranked)
order back, so retrieval never breaks because of the re-ranker.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Callable, List, Optional, Sequence, TypeVar

from server.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RerankService:
    """Re-order candidates by cross-encoder relevance to a query."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or settings.RERANK_MODEL
        self._model = None
        self._load_failed = False

    @property
    def model(self):
        """Lazily load the CrossEncoder, caching success/failure.

        Returns ``None`` (once) if the model cannot be loaded, so callers can fall
        back to the vector order instead of raising on every request.
        """
        if self._model is None and not self._load_failed:
            try:
                from sentence_transformers import CrossEncoder

                logger.info("rerank: loading cross-encoder model %s", self.model_name)
                self._model = CrossEncoder(self.model_name)
            except Exception:
                self._load_failed = True
                logger.exception(
                    "rerank: failed to load model %s; re-ranking disabled",
                    self.model_name,
                )
        return self._model

    def rerank(
        self,
        query: str,
        items: Sequence[T],
        *,
        top_k: int,
        text_getter: Callable[[T], str],
    ) -> List[T]:
        """Return the ``top_k`` ``items`` most relevant to ``query``.

        ``text_getter`` extracts the text to score from each item, keeping this
        service decoupled from any particular row/DTO shape. On an unavailable
        model or a scoring error, the original order (truncated to ``top_k``) is
        returned so retrieval still succeeds.
        """
        items = list(items)
        if not items:
            return []

        model = self.model
        if model is None:
            return items[:top_k]

        try:
            pairs = [(query, text_getter(item) or "") for item in items]
            scores = model.predict(pairs)
            ranked = sorted(
                zip(items, scores), key=lambda pair: pair[1], reverse=True
            )
            top = [item for item, _ in ranked[:top_k]]
            logger.info(
                "rerank: query_len=%d candidates=%d -> kept=%d",
                len(query),
                len(items),
                len(top),
            )
            return top
        except Exception:
            logger.exception("rerank: scoring failed; falling back to vector order")
            return items[:top_k]


@lru_cache(maxsize=1)
def get_rerank_service() -> RerankService:
    """Return the shared :class:`RerankService` singleton."""
    return RerankService()
