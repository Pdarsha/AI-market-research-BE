"""LLM-based re-ranking service.

Second stage of a two-stage retrieval pipeline. The vector search (stage 1)
retrieves a broad candidate pool by embedding cosine similarity, which is fast
but only approximates relevance. This service (stage 2) re-scores those
candidates against the query with an LLM (listwise ranking, à la RankGPT) and
returns the most relevant ones.

It reuses the same OpenAI-compatible endpoint (OpenRouter) as the rest of the app
with ``settings.RERANK_MODEL`` (e.g. ``openai/gpt-4.1-mini``) — no HuggingFace /
local model download. Any failure (API error or unparseable output) degrades
gracefully: the caller gets the original vector-ranked order back, so retrieval
never breaks because of the re-ranker.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from typing import Callable, Sequence, TypeVar

from openai import OpenAI

from server.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Cap the passage text sent to the ranker — a relevance judgement doesn't need the
# full chunk, and this keeps the prompt (and cost/latency) bounded.
_MAX_PASSAGE_CHARS = 600


class RerankService:
    """Re-order candidates by LLM-judged relevance to a query."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.RERANK_MODEL
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        """Lazily create the OpenAI-compatible client (OpenRouter), cached."""
        if self._client is None:
            self._client = OpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL or "https://openrouter.ai/api/v1",
            )
        return self._client

    def rerank(
        self,
        query: str,
        items: Sequence[T],
        *,
        top_k: int,
        text_getter: Callable[[T], str],
    ) -> list[T]:
        """Return the ``top_k`` ``items`` most relevant to ``query``.

        ``text_getter`` extracts the text to score from each item, keeping this
        service decoupled from any particular row/DTO shape. On an API error or an
        unparseable model response, the original order (truncated to ``top_k``) is
        returned so retrieval still succeeds.
        """
        items = list(items)
        if len(items) <= 1:
            return items[:top_k]

        # Number the passages so the model can refer to them by index.
        passages = []
        for index, item in enumerate(items, start=1):
            text = (text_getter(item) or "").strip().replace("\n", " ")
            passages.append(f"[{index}] {text[:_MAX_PASSAGE_CHARS]}")
        passages_block = "\n".join(passages)

        system_prompt = (
            "You are a search-result re-ranking assistant. Given a user query and a "
            "list of numbered passages, rank the passages by how well they answer the "
            "query. Respond with ONLY a JSON array of passage numbers ordered from "
            "most to least relevant, e.g. [3, 1, 2]. Do not include any other text."
        )
        user_prompt = (
            f"Query:\n{query}\n\n"
            f"Passages:\n{passages_block}\n\n"
            f"Return the top {top_k} passage numbers as a JSON array."
        )

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=100,
            )
            content = (completion.choices[0].message.content or "").strip()
        except Exception:
            logger.exception("rerank: LLM ranking failed; falling back to vector order")
            return items[:top_k]

        order = self._parse_ranking(content, n=len(items))
        if not order:
            logger.warning(
                "rerank: could not parse ranking from model output; using vector order"
            )
            return items[:top_k]

        # Map the 1-indexed ranking back to items, then append any candidates the
        # model omitted (so we never silently drop results), and keep top_k.
        ranked = [items[i - 1] for i in order]
        ranked_set = set(order)
        ranked.extend(
            item for i, item in enumerate(items, start=1) if i not in ranked_set
        )
        top = ranked[:top_k]
        logger.info(
            "rerank: model=%s candidates=%d -> kept=%d",
            self.model_name,
            len(items),
            len(top),
        )
        return top

    @staticmethod
    def _parse_ranking(content: str, *, n: int) -> list[int]:
        """Extract an ordered list of valid 1..n passage indices from ``content``.

        Accepts a clean JSON array, a JSON array embedded in prose, or a bare list
        of numbers. Out-of-range and duplicate indices are dropped.
        """
        match = re.search(r"\[[^\]]*\]", content)
        raw = match.group(0) if match else content
        try:
            candidates = [int(x) for x in json.loads(raw)]
        except Exception:
            candidates = [int(x) for x in re.findall(r"\d+", raw)]

        indices: list[int] = []
        for idx in candidates:
            if 1 <= idx <= n and idx not in indices:
                indices.append(idx)
        return indices


@lru_cache(maxsize=1)
def get_rerank_service() -> RerankService:
    """Return the shared :class:`RerankService` singleton."""
    return RerankService()
