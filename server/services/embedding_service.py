"""OpenAI-backed embedding service.

Single responsibility: turn text into embedding vectors using the OpenAI
Embeddings API. Contains no business logic (no chunking, no persistence, no
retrieval). Exposes both synchronous and asynchronous methods so it can be
consumed from sync request handlers and async ingestion paths alike.
"""

from __future__ import annotations

import logging
import asyncio
from functools import lru_cache

from openai import AsyncOpenAI, OpenAI, OpenAIError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from server.services.audit_logger import audit_context
from server.config import settings

logger = logging.getLogger(__name__)

# text-embedding-3-* models accept a ``dimensions`` argument to shorten the
# output vector; older models (e.g. text-embedding-ada-002) do not.
_MODELS_SUPPORTING_DIMENSIONS = ("text-embedding-3",)

_sync_client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
)

_async_client = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY,
)


class EmbeddingService:
    """Generate embeddings with the OpenAI Embeddings API.

    The OpenAI clients are created lazily and reused. Requests are automatically
    batched (``settings.EMBEDDING_BATCH_SIZE``) and retried with exponential
    backoff on transient API errors.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        batch_size: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        self._api_key = api_key or settings.OPENAI_API_KEY
        self._base_url = base_url or settings.OPENAI_BASE_URL or None
        self.model = model or settings.EMBEDDING_MODEL
        self.dimensions = dimensions or settings.EMBEDDING_DIMENSIONS
        self.batch_size = max(1, batch_size or settings.EMBEDDING_BATCH_SIZE)
        self._max_retries = (
            max_retries if max_retries is not None else settings.EMBEDDING_MAX_RETRIES
        )

        self._client: OpenAI | None = None
        self._aclient: AsyncOpenAI | None = None

    # -- clients -----------------------------------------------------------

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self._require_api_key(),
                max_retries=self._max_retries,
            )
        return self._client

    @property
    def aclient(self) -> AsyncOpenAI:
        if self._aclient is None:
            self._aclient = AsyncOpenAI(
                api_key=self._require_api_key(),
                max_retries=self._max_retries,
            )
        return self._aclient

    def _require_api_key(self) -> str:
        if not self._api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured; set it in the environment."
            )
        return self._api_key

    # -- request helpers ---------------------------------------------------

    def _request_kwargs(self, batch: list[str]) -> dict:
        kwargs: dict = {"model": self.model, "input": batch}
        if self.dimensions and self.model.startswith(_MODELS_SUPPORTING_DIMENSIONS):
            kwargs["dimensions"] = self.dimensions
        return kwargs

    @staticmethod
    def _sanitize(texts: list[str]) -> list[str]:
        # The API rejects empty strings; substitute a single space so indices
        # stay aligned with the caller's input.
        return [t if (t and t.strip()) else " " for t in texts]

    # -- synchronous API ---------------------------------------------------

    def embed_text(self, text: str) -> list[float]:
        """Embed a single string and return its vector."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings, batching requests transparently."""
        if not texts:
            return []
        clean = self._sanitize(texts)
        vectors: list[list[float]] = []
        for start in range(0, len(clean), self.batch_size):
            vectors.extend(
                self._embed_batch_sync(clean[start : start + self.batch_size])
            )
        return vectors

    @retry(
        retry=retry_if_exception_type(OpenAIError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _embed_batch_sync(self, batch: list[str]) -> list[list[float]]:

        with audit_context(
            action="EMBEDDING_GENERATION",
            metadata={
                "model": self.model,
                "batch_size": len(batch),
                "async": False,
            },
        ):
            try:
                # response = self.client.embeddings.create(**self._request_kwargs(batch))
                response = _sync_client.embeddings.create(
                    model="text-embedding-3-small", input=batch
                )
            except OpenAIError as exc:
                logger.exception(
                    "OpenAI embedding request failed (model=%s, batch_size=%d)",
                    self.model,
                    len(batch),
                )
                raise
            return [item.embedding for item in response.data]

    # -- asynchronous API --------------------------------------------------

    async def aembed_text(self, text: str) -> list[float]:
        return (await self.aembed_texts([text]))[0]

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        clean = self._sanitize(texts)
        tasks = []
        for start in range(0, len(clean), self.batch_size):
            tasks.append(
                self._embed_batch_async(clean[start : start + self.batch_size])
            )

        results = await asyncio.gather(*tasks)
        vectors: list[list[float]] = []
        for batch_result in results:
            vectors.extend(batch_result)

        return vectors

    @retry(
        retry=retry_if_exception_type(OpenAIError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _embed_batch_async(self, batch: list[str]) -> list[list[float]]:

        with audit_context(
            action="EMBEDDING_GENERATION",
            metadata={
                "model": self.model,
                "batch_size": len(batch),
                "async": True,
            },
        ):
            try:
                # response = await self.aclient.embeddings.create(**self._request_kwargs(batch))
                response = await _async_client.embeddings.create(
                    model="text-embedding-3-small", input=batch
                )
            except OpenAIError as exc:
                logger.exception(
                    "OpenAI async embedding request failed (model=%s, batch_size=%d)",
                    self.model,
                    len(batch),
                )
                raise
            return [item.embedding for item in response.data]


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Return the shared :class:`EmbeddingService` singleton."""
    return EmbeddingService()
