"""Centralized, resilient OpenAI and LLM client management.

Provides:
- Shared connection-pooled OpenAI & AsyncOpenAI client factories.
- Configured HTTPTransport with automated connection/DNS retries and keepalive connection pooling.
- Automatic support for OPENAI_BASE_URL and custom endpoints.
- Resilient execution helpers with exponential backoff for transient connection errors,
  DNS resolution glitches, rate limits, and model compatibility fallbacks.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import httpx
import openai
from openai import AsyncOpenAI, OpenAI

from server.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Transient exceptions that should be retried automatically
RETRYABLE_EXCEPTIONS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.NetworkError,
    ConnectionError,
    TimeoutError,
)

_global_sync_client: Optional[OpenAI] = None
_global_async_client: Optional[AsyncOpenAI] = None
_global_http_client: Optional[httpx.Client] = None
_global_async_http_client: Optional[httpx.AsyncClient] = None


def get_http_client(
    timeout: float = 120.0,
    connect_timeout: float = 15.0,
    retries: int = 3,
    max_keepalive: int = 50,
    max_connections: int = 100,
) -> httpx.Client:
    """Create or return shared connection-pooled httpx.Client with transport-level retries."""
    global _global_http_client
    if _global_http_client is None or _global_http_client.is_closed:
        transport = httpx.HTTPTransport(retries=retries)
        _global_http_client = httpx.Client(
            transport=transport,
            timeout=httpx.Timeout(timeout=timeout, connect=connect_timeout),
            limits=httpx.Limits(
                max_keepalive_connections=max_keepalive,
                max_connections=max_connections,
                keepalive_expiry=60.0,
            ),
        )
    return _global_http_client


def get_async_http_client(
    timeout: float = 120.0,
    connect_timeout: float = 15.0,
    retries: int = 3,
    max_keepalive: int = 50,
    max_connections: int = 100,
) -> httpx.AsyncClient:
    """Create or return shared connection-pooled httpx.AsyncClient with transport-level retries."""
    global _global_async_http_client
    if _global_async_http_client is None or _global_async_http_client.is_closed:
        transport = httpx.AsyncHTTPTransport(retries=retries)
        _global_async_http_client = httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(timeout=timeout, connect=connect_timeout),
            limits=httpx.Limits(
                max_keepalive_connections=max_keepalive,
                max_connections=max_connections,
                keepalive_expiry=60.0,
            ),
        )
    return _global_async_http_client


def get_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_retries: int = 3,
    timeout: float = 120.0,
) -> OpenAI:
    """Get a resilient OpenAI client with connection pooling, retries, and base_url support."""
    global _global_sync_client

    key = api_key or settings.OPENAI_API_KEY
    b_url = base_url or settings.OPENAI_BASE_URL or None

    # If default parameters match, reuse global singleton client
    if not api_key and not base_url and _global_sync_client is not None:
        return _global_sync_client

    http_client = get_http_client(timeout=timeout)
    client_kwargs: Dict[str, Any] = {
        "api_key": key or "missing-key",
        "http_client": http_client,
        "max_retries": max_retries,
    }
    if b_url:
        client_kwargs["base_url"] = b_url

    client = OpenAI(**client_kwargs)
    if not api_key and not base_url:
        _global_sync_client = client
    return client


def get_async_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_retries: int = 3,
    timeout: float = 120.0,
) -> AsyncOpenAI:
    """Get a resilient AsyncOpenAI client with connection pooling, retries, and base_url support."""
    global _global_async_client

    key = api_key or settings.OPENAI_API_KEY
    b_url = base_url or settings.OPENAI_BASE_URL or None

    if not api_key and not base_url and _global_async_client is not None:
        return _global_async_client

    http_client = get_async_http_client(timeout=timeout)
    client_kwargs: Dict[str, Any] = {
        "api_key": key or "missing-key",
        "http_client": http_client,
        "max_retries": max_retries,
    }
    if b_url:
        client_kwargs["base_url"] = b_url

    client = AsyncOpenAI(**client_kwargs)
    if not api_key and not base_url:
        _global_async_client = client
    return client


def execute_with_llm_retry(
    func: Callable[[], T],
    *,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: float = 0.25,
    operation_name: str = "LLM call",
) -> T:
    """Execute a synchronous LLM call with exponential backoff and jitter for transient connection/API errors."""
    attempt = 0
    while True:
        try:
            return func()
        except Exception as e:
            attempt += 1
            err_str = str(e)
            is_retryable = isinstance(e, RETRYABLE_EXCEPTIONS) or any(
                term in err_str.lower()
                for term in [
                    "connection error",
                    "temporary failure in name resolution",
                    "failed to resolve",
                    "connecterror",
                    "timed out",
                    "timeout",
                    "rate limit",
                    "503",
                    "502",
                    "504",
                    "429",
                ]
            )
            if not is_retryable or attempt >= max_attempts:
                logger.error(
                    f"[{operation_name}] Failed after {attempt} attempt(s): {err_str}",
                    exc_info=(attempt >= max_attempts),
                )
                raise e

            sleep_time = (initial_delay * (backoff_factor ** (attempt - 1))) + random.uniform(
                0, jitter
            )
            logger.warning(
                f"[{operation_name}] Attempt {attempt}/{max_attempts} failed with transient error: {err_str}. Retrying in {sleep_time:.2f}s..."
            )
            time.sleep(sleep_time)


async def execute_with_async_llm_retry(
    func: Callable[[], Any],
    *,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: float = 0.25,
    operation_name: str = "Async LLM call",
) -> Any:
    """Execute an asynchronous LLM call with exponential backoff and jitter for transient connection/API errors."""
    attempt = 0
    while True:
        try:
            res = func()
            if asyncio.iscoroutine(res):
                return await res
            return res
        except Exception as e:
            attempt += 1
            err_str = str(e)
            is_retryable = isinstance(e, RETRYABLE_EXCEPTIONS) or any(
                term in err_str.lower()
                for term in [
                    "connection error",
                    "temporary failure in name resolution",
                    "failed to resolve",
                    "connecterror",
                    "timed out",
                    "timeout",
                    "rate limit",
                    "503",
                    "502",
                    "504",
                    "429",
                ]
            )
            if not is_retryable or attempt >= max_attempts:
                logger.error(
                    f"[{operation_name}] Failed after {attempt} attempt(s): {err_str}",
                    exc_info=(attempt >= max_attempts),
                )
                raise e

            sleep_time = (initial_delay * (backoff_factor ** (attempt - 1))) + random.uniform(
                0, jitter
            )
            logger.warning(
                f"[{operation_name}] Attempt {attempt}/{max_attempts} failed with transient error: {err_str}. Retrying in {sleep_time:.2f}s..."
            )
            await asyncio.sleep(sleep_time)
