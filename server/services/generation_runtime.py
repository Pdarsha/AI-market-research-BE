"""Keeps an AI generation alive independently of the client connection.

The chat endpoint streams SSE frames produced by a generator. Starlette only pumps
that generator while the client is reading it, so a browser refresh (client
disconnect) used to stop the LLM loop mid-way AND skip the persistence that runs
after it -- the response was lost.

This module inverts that ownership:

    produce()  ->  worker thread  ->  pub/sub  ->  relay generator  ->  client
                   (runs to completion,            (ends when client
                    always persists)                disconnects)

A disconnect ends the *relay* only. The worker keeps generating and persisting.
The single way to stop a generation is an explicit cancel() -- the Stop button --
which sets an Event the producer checks between tokens.

If a browser refreshes or reconnects, it can re-subscribe to the in-flight
generation via the generation stream endpoint, resuming token streaming from its
last received chunk without restarting generation.
"""

from datetime import datetime, timezone
import logging
import queue
import threading
import uuid
from typing import Callable, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

# Sentinel pushed by the worker to tell relays the generation is finished.
_DONE = object()


class GenerationCancelled(Exception):
    """Raised inside the pipeline when the user pressed Stop.

    The chat pipeline does a lot of cancellable work -- query rewriting, intent
    classification, retrieval, web search, deep research -- before the first
    token is ever requested from the LLM. Each of those stages calls
    ``handle.raise_if_cancelled()``, which raises this so the pipeline unwinds
    instead of running on to call the model.
    """

    def __init__(self, generation_id: str, stage: str):
        self.generation_id = generation_id
        self.stage = stage
        super().__init__(f"Generation {generation_id} cancelled at stage '{stage}'")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class GenerationHandle:
    """Channel and state manager for one AI generation, addressed by generation_id."""

    def __init__(
        self,
        generation_id: str,
        user_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        user_message_id: Optional[str] = None,
        assistant_message_id: Optional[str] = None,
        mode: Optional[str] = "auto",
    ):
        self.generation_id = generation_id
        self.user_id = str(user_id) if user_id is not None else None
        self.chat_id = str(chat_id) if chat_id is not None else None
        self.user_message_id = user_message_id
        self.assistant_message_id = assistant_message_id
        self.mode = mode

        self.status = "generating"  # "generating" | "completed" | "failed" | "cancelled"
        self.created_at = _utc_now_iso()
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.error: Optional[str] = None

        self.tokens: List[str] = []
        self.frames: List[str] = []
        self.current_text: str = ""
        self.final_data: Optional[dict] = None

        self._cancelled = threading.Event()
        self._done = threading.Event()
        self._subscribers: List[queue.Queue] = []
        self._lock = threading.Lock()

        logger.info(
            "[GENERATION_CREATED] generation_id=%s chat_id=%s user_id=%s",
            self.generation_id,
            self.chat_id,
            self.user_id,
        )

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    @property
    def is_done(self) -> bool:
        return self._done.is_set()

    def cancel(self) -> None:
        with self._lock:
            self._cancelled.set()
            self.status = "cancelled"
            self.completed_at = _utc_now_iso()
        logger.info(
            "[GENERATION_CANCELLED] generation_id=%s chat_id=%s user_id=%s",
            self.generation_id,
            self.chat_id,
            self.user_id,
        )

    def raise_if_cancelled(self, stage: str) -> None:
        """Abort the pipeline here if the user has pressed Stop.

        Called at every stage boundary so an explicit cancel takes effect
        without waiting for the current stage -- or the LLM call after it -- to
        finish.
        """
        if self._cancelled.is_set():
            logger.info(
                "[GENERATION_CANCEL_CHECKPOINT] generation_id=%s chat_id=%s stage=%s",
                self.generation_id,
                self.chat_id,
                stage,
            )
            raise GenerationCancelled(self.generation_id, stage)

    def mark_cancelled_done(self) -> None:
        """Close out a generation cancelled before its stream ever started.

        Keeps the terminal state at "cancelled" and releases any subscriber that
        is already waiting on the relay.
        """
        with self._lock:
            self.status = "cancelled"
            self.completed_at = _utc_now_iso()
            self._done.set()
            for sub_queue in list(self._subscribers):
                try:
                    sub_queue.put_nowait(_DONE)
                except Exception:
                    pass
        logger.info(
            "[GENERATION_CANCELLED_FINAL] generation_id=%s chat_id=%s frames=%d",
            self.generation_id,
            self.chat_id,
            len(self.frames),
        )

    def mark_started(self) -> None:
        with self._lock:
            if not self.started_at:
                self.started_at = _utc_now_iso()
                self.status = "generating"
        logger.info(
            "[GENERATION_STARTED] generation_id=%s chat_id=%s user_id=%s",
            self.generation_id,
            self.chat_id,
            self.user_id,
        )

    def record_token(self, token: str) -> None:
        with self._lock:
            self.tokens.append(token)
            self.current_text += token
            token_count = len(self.tokens)
        if token_count % 50 == 0:
            logger.info(
                "[GENERATION_PROGRESS] generation_id=%s tokens=%d text_len=%d",
                self.generation_id,
                token_count,
                len(self.current_text),
            )

    def push_frame(self, frame: str) -> None:
        with self._lock:
            self.frames.append(frame)
            for sub_queue in list(self._subscribers):
                try:
                    sub_queue.put_nowait(frame)
                except Exception:
                    pass

    def mark_completed(
        self,
        final_data: Optional[dict] = None,
        assistant_message_id: Optional[str] = None,
        response_text: Optional[str] = None,
    ) -> None:
        with self._lock:
            self.status = "completed" if not self._cancelled.is_set() else "cancelled"
            if final_data is not None:
                self.final_data = final_data
            if assistant_message_id:
                self.assistant_message_id = assistant_message_id
            if response_text is not None:
                self.current_text = response_text
            self.completed_at = _utc_now_iso()
            self._done.set()
            for sub_queue in list(self._subscribers):
                try:
                    sub_queue.put_nowait(_DONE)
                except Exception:
                    pass
        logger.info(
            "[GENERATION_COMPLETED] generation_id=%s chat_id=%s status=%s frames=%d",
            self.generation_id,
            self.chat_id,
            self.status,
            len(self.frames),
        )

    def mark_failed(self, error: str) -> None:
        with self._lock:
            self.status = "failed"
            self.error = error
            self.completed_at = _utc_now_iso()
            self._done.set()
            for sub_queue in list(self._subscribers):
                try:
                    sub_queue.put_nowait(_DONE)
                except Exception:
                    pass
        logger.error(
            "[GENERATION_FAILED] generation_id=%s chat_id=%s error=%s",
            self.generation_id,
            self.chat_id,
            error,
        )

    def subscribe(self, last_chunk_index: int = 0) -> Iterator[str]:
        """Subscribe to this generation, catching up from `last_chunk_index`."""
        sub_queue: "queue.Queue" = queue.Queue()
        with self._lock:
            start_idx = max(0, int(last_chunk_index))
            catch_up = list(self.frames[start_idx:])
            if not self._done.is_set():
                self._subscribers.append(sub_queue)
            else:
                sub_queue.put_nowait(_DONE)

        for frame in catch_up:
            yield frame

        if self._done.is_set() and sub_queue.empty():
            return

        try:
            while True:
                frame = sub_queue.get()
                if frame is _DONE:
                    return
                yield frame
        finally:
            with self._lock:
                if sub_queue in self._subscribers:
                    self._subscribers.remove(sub_queue)

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "generation_id": self.generation_id,
                "chat_id": self.chat_id,
                "user_id": self.user_id,
                "user_message_id": self.user_message_id,
                "assistant_message_id": self.assistant_message_id,
                "status": self.status,
                "mode": self.mode,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "content": self.current_text,
                "chunk_count": len(self.frames),
                "error": self.error,
                "final_data": self.final_data,
            }


_REGISTRY: Dict[str, GenerationHandle] = {}
_LOCK = threading.Lock()


def register(
    user_id: Optional[str] = None,
    chat_id: Optional[str] = None,
    generation_id: Optional[str] = None,
    user_message_id: Optional[str] = None,
    assistant_message_id: Optional[str] = None,
    mode: Optional[str] = "auto",
) -> GenerationHandle:
    """Create and track a generation handle."""
    gen_id = generation_id or str(uuid.uuid4())
    handle = GenerationHandle(
        generation_id=gen_id,
        user_id=user_id,
        chat_id=chat_id,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
        mode=mode,
    )
    with _LOCK:
        _REGISTRY[handle.generation_id] = handle
    return handle


def get(generation_id: str) -> Optional[GenerationHandle]:
    with _LOCK:
        return _REGISTRY.get(generation_id)


def unregister(generation_id: str) -> None:
    with _LOCK:
        _REGISTRY.pop(generation_id, None)


def active_for_chat(chat_id: str, user_id: Optional[str] = None) -> list:
    """Handles still actively generating for a chat."""
    with _LOCK:
        results = []
        for h in _REGISTRY.values():
            if str(h.chat_id) == str(chat_id) and h.status == "generating":
                if user_id is None or (h.user_id is not None and str(h.user_id) == str(user_id)):
                    results.append(h)
        return results


def get_active_for_chat(chat_id: str, user_id: Optional[str] = None) -> Optional[GenerationHandle]:
    """Return the active GenerationHandle for a chat if one exists."""
    actives = active_for_chat(chat_id, user_id=user_id)
    return actives[-1] if actives else None


def discard_pending_for_chat(chat_id: str, user_id: Optional[str] = None) -> int:
    """Drop handles for a chat whose worker never started.

    The generation is registered early, before the pipeline can still fail or
    raise (validation errors, provider outages). Those handles have no worker to
    unregister them, so they would linger in the registry and make a dead turn
    look like an in-flight one to `/generations/active`.
    """
    removed = 0
    with _LOCK:
        for gen_id, handle in list(_REGISTRY.items()):
            if str(handle.chat_id) != str(chat_id):
                continue
            if user_id is not None and handle.user_id is not None and str(handle.user_id) != str(user_id):
                continue
            if handle.started_at is None:
                _REGISTRY.pop(gen_id, None)
                removed += 1
    if removed:
        logger.info(
            "[GENERATION_DISCARDED] chat_id=%s handles=%d (never started)",
            chat_id,
            removed,
        )
    return removed


def cancel_for_chat(chat_id: str, user_id: Optional[str] = None) -> bool:
    """Cancel any in-flight generation for a chat. Returns True if any handle was cancelled."""
    actives = active_for_chat(chat_id, user_id=user_id)
    cancelled = False
    for handle in actives:
        handle.cancel()
        cancelled = True
    return cancelled


def cancel(generation_id: str, user_id: Optional[str] = None) -> bool:
    """Explicitly cancel one generation. Returns True if a live handle was cancelled.

    `user_id`, when given, is enforced: a user can never cancel another user's
    generation.
    """
    if not generation_id or generation_id in ("active", "current", "latest"):
        return False
    handle = get(generation_id)
    if handle is None:
        return False
    if user_id is not None and handle.user_id is not None and str(handle.user_id) != str(user_id):
        logger.warning(
            "[GENERATION] Refused cross-user cancel of %s by user %s",
            generation_id,
            user_id,
        )
        return False
    handle.cancel()
    return True


def detached_stream(
    produce: Callable[[], Iterator[str]],
    handle: GenerationHandle,
) -> Iterator[str]:
    """Run `produce()` on a worker thread; relay its frames to the client.

    The worker always drains `produce()` to completion, so the generation and its
    persistence survive the client disconnecting. The returned generator is what
    Starlette iterates; when the client vanishes it is closed, which stops the
    relay without touching the worker.
    """
    client_gone = threading.Event()

    def _worker() -> None:
        handle.mark_started()
        try:
            for frame in produce():
                handle.push_frame(frame)
            if not handle.is_done:
                handle.mark_completed()
        except Exception as exc:
            handle.mark_failed(str(exc))
            logger.exception("[GENERATION_FAILED] generation_id=%s", handle.generation_id)
        finally:
            unregister(handle.generation_id)

    worker = threading.Thread(
        target=_worker,
        name=f"generation-{handle.generation_id[:8]}",
        daemon=False,  # never killed on request teardown
    )
    worker.start()

    logger.info(
        "[GENERATION_STREAM_CONNECTED] generation_id=%s chat_id=%s",
        handle.generation_id,
        handle.chat_id,
    )

    def _relay() -> Iterator[str]:
        try:
            for frame in handle.subscribe(last_chunk_index=0):
                yield frame
        finally:
            if not client_gone.is_set():
                client_gone.set()
                logger.info(
                    "[GENERATION_STREAM_DISCONNECTED] generation_id=%s (worker_alive=%s)",
                    handle.generation_id,
                    worker.is_alive(),
                )

    return _relay()
