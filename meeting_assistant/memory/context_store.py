"""
memory/context_store.py — Meeting context memory module.

Responsibility:
    Receive clean transcribed text segments from whisper_engine.py, maintain
    the full meeting transcript, provide a sliding context window for agents,
    and produce a ContextSnapshot dataclass that every agent reads from.

Layer position:
    transcription/whisper_engine.py  →  [THIS FILE]  →  agents/

Receives from whisper_engine.py:
    str — a clean transcribed text segment (never None — pipeline filters
          those out before calling add_segment())

Produces for agents/:
    ContextSnapshot dataclass containing:
        - full_transcript : str  — entire meeting joined as one string
        - recent_context  : str  — last max_context_chars characters
        - chunk_count     : int  — total segments added so far
        - word_count      : int  — approximate total words spoken

Design decisions:
    - ContextSnapshot is a frozen dataclass — agents cannot accidentally
      mutate the shared memory state.
    - Sliding window is computed from the tail of the full transcript,
      never from a separate buffer — single source of truth.
    - Vector store is opt-in via config. When disabled the module has
      zero extra dependencies — no sentence-transformers needed.
    - Thread safety: add_segment() and get_snapshot() are protected by
      a threading.Lock so the pipeline thread and any future async agents
      cannot corrupt the transcript list simultaneously.

Usage:
    store = ContextStore()

    store.add_segment("Alice will own the backend migration.")
    store.add_segment("Target date is end of October.")

    snapshot = store.get_snapshot()
    print(snapshot.recent_context)   # last 3000 chars
    print(snapshot.chunk_count)      # 2
    print(snapshot.word_count)       # ~12

    store.reset()                    # call between meetings
"""

import threading
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from config import settings


# ── ContextSnapshot ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ContextSnapshot:
    """
    Immutable snapshot of meeting memory at a point in time.

    Passed to every agent on each pipeline tick. Frozen so agents
    cannot accidentally modify shared memory state.

    Attributes:
        full_transcript: Complete meeting transcript joined as one string.
                         Used by storage/writer.py to save the full record.
        recent_context:  Sliding window — last max_context_chars characters.
                         This is what agents actually read for their prompts.
        chunk_count:     Total number of segments added so far.
                         Pipeline uses this to decide when to run agents.
        word_count:      Approximate total words spoken in the meeting.
    """
    full_transcript: str
    recent_context:  str
    chunk_count:     int
    word_count:      int


# ── ContextStore ───────────────────────────────────────────────────────────────

class ContextStore:
    """
    Maintains the complete memory of a meeting in progress.

    Segments are appended as they arrive from the transcription layer.
    The store provides agents with a ready-to-use ContextSnapshot on demand.

    Attributes:
        chunk_count: Number of segments added since last reset().
    """

    def __init__(self) -> None:
        # Primary storage — ordered list of all transcribed segments
        self._segments: list[str] = []

        # Protects _segments against concurrent reads/writes
        self._lock = threading.Lock()

        # Optional vector store — only initialised if enabled in config
        self._vector_store: Optional[_VectorStore] = None

        if settings.memory.enable_vector_store:
            try:
                self._vector_store = _VectorStore(
                    model_name=settings.memory.embedding_model,
                    top_k=settings.memory.top_k_retrieval,
                )
                logger.info(
                    f"Vector store enabled | "
                    f"model={settings.memory.embedding_model} | "
                    f"top_k={settings.memory.top_k_retrieval}"
                )
            except ImportError as e:
                logger.warning(
                    f"Vector store requested but could not be initialised: {e}\n"
                    "Install sentence-transformers and faiss-cpu to enable it.\n"
                    "Falling back to sliding window only."
                )
                self._vector_store = None

        logger.info(
            f"ContextStore ready | "
            f"max_context_chars={settings.memory.max_context_chars} | "
            f"vector_store={'on' if self._vector_store else 'off'}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def chunk_count(self) -> int:
        """Total number of segments added since last reset."""
        with self._lock:
            return len(self._segments)

    def add_segment(self, text: str) -> None:
        """
        Add a new transcribed text segment to memory.

        Called by the pipeline for every non-None result from
        whisper_engine.transcribe(). Do not call with empty strings
        or None — the pipeline is responsible for filtering those out.

        Args:
            text: Clean transcribed text from whisper_engine.py.
                  Must be a non-empty string.

        Raises:
            ValueError: If text is empty or not a string.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"add_segment() requires a non-empty string, got: {repr(text)}"
            )

        text = text.strip()

        with self._lock:
            self._segments.append(text)
            count = len(self._segments)

        # Embed the segment if vector store is enabled
        # Done outside the lock — embedding is slow and shouldn't block reads
        if self._vector_store is not None:
            self._vector_store.add(text)

        logger.debug(
            f"Segment added | chunk={count} | "
            f"chars={len(text)} | preview='{text[:60]}'"
        )

    def get_snapshot(self) -> ContextSnapshot:
        """
        Return a frozen snapshot of the current memory state.

        Called by the pipeline once per agent-run tick. Produces a
        ContextSnapshot with a pre-trimmed sliding context window
        ready to drop directly into agent prompts.

        Returns:
            ContextSnapshot with full_transcript, recent_context,
            chunk_count, and word_count populated.
        """
        with self._lock:
            segments = list(self._segments)  # shallow copy — release lock fast

        if not segments:
            return ContextSnapshot(
                full_transcript="",
                recent_context="",
                chunk_count=0,
                word_count=0,
            )

        full_transcript = " ".join(segments)
        recent_context  = self._build_window(segments)
        word_count      = len(full_transcript.split())

        snapshot = ContextSnapshot(
            full_transcript=full_transcript,
            recent_context=recent_context,
            chunk_count=len(segments),
            word_count=word_count,
        )

        logger.debug(
            f"Snapshot | chunks={snapshot.chunk_count} | "
            f"words={snapshot.word_count} | "
            f"context_chars={len(snapshot.recent_context)}"
        )

        return snapshot

    def retrieve(self, query: str) -> list[str]:
        """
        Retrieve the most semantically relevant segments for a query.

        Only available when enable_vector_store=True in config.
        Falls back to returning the recent context window as a single
        item if the vector store is disabled.

        Args:
            query: Natural language question or topic to search for.
                   e.g. "what deadlines were mentioned?"

        Returns:
            List of up to top_k most relevant segments as strings.
        """
        if self._vector_store is not None:
            return self._vector_store.search(query)

        # Fallback — return the recent window as one passage
        with self._lock:
            segments = list(self._segments)
        return [self._build_window(segments)] if segments else []

    def reset(self) -> None:
        """
        Clear all memory. Call between meetings — not during one.

        Clears the transcript, resets chunk count, and clears the
        vector store index if enabled.
        """
        with self._lock:
            self._segments.clear()

        if self._vector_store is not None:
            self._vector_store.clear()

        logger.info("ContextStore reset — memory cleared.")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_window(self, segments: list[str]) -> str:
        """
        Build the sliding context window from the tail of segments.

        Walks segments from newest to oldest, accumulating text until
        max_context_chars is reached, then returns the result in
        chronological order (oldest → newest).

        Args:
            segments: List of all transcript segments in order.

        Returns:
            A single string of the most recent content up to
            max_context_chars characters, space-joined.
        """
        max_chars = settings.memory.max_context_chars
        window: list[str] = []
        chars_so_far = 0

        # Walk backwards — newest segment first
        for segment in reversed(segments):
            segment_len = len(segment) + 1  # +1 for the joining space
            if chars_so_far + segment_len > max_chars:
                break
            window.append(segment)
            chars_so_far += segment_len

        # Reverse back to chronological order
        window.reverse()

        return " ".join(window)


# ── Optional vector store (sentence-transformers + faiss) ─────────────────────

class _VectorStore:
    """
    In-memory semantic search over meeting segments.

    Uses sentence-transformers to embed each segment into a dense
    vector and faiss for nearest-neighbour retrieval.

    This class is only instantiated when:
        settings.memory.enable_vector_store == True

    It is intentionally private (underscore prefix) — external code
    uses ContextStore.retrieve() instead of calling this directly.
    """

    def __init__(self, model_name: str, top_k: int) -> None:
        # These imports are guarded — only attempted when vector store
        # is enabled, so users without these packages are unaffected.
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            import numpy as np
        except ImportError as e:
            raise ImportError(
                f"Vector store requires sentence-transformers and faiss-cpu: {e}\n"
                "Install with: pip install sentence-transformers faiss-cpu"
            ) from e

        self._np = np
        self._faiss = faiss
        self._model = SentenceTransformer(model_name)
        self._top_k = top_k

        # Embedding dimension for all-MiniLM-L6-v2 is 384
        # Will be set on first add() call
        self._dim: Optional[int] = None
        self._index = None
        self._segments: list[str] = []

        logger.info(f"_VectorStore initialised | model={model_name}")

    def add(self, text: str) -> None:
        """Embed a segment and add it to the faiss index."""
        vector = self._model.encode([text], convert_to_numpy=True)

        if self._index is None:
            self._dim = vector.shape[1]
            # IndexFlatL2 — exact nearest neighbour, good for small indexes
            self._index = self._faiss.IndexFlatL2(self._dim)

        self._index.add(vector.astype(self._np.float32))
        self._segments.append(text)

    def search(self, query: str) -> list[str]:
        """Return the top_k most semantically similar segments."""
        if self._index is None or len(self._segments) == 0:
            return []

        vector = self._model.encode([query], convert_to_numpy=True)
        k = min(self._top_k, len(self._segments))
        _, indices = self._index.search(
            vector.astype(self._np.float32), k
        )

        return [
            self._segments[i]
            for i in indices[0]
            if i < len(self._segments)
        ]

    def clear(self) -> None:
        """Reset the index and segment list."""
        self._index = None
        self._segments.clear()
        logger.debug("_VectorStore index cleared.")
