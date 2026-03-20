"""
transcription/whisper_engine.py — Speech-to-text using OpenAI Whisper (local).

Responsibility:
    Accept a raw audio chunk from audio/recorder.py, transcribe it to text
    using a locally running Whisper model, filter out silence hallucinations,
    and return a clean text string (or None if the chunk had no real speech).

Position in pipeline:
    audio/recorder.py  →  [numpy float32 array]  →  whisper_engine.py
    whisper_engine.py  →  [str | None]            →  memory/context_store.py

Key design decisions:
    - Model is loaded ONCE at construction time, not on every transcribe() call.
      Loading whisper.base takes ~1-2 seconds. Reloading per chunk would stall
      the pipeline completely.
    - Returns None on silence/hallucinations instead of raising an exception.
      The pipeline simply skips None results — no crash, no interruption.
    - All config is read from config.py — no hardcoded values here.

Usage:
    engine = WhisperEngine()           # loads model once (~1-2s)

    text = engine.transcribe(chunk)    # chunk = numpy float32 array from recorder
    if text:
        # real speech detected — pass to memory
    else:
        # silence or hallucination — skip this chunk
"""

from typing import Optional

import numpy as np
import whisper
from loguru import logger

from config import settings


class WhisperEngineError(Exception):
    """Raised when the Whisper model fails to load or transcribe."""
    pass


class WhisperEngine:
    """
    Wraps the local Whisper model for real-time speech-to-text transcription.

    The model is loaded once on construction and reused for every chunk.
    Each call to transcribe() is synchronous — it blocks until the chunk
    is fully processed (typically 1-3 seconds for the base model on CPU).

    Attributes:
        model_size:  Whisper model variant loaded (e.g. "base").
        is_ready:    True once the model has loaded successfully.
    """

    # Known Whisper hallucinations on silence — immutable frozenset so it
    # cannot be accidentally mutated. Checked in _filter() after lowercasing
    # and stripping punctuation from the transcription result.
    _HALLUCINATION_PHRASES: frozenset = frozenset({
        "thank you",
        "thanks for watching",
        "please subscribe",
        "like and subscribe",
        "you",
        "bye",
        "...",
        ". . .",
        "www.",
        "[silence]",
        "[music]",
        "[blank_audio]",
    })

    def __init__(self) -> None:
        self.model_size: str = settings.whisper.model_size
        self.is_ready: bool = False
        self._model = None

        self._load_model()

    # ── Public API ────────────────────────────────────────────────────────────

    def transcribe(self, audio_chunk: np.ndarray) -> Optional[str]:
        """
        Transcribe a single audio chunk to text.

        Accepts the numpy array produced by AudioRecorder.get_chunk() directly —
        no conversion needed.

        Args:
            audio_chunk: float32 numpy array of shape (N,), values in [-1.0, 1.0].
                         Produced by audio/recorder.py at 16,000 Hz sample rate.

        Returns:
            Cleaned transcription string if real speech was detected.
            None if the chunk was silence, noise, or a known hallucination.

        Raises:
            WhisperEngineError: If the model is not loaded or transcription
                                 fails with an unexpected error.
        """
        if not self.is_ready or self._model is None:
            raise WhisperEngineError(
                "WhisperEngine is not ready. "
                "The model may have failed to load during initialisation."
            )

        self._validate_chunk(audio_chunk)

        try:
            result = self._model.transcribe(
                audio_chunk,
                language=settings.whisper.language,
                task=settings.whisper.task,
                # fp16=False forces float32 on CPU — avoids a UserWarning on
                # machines without a CUDA GPU
                fp16=False,
                # verbose=False suppresses Whisper's own per-segment stdout logs
                verbose=False,
            )
        except Exception as e:
            raise WhisperEngineError(f"Whisper transcription failed: {e}") from e

        raw_text: str = result.get("text", "").strip()

        logger.debug(f"Whisper raw output: '{raw_text[:80]}'")

        return self._filter(raw_text)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """
        Load the Whisper model from disk (or download it on first run).

        Whisper downloads the model to ~/.cache/whisper/ on first use.
        Subsequent runs load it from the cache — no internet needed.

        Model sizes and approximate RAM usage:
            tiny   →  ~75 MB   (fastest, lowest accuracy)
            base   →  ~145 MB  (good balance for real-time CPU use)  ← default
            small  →  ~465 MB  (better accuracy, still usable on CPU)
            medium →  ~1.5 GB  (high accuracy, needs GPU for real-time)
            large  →  ~3 GB    (best accuracy, requires GPU)
        """
        logger.info(
            f"Loading Whisper model '{self.model_size}' "
            f"(this may take a moment on first run while downloading)..."
        )

        try:
            self._model = whisper.load_model(self.model_size)
            self.is_ready = True
            logger.info(f"Whisper '{self.model_size}' model loaded and ready.")

        except Exception as e:
            self.is_ready = False
            raise WhisperEngineError(
                f"Failed to load Whisper model '{self.model_size}': {e}\n"
                "Ensure openai-whisper is installed: pip install openai-whisper"
            ) from e

    def _validate_chunk(self, audio_chunk: np.ndarray) -> None:
        """
        Validate that the incoming audio chunk has the correct shape and dtype
        before passing it to Whisper.

        Args:
            audio_chunk: The array to validate.

        Raises:
            WhisperEngineError: If the array is not a valid 1D float32 array.
        """
        if not isinstance(audio_chunk, np.ndarray):
            raise WhisperEngineError(
                f"Expected numpy array, got {type(audio_chunk).__name__}. "
                "Make sure you are passing the output of AudioRecorder.get_chunk()."
            )

        if audio_chunk.ndim != 1:
            raise WhisperEngineError(
                f"Expected 1D audio array, got shape {audio_chunk.shape}. "
                "Audio must be flattened to mono before transcription."
            )

        if audio_chunk.dtype != np.float32:
            raise WhisperEngineError(
                f"Expected float32 array, got dtype={audio_chunk.dtype}. "
                "AudioRecorder already produces float32 — check the pipeline."
            )

        if len(audio_chunk) == 0:
            raise WhisperEngineError("Received an empty audio chunk.")

    def _filter(self, text: str) -> Optional[str]:
        """
        Decide whether a transcription result represents real speech.

        Filters out:
        1. Empty strings — Whisper returned nothing.
        2. Strings shorter than min_segment_length — too short to be meaningful.
        3. Known hallucination phrases — Whisper false positives on silence.

        Args:
            text: Raw stripped text from Whisper.

        Returns:
            The original text if it passes all filters, otherwise None.
        """
        # Filter 1: empty
        if not text:
            logger.debug("Filtered: empty transcription.")
            return None

        # Filter 2: too short
        if len(text) < settings.whisper.min_segment_length:
            logger.debug(f"Filtered: too short ({len(text)} chars) — '{text}'")
            return None

        # Filter 3: known hallucination phrase
        if text.lower().strip(".! ") in self._HALLUCINATION_PHRASES:
            logger.debug(f"Filtered: known hallucination — '{text}'")
            return None

        logger.info(f"Transcribed ({len(text)} chars): '{text[:80]}{'...' if len(text) > 80 else ''}'")
        return text
