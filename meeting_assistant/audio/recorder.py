"""
audio/recorder.py — Live microphone capture module.

Responsibility:
    Continuously record audio from the default microphone in fixed-duration
    chunks and place each chunk into a thread-safe queue for the transcription
    module to consume.

How it works:
    1. AudioRecorder.start() opens the microphone stream in a background thread.
    2. sounddevice calls _audio_callback() every `blocksize` frames (non-blocking).
    3. _audio_callback() accumulates incoming frames into a buffer.
    4. Once the buffer reaches `chunk_duration_seconds` worth of audio, it is
       copied into `self.queue` as a numpy array and the buffer is reset.
    5. The pipeline calls AudioRecorder.get_chunk() to pop chunks one at a time.
    6. AudioRecorder.stop() closes the stream cleanly.

Thread safety:
    sounddevice calls _audio_callback() on a private audio thread.
    queue.Queue is thread-safe — no locks needed.

Usage:
    recorder = AudioRecorder()
    recorder.start()

    chunk = recorder.get_chunk()   # blocks until a chunk is ready
    # chunk is a numpy float32 array of shape (N,) — ready for Whisper

    recorder.stop()
"""

import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd
from loguru import logger

from config import settings


class AudioRecorderError(Exception):
    """Raised when the microphone cannot be opened or fails during recording."""
    pass


class AudioRecorder:
    """
    Captures microphone audio in fixed-duration chunks.

    Each chunk is a numpy float32 array of shape (samples,) at
    settings.audio.sample_rate Hz — exactly what Whisper expects.

    Attributes:
        queue:      Thread-safe queue holding captured chunks.
                    Each item is a numpy array ready for transcription.
        is_running: True while the microphone stream is active.
    """

    def __init__(self) -> None:
        self._sample_rate: int = settings.audio.sample_rate
        self._channels: int = settings.audio.channels
        self._blocksize: int = settings.audio.blocksize
        self._chunk_samples: int = int(
            settings.audio.sample_rate * settings.audio.chunk_duration_seconds
        )

        # Thread-safe queue — transcription module reads from here
        # maxsize=10 prevents unbounded memory growth if transcription falls behind
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=10)

        self.is_running: bool = False

        # Internal accumulation buffer — fills up until one full chunk is ready
        self._buffer: list[np.ndarray] = []
        self._buffer_samples: int = 0

        # sounddevice stream handle
        self._stream: Optional[sd.InputStream] = None

        # Protects is_running flag across threads
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Open the microphone and begin capturing audio.

        Raises:
            AudioRecorderError: If no microphone is found or the device
                                 cannot be opened at the required sample rate.
        """
        with self._lock:
            if self.is_running:
                logger.warning("AudioRecorder.start() called but already running — ignoring.")
                return

        logger.info(
            f"Opening microphone | "
            f"sample_rate={self._sample_rate} Hz | "
            f"chunk_duration={settings.audio.chunk_duration_seconds}s | "
            f"chunk_samples={self._chunk_samples}"
        )

        try:
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="float32",
                blocksize=self._blocksize,
                callback=self._audio_callback,
            )
            self._stream.start()

            with self._lock:
                self.is_running = True

            logger.info("Microphone stream started — recording.")

        except sd.PortAudioError as e:
            raise AudioRecorderError(
                f"Could not open microphone: {e}\n"
                "Check that a microphone is connected and not in use by another app."
            ) from e

    def stop(self) -> None:
        """
        Stop recording and close the microphone stream cleanly.
        Safe to call multiple times.
        """
        with self._lock:
            if not self.is_running:
                return
            self.is_running = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        logger.info("Microphone stream stopped.")

    def get_chunk(self, timeout: float = 30.0) -> np.ndarray:
        """
        Block until a full audio chunk is available and return it.

        Args:
            timeout: Maximum seconds to wait for a chunk.
                     Raise AudioRecorderError if exceeded — likely means
                     the microphone has stalled.

        Returns:
            numpy float32 array of shape (chunk_samples,), normalised to [-1, 1].
            Ready to pass directly to whisper.transcribe().

        Raises:
            AudioRecorderError: If no chunk arrives within `timeout` seconds.
        """
        try:
            chunk = self.queue.get(timeout=timeout)
            logger.debug(f"Chunk dequeued | samples={len(chunk)} | queue_size={self.queue.qsize()}")
            return chunk
        except queue.Empty:
            raise AudioRecorderError(
                f"No audio chunk received within {timeout}s. "
                "Is the microphone still connected and active?"
            )

    def available(self) -> bool:
        """Return True if at least one chunk is ready to be consumed."""
        return not self.queue.empty()

    # ── Internal callback (runs on sounddevice audio thread) ──────────────────

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time,       # sounddevice CData time struct — unused
        status: sd.CallbackFlags,
    ) -> None:
        """
        Called by sounddevice on every `blocksize` frames of audio.

        This runs on a private audio thread — keep it fast and non-blocking.
        No file I/O, no logging with I/O, no heavy computation here.

        Args:
            indata:  numpy array of shape (blocksize, channels), float32.
            frames:  Number of frames in indata (== blocksize).
            time:    Timestamp struct (unused).
            status:  Flags indicating buffer over/underflow conditions.
        """
        if status:
            # Log on the main thread to avoid blocking the audio thread.
            # status.input_overflow means we're not consuming fast enough.
            logger.warning(f"Audio callback status: {status}")

        # Flatten to mono (shape: (frames,)) — Whisper requires 1D input
        mono = indata[:, 0].copy()

        self._buffer.append(mono)
        self._buffer_samples += len(mono)

        # Once we have enough samples for one full chunk, package and enqueue
        if self._buffer_samples >= self._chunk_samples:
            full = np.concatenate(self._buffer)

            # Slice out exactly one chunk
            chunk = full[: self._chunk_samples]

            # Keep any overflow frames for the next chunk
            overflow = full[self._chunk_samples :]
            if len(overflow) > 0:
                self._buffer = [overflow]
                self._buffer_samples = len(overflow)
            else:
                self._buffer = []
                self._buffer_samples = 0

            # Drop oldest chunk if the pipeline hasn't caught up (queue full)
            # This prevents memory bloat during long silences or slow processing
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                    logger.warning(
                        "Audio queue full — dropped oldest chunk. "
                        "Transcription may be falling behind."
                    )
                except queue.Empty:
                    pass

            self.queue.put_nowait(chunk)

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def list_devices() -> None:
        """
        Print all available audio input devices to the terminal.
        Useful for debugging microphone detection issues.

        Usage:
            from audio.recorder import AudioRecorder
            AudioRecorder.list_devices()
        """
        print("\nAvailable audio input devices:")
        print(sd.query_devices())
        default = sd.query_devices(kind="input")
        print(f"\nDefault input device: {default['name']}")

    def __enter__(self) -> "AudioRecorder":
        """Support use as a context manager: `with AudioRecorder() as r:`"""
        self.start()
        return self

    def __exit__(self, *args) -> None:
        """Ensure stream is stopped even if an exception occurs."""
        self.stop()
