"""
config.py — Central configuration for the Meeting Assistant.

All tuneable parameters live here. Never hardcode values in modules.
Import this anywhere with: from config import settings
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

MEETING_NOTES_PATH = OUTPUT_DIR / "meeting_notes.json"


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class AudioConfig:
    """Microphone capture settings."""

    # Sample rate expected by Whisper (do not change)
    sample_rate: int = 16_000

    # Number of audio channels (1 = mono, required by Whisper)
    channels: int = 1

    # Duration of each captured chunk in seconds
    chunk_duration_seconds: float = 7.0

    # Internal buffer size passed to sounddevice
    blocksize: int = 1024


@dataclass
class WhisperConfig:
    """Whisper transcription settings."""

    # Model size: "tiny", "base", "small", "medium", "large"
    # "base" is the sweet spot for real-time use on CPU.
    # Use "small" or "medium" if you have a GPU.
    model_size: Literal["tiny", "base", "small", "medium", "large"] = "base"

    # Language hint — set to None for auto-detection
    language: str | None = "en"

    # Whisper task: "transcribe" or "translate" (translate → English)
    task: Literal["transcribe", "translate"] = "transcribe"

    # Minimum number of characters for a segment to be accepted.
    # Filters out Whisper hallucinations on silence (e.g. "Thank you." "...")
    min_segment_length: int = 10


@dataclass
class MemoryConfig:
    """Context memory settings."""

    # Maximum characters kept in the sliding context window
    # sent to agents. ~3000 chars ≈ ~750 tokens, fits comfortably
    # in a Gemini 2.0 Flash prompt alongside the system message.
    max_context_chars: int = 3000

    # Whether to enable vector-based semantic retrieval.
    # Requires `sentence-transformers` to be installed.
    # Set to False for the initial version.
    enable_vector_store: bool = False

    # Embedding model used when vector store is enabled.
    # This runs locally (no API calls).
    embedding_model: str = "all-MiniLM-L6-v2"

    # Number of semantically relevant passages to retrieve
    # (used only when enable_vector_store=True)
    top_k_retrieval: int = 3


@dataclass
class LLMConfig:
    """LLM backend settings — using Google Gemini via AI Studio (free tier)."""

    # Gemini 2.0 Flash is used for all agents.
    # It handles both high-quality summarisation and structured extraction well.
    # Free tier: 250 req/day, 1M tokens/min — more than enough for meetings.
    summary_model: str = "gemini-2.0-flash"
    extraction_model: str = "gemini-2.0-flash"

    # Gemini's OpenAI-compatible base URL (no SDK change needed)
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

    # Maximum tokens in the LLM response
    max_tokens: int = 1024

    # Temperature: 0.0 = deterministic, good for structured extraction.
    # Summary agent gets a slightly higher value for more natural language.
    summary_temperature: float = 0.4
    extraction_temperature: float = 0.0

    # Request timeout in seconds
    timeout: int = 30


@dataclass
class PipelineConfig:
    """Orchestration loop settings."""

    # Run agents every N audio chunks (not every chunk).
    # At chunk_duration=7s, agent_run_interval=2 → agents run every 14 s.
    # Increase this to reduce API costs.
    agent_run_interval: int = 2

    # Maximum number of consecutive transcription errors before the
    # pipeline pauses and asks the user to check the microphone.
    max_consecutive_errors: int = 5

    # How many seconds the Streamlit frontend waits before re-reading
    # the JSON file. This value is read by frontend/app.py.
    frontend_poll_interval_seconds: int = 3


@dataclass
class Settings:
    """Root settings object. Import this in every module."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    # Gemini API key — reads from environment variable.
    # Get a free key at https://aistudio.google.com → "Get API key"
    # Set it with: export GEMINI_API_KEY="AIza..."
    # Never hardcode this value here.
    gemini_api_key: str = field(
        default_factory=lambda: os.environ.get("GEMINI_API_KEY", "")
    )

    def validate(self) -> None:
        """
        Call this at startup to catch misconfiguration early.
        Raises ValueError with a helpful message on the first problem found.
        """
        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set.\n"
                "Get a free key at https://aistudio.google.com\n"
                "Then run: export GEMINI_API_KEY='AIza...'"
            )

        if self.audio.chunk_duration_seconds < 3:
            raise ValueError(
                "chunk_duration_seconds must be >= 3. "
                "Shorter chunks give Whisper too little audio context."
            )

        if self.memory.max_context_chars < 500:
            raise ValueError(
                "max_context_chars must be >= 500. "
                "Too small and agents lose all conversational context."
            )

        if self.pipeline.agent_run_interval < 1:
            raise ValueError("agent_run_interval must be >= 1.")


# ── Singleton ──────────────────────────────────────────────────────────────────
# Import `settings` directly anywhere in the codebase.
# Example: from config import settings

settings = Settings()
