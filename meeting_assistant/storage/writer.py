"""
storage/writer.py — Meeting notes persistence module.

Responsibility:
    Receive the current meeting state from pipeline/runner.py, serialize
    it to a structured JSON file atomically, and provide a load function
    so the Streamlit frontend can read the latest state.

Layer position:
    pipeline/runner.py  →  [THIS FILE]  →  output/meeting_notes.json
                                                  ↓
                                          frontend/app.py (reads)

Receives from pipeline/runner.py:
    MeetingState dataclass containing:
        summary         : str
        topics          : list[str]
        action_items    : list[dict]
        full_transcript : str
        word_count      : int
        last_updated    : str  (ISO 8601 timestamp)

Produces:
    output/meeting_notes.json — structured JSON file read by frontend

Atomic write design:
    Writes to a .tmp file first, then renames to the final path.
    A rename is atomic on all major operating systems — the frontend
    always sees either the previous complete file or the new complete
    file, never a half-written corrupt file.

    meeting_notes.tmp  →  (rename)  →  meeting_notes.json

Usage:
    writer = MeetingWriter()

    state = MeetingState(
        summary=\"The team discussed Q3 roadmap...\",
        topics=[\"Q3 planning\", \"Backend migration\"],
        action_items=[{\"person\": \"Alice\", \"task\": \"...\", \"deadline\": \"...\"}],
        full_transcript=\"Alice: Let's start with...\",
        word_count=312,
        last_updated=\"2024-10-15T14:32:01\",
    )

    writer.save(state)       # writes meeting_notes.json atomically
    loaded = writer.load()   # reads and returns current state as dict
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

from config import MEETING_NOTES_PATH, OUTPUT_DIR


# ── MeetingState ───────────────────────────────────────────────────────────────

@dataclass
class MeetingState:
    """
    Complete snapshot of meeting outputs at a point in time.

    This is what the pipeline builds each tick and passes to
    MeetingWriter.save(). All fields are JSON-serializable.

    Attributes:
        summary:         Latest meeting summary from SummaryAgent.
        topics:          Current discussion topics from TopicAgent.
        action_items:    Accumulated action items from ActionAgent.
                         Each dict has keys: person, task, deadline.
        full_transcript: Complete transcript from ContextSnapshot.
        word_count:      Total words spoken so far.
        last_updated:    ISO 8601 timestamp of when this state was saved.
    """
    summary:          str
    topics:           list[str]
    action_items:     list[dict]
    full_transcript:  str
    word_count:       int
    last_updated:     str


# ── MeetingWriter ──────────────────────────────────────────────────────────────

class MeetingWriter:
    """
    Writes and reads meeting state to/from a JSON file on disk.

    The output file is read by:
        - frontend/app.py (Streamlit dashboard — polls every 3 seconds)
        - Any post-meeting analysis or export tools

    Atomic writes guarantee the frontend never reads a corrupt file.
    """

    def __init__(self) -> None:
        # Ensure the output directory exists (config creates it, but be safe)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Temp file path used during atomic write
        self._tmp_path: Path = MEETING_NOTES_PATH.with_suffix(".tmp")

        logger.info(
            f"MeetingWriter initialised | "
            f"output={MEETING_NOTES_PATH}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def save(self, state: MeetingState) -> None:
        """
        Serialize and atomically write the meeting state to disk.

        Converts MeetingState to a JSON-serializable dict, writes to a
        temporary file, then renames it to the final path. This ensures
        the output file is never in a partially-written state.

        Args:
            state: Current MeetingState from the pipeline runner.

        Raises:
            MeetingWriterError: If the file cannot be written due to
                                 permissions or disk space issues.
        """
        try:
            data = asdict(state)

            # Write to temp file first
            with open(self._tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Atomic rename to final path
            os.replace(self._tmp_path, MEETING_NOTES_PATH)

            logger.debug(
                f"Meeting notes saved | "
                f"words={state.word_count} | "
                f"topics={len(state.topics)} | "
                f"actions={len(state.action_items)}"
            )

        except OSError as e:
            raise MeetingWriterError(
                f"Failed to write meeting notes to {MEETING_NOTES_PATH}: {e}"
            ) from e

    def load(self) -> Optional[dict]:
        """
        Read and return the current meeting notes from disk.

        Used by the Streamlit frontend to get the latest state,
        and by the pipeline to restore state after a restart.

        Returns:
            Parsed dict matching the MeetingState structure, or None
            if the file does not exist yet (meeting just started).

        Raises:
            MeetingWriterError: If the file exists but cannot be parsed.
        """
        if not MEETING_NOTES_PATH.exists():
            logger.debug("No meeting notes file found — meeting not started yet.")
            return None

        try:
            with open(MEETING_NOTES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.debug("Meeting notes loaded from disk.")
            return data

        except json.JSONDecodeError as e:
            raise MeetingWriterError(
                f"Meeting notes file is corrupted: {e}\n"
                f"Path: {MEETING_NOTES_PATH}"
            ) from e

        except OSError as e:
            raise MeetingWriterError(
                f"Could not read meeting notes file: {e}"
            ) from e

    def clear(self) -> None:
        """
        Delete the meeting notes file from disk.

        Call between meetings to start fresh. Safe to call even if
        the file does not exist.
        """
        if MEETING_NOTES_PATH.exists():
            MEETING_NOTES_PATH.unlink()
            logger.info("Meeting notes file deleted.")

        if self._tmp_path.exists():
            self._tmp_path.unlink()

    # ── Convenience class method ──────────────────────────────────────────────

    @staticmethod
    def build_state(
        summary:         str,
        topics:          list[str],
        action_items:    list[dict],
        full_transcript: str,
        word_count:      int,
    ) -> MeetingState:
        """
        Convenience factory — build a MeetingState with auto timestamp.

        Called by the pipeline runner so it does not need to import
        datetime or manually construct the last_updated string.

        Args:
            summary:         Latest summary from SummaryAgent.
            topics:          Current topics from TopicAgent.
            action_items:    All action items from ActionAgent.
            full_transcript: Full transcript from ContextSnapshot.
            word_count:      Total word count from ContextSnapshot.

        Returns:
            MeetingState with last_updated set to current UTC time.
        """
        return MeetingState(
            summary=summary,
            topics=topics,
            action_items=action_items,
            full_transcript=full_transcript,
            word_count=word_count,
            last_updated=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        )


# ── MeetingWriterError ─────────────────────────────────────────────────────────

class MeetingWriterError(Exception):
    """Raised when the writer cannot save or load the meeting notes file."""
    pass
