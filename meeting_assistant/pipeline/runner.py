"""
pipeline/runner.py — Main pipeline orchestrator.

Responsibility:
    Own and coordinate every module in the system. Run the continuous
    tick loop that captures audio, transcribes it, updates memory,
    runs agents, and saves results to disk.

This is the only file that imports from every other module.
All other modules are unaware of each other — they communicate
only through the runner.

Loop (one tick = one audio chunk ≈ 7 seconds):
    1. get_chunk()       — block until audio is ready
    2. transcribe()      — convert audio to text (or None on silence)
    3. add_segment()     — update context memory
    4. [every N ticks]   — run all three agents on current snapshot
    5. save()            — write updated meeting notes to disk
    6. repeat

Usage:
    runner = PipelineRunner()
    runner.run()           # blocks until Ctrl+C or fatal error
    runner.shutdown()      # called automatically on exit
"""

import time
from typing import Optional

from loguru import logger

from audio.recorder import AudioRecorder, AudioRecorderError
from transcription.whisper_engine import WhisperEngine, WhisperEngineError
from memory.context_store import ContextStore, ContextSnapshot
from agents.summary_agent import SummaryAgent
from agents.topic_agent import TopicAgent
from agents.action_agent import ActionAgent
from storage.writer import MeetingWriter, MeetingWriterError
from config import settings


class PipelineRunner:
    """
    Orchestrates the full meeting assistant pipeline.

    Creates and owns all module instances. Runs the main tick loop
    that drives audio capture → transcription → memory → agents → storage.

    Attributes:
        is_running: True while the main loop is active.
    """

    def __init__(self) -> None:
        logger.info("Initialising pipeline...")

        # ── Instantiate all modules ────────────────────────────────────────
        # Order matters — WhisperEngine loads the model here (~2-5s on first run)
        self._recorder      = AudioRecorder()
        self._engine        = WhisperEngine()
        self._store         = ContextStore()
        self._summary_agent = SummaryAgent()
        self._topic_agent   = TopicAgent()
        self._action_agent  = ActionAgent()
        self._writer        = MeetingWriter()

        # ── Internal state ─────────────────────────────────────────────────
        self.is_running: bool = False

        # Increments each tick — used to schedule agent runs
        self._tick_count: int = 0

        # Tracks consecutive transcription failures
        self._consecutive_errors: int = 0

        # Cached last valid agent outputs — used when safe_run() returns None
        self._last_summary:      str       = ""
        self._last_topics:       list[str] = []
        self._last_action_items: list[dict]= []

        logger.info("Pipeline ready — all modules initialised.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Start the main pipeline loop.

        Opens the microphone and runs continuously until:
            - The user presses Ctrl+C
            - An unrecoverable error occurs

        Calls shutdown() automatically on exit to save final notes.
        """
        logger.info("=" * 60)
        logger.info("  Meeting Assistant started — listening...")
        logger.info("  Press Ctrl+C to stop and save final notes.")
        logger.info("=" * 60)

        try:
            self._recorder.start()
            self.is_running = True
            self._loop()

        except AudioRecorderError as e:
            logger.error(f"Microphone error: {e}")

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")

        except Exception as e:
            logger.error(f"Unrecoverable pipeline error: {e}")

        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """
        Stop the pipeline cleanly and save final meeting notes.

        Safe to call multiple times — guards against double-shutdown.
        Called automatically by run() on any exit path.
        """
        if not self.is_running:
            return

        self.is_running = False
        logger.info("Shutting down pipeline...")

        # Stop microphone first
        self._recorder.stop()

        # Run one final agent pass on everything collected
        logger.info("Running final agent pass...")
        snapshot = self._store.get_snapshot()

        if snapshot.chunk_count > 0:
            self._run_agents(snapshot)
            self._save()
            logger.info(
                f"Final notes saved | "
                f"chunks={snapshot.chunk_count} | "
                f"words={snapshot.word_count}"
            )
        else:
            logger.info("No speech detected — nothing to save.")

        logger.info("Pipeline shut down cleanly.")

    # ── Private — main loop ────────────────────────────────────────────────────

    def _loop(self) -> None:
        """
        The core tick loop. Runs until is_running becomes False.

        Each iteration:
            1. Blocks on get_chunk() until audio is ready
            2. Transcribes the chunk
            3. Updates memory if transcription succeeded
            4. Runs agents every agent_run_interval ticks
            5. Saves updated notes to disk
        """
        while self.is_running:
            self._tick_count += 1
            logger.debug(f"── Tick {self._tick_count} ──")

            # ── Step 1: Get audio chunk ────────────────────────────────────
            try:
                chunk = self._recorder.get_chunk()
            except AudioRecorderError as e:
                logger.warning(f"Audio chunk error: {e}")
                self._handle_error()
                continue

            # ── Step 2: Transcribe ─────────────────────────────────────────
            try:
                text = self._engine.transcribe(chunk)
            except WhisperEngineError as e:
                logger.warning(f"Transcription error: {e}")
                self._handle_error()
                continue

            # None = silence or hallucination — skip this tick
            if text is None:
                logger.debug("Tick skipped — silence or hallucination.")
                self._consecutive_errors = 0
                continue

            # ── Step 3: Update memory ──────────────────────────────────────
            self._consecutive_errors = 0
            self._store.add_segment(text)

            # ── Step 4: Run agents (every N ticks) ────────────────────────
            if self._tick_count % settings.pipeline.agent_run_interval == 0:
                snapshot = self._store.get_snapshot()
                self._run_agents(snapshot)

                # ── Step 5: Save to disk ───────────────────────────────────
                self._save()

    # ── Private — agent coordination ───────────────────────────────────────────

    def _run_agents(self, snapshot: ContextSnapshot) -> None:
        """
        Run all three agents on the current snapshot and cache results.

        Each agent uses safe_run() — a failing agent does not affect
        the others. If an agent returns None, the previous cached result
        is kept so the display never goes blank.

        Args:
            snapshot: Current ContextSnapshot from the context store.
        """
        logger.debug(
            f"Running agents | "
            f"chunks={snapshot.chunk_count} | "
            f"context_chars={len(snapshot.recent_context)}"
        )

        # Summary agent — returns str or None
        summary = self._summary_agent.safe_run(snapshot)
        if summary is not None:
            self._last_summary = summary

        # Topic agent — returns list[str] or None
        topics = self._topic_agent.safe_run(snapshot)
        if topics is not None:
            self._last_topics = topics

        # Action agent — returns list[dict] or None
        action_items = self._action_agent.safe_run(snapshot)
        if action_items is not None:
            self._last_action_items = action_items

        logger.debug(
            f"Agents done | "
            f"summary_chars={len(self._last_summary)} | "
            f"topics={len(self._last_topics)} | "
            f"actions={len(self._last_action_items)}"
        )

    # ── Private — persistence ──────────────────────────────────────────────────

    def _save(self) -> None:
        """
        Build a MeetingState from cached results and write to disk.

        Uses the most recent successful output from each agent.
        If no agents have run yet, all fields will be empty — the
        writer still saves so the frontend has a file to read from.
        """
        snapshot = self._store.get_snapshot()

        try:
            state = MeetingWriter.build_state(
                summary=self._last_summary,
                topics=self._last_topics,
                action_items=self._last_action_items,
                full_transcript=snapshot.full_transcript,
                word_count=snapshot.word_count,
            )
            self._writer.save(state)

        except MeetingWriterError as e:
            logger.error(f"Failed to save meeting notes: {e}")

    # ── Private — error handling ───────────────────────────────────────────────

    def _handle_error(self) -> None:
        """
        Track consecutive errors and pause the loop if too many occur.

        After max_consecutive_errors failures in a row, waits 5 seconds
        before continuing — gives the microphone or Whisper time to
        recover without spinning the CPU.
        """
        self._consecutive_errors += 1

        if self._consecutive_errors >= settings.pipeline.max_consecutive_errors:
            logger.warning(
                f"{self._consecutive_errors} consecutive errors — "
                f"pausing for 5 seconds. "
                f"Check microphone connection."
            )
            time.sleep(5)
            self._consecutive_errors = 0
