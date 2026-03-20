"""
agents/base_agent.py — Abstract base class for all meeting agents.

Responsibility:
    Define the shared interface and common utilities that every agent
    inherits. Ensures all agents behave consistently and reduces
    boilerplate in the individual agent files.

All agents inherit from BaseAgent and must implement:
    - SYSTEM_PROMPT : str        — the agent's role and instructions
    - run(snapshot) → any        — the agent's main logic

Usage (by subclasses only — never instantiate BaseAgent directly):
    class SummaryAgent(BaseAgent):
        SYSTEM_PROMPT = "You are an expert meeting summarizer..."

        def run(self, snapshot: ContextSnapshot) -> str:
            messages = self._build_messages(
                system=self.SYSTEM_PROMPT,
                user=f"Summarize this: {snapshot.recent_context}"
            )
            return self._client.chat(messages)
"""

from abc import ABC, abstractmethod
from typing import Any

from loguru import logger

from llm.llm_client import get_client, LLMClient, LLMError
from memory.context_store import ContextSnapshot


class BaseAgent(ABC):
    """
    Abstract base class for all meeting assistant agents.

    Provides:
        - A shared LLMClient instance via get_client()
        - _build_messages() to format prompts into the API message list
        - _safe_run() to wrap run() with error handling

    Subclasses must define:
        - SYSTEM_PROMPT : class-level str
        - run(snapshot: ContextSnapshot) -> Any
    """

    # Subclasses override this with their specific role and instructions
    SYSTEM_PROMPT: str = ""

    def __init__(self) -> None:
        # All agents share the same singleton LLMClient — one connection,
        # no redundant initialisation
        self._client: LLMClient = get_client()
        logger.info(f"{self.__class__.__name__} initialised.")

    # ── Interface (subclasses must implement) ─────────────────────────────────

    @abstractmethod
    def run(self, snapshot: ContextSnapshot) -> Any:
        """
        Execute the agent's main task on the current meeting snapshot.

        Called by the pipeline on every agent-run tick. Must return a
        result that the pipeline can store and display.

        Args:
            snapshot: Current ContextSnapshot from ContextStore.get_snapshot().
                      Use snapshot.recent_context for the agent's input text.

        Returns:
            Agent-specific result:
                SummaryAgent    → str
                TopicAgent      → list[str]
                ActionAgent     → list[dict]

        Raises:
            LLMError: If the Gemini API call fails after retries.
        """
        ...

    # ── Shared helpers (available to all subclasses) ──────────────────────────

    def _build_messages(
        self,
        system: str,
        user: str,
    ) -> list[dict[str, str]]:
        """
        Format a system prompt and user prompt into the message list
        that LLMClient.chat() and LLMClient.chat_json() expect.

        Args:
            system: The agent's system prompt — defines its role and
                    output format. Use self.SYSTEM_PROMPT here.
            user:   The user message — the actual content to process,
                    typically built from snapshot.recent_context.

        Returns:
            List of two message dicts:
                [
                    {"role": "system", "content": "<system prompt>"},
                    {"role": "user",   "content": "<user prompt>"},
                ]
        """
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

    def safe_run(self, snapshot: ContextSnapshot) -> Any | None:
        """
        Run the agent with full error handling.

        Wraps run() so that a single failing agent tick does not crash
        the pipeline. Logs the error and returns None — the pipeline
        skips None results and keeps the previous valid output.

        Use this in the pipeline instead of calling run() directly.

        Args:
            snapshot: Current ContextSnapshot from the context store.

        Returns:
            The result of run() on success, or None on any error.
        """
        if not snapshot.recent_context.strip():
            logger.debug(
                f"{self.__class__.__name__}.safe_run() skipped — "
                "context is empty (meeting just started)."
            )
            return None

        try:
            result = self.run(snapshot)
            return result

        except LLMError as e:
            logger.error(
                f"{self.__class__.__name__} failed (LLM error): {e}"
            )
            return None

        except Exception as e:
            logger.error(
                f"{self.__class__.__name__} failed (unexpected error): {e}"
            )
            return None
