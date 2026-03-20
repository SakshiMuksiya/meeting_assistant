"""
agents/summary_agent.py — Incremental meeting summarization agent.

Responsibility:
    Maintain and continuously update a concise summary of the meeting
    using incremental prompting — each run refines the previous summary
    rather than regenerating from scratch.

Receives:
    ContextSnapshot.recent_context — the latest conversation window

Produces:
    str — updated meeting summary

Incremental design:
    On each run(), the agent receives:
        1. Its own previous summary (stored in self._previous_summary)
        2. The new context window from the snapshot
    It then asks Gemini to update the summary incorporating the new
    content — not rewrite it entirely. This means:
        - Only new content is processed each tick
        - The summary gets progressively more detailed and accurate
        - API token usage stays proportional to NEW speech, not total
"""

from loguru import logger

from agents.base_agent import BaseAgent
from memory.context_store import ContextSnapshot


class SummaryAgent(BaseAgent):
    """
    Produces an incrementally updated plain-text meeting summary.

    The summary starts empty and is refined on every run() call.
    The agent always receives its own last output as context so it
    can build on previous understanding rather than starting fresh.

    Returns:
        str — updated summary, or the previous summary unchanged if
              the new context adds nothing significant.
    """

    SYSTEM_PROMPT = """You are an expert meeting summarizer with years of \
experience distilling complex discussions into clear, concise summaries.

Your task is to maintain a running summary of a meeting in progress.
You will be given:
  1. The current summary (may be empty at the start of the meeting)
  2. A new excerpt of the meeting conversation

Instructions:
- Update the summary to incorporate important new information from the excerpt
- Keep the summary concise — aim for 3 to 6 sentences
- Preserve all important decisions, key points, and context from the previous summary
- Do NOT repeat the same information twice
- Write in past tense (e.g. "The team discussed...", "Alice agreed to...")
- If the excerpt contains no new meaningful information (silence, filler words),
  return the previous summary unchanged
- Output ONLY the updated summary — no preamble, no explanation"""

    def __init__(self) -> None:
        super().__init__()
        # Stores the last successful summary — passed back to Gemini
        # on the next run so it can build on previous understanding
        self._previous_summary: str = ""

    # ── Core method ───────────────────────────────────────────────────────────

    def run(self, snapshot: ContextSnapshot) -> str:
        """
        Generate an updated meeting summary using incremental prompting.

        Passes the previous summary alongside new context so Gemini
        refines rather than regenerates — keeps token usage low and
        summary quality high throughout long meetings.

        Args:
            snapshot: Current ContextSnapshot from the context store.
                      Uses snapshot.recent_context as the new content.

        Returns:
            Updated summary as a plain string. Falls back to the
            previous summary if the API returns an empty response.
        """
        user_prompt = self._build_user_prompt(snapshot.recent_context)

        messages = self._build_messages(
            system=self.SYSTEM_PROMPT,
            user=user_prompt,
        )

        updated_summary = self._client.chat(
            messages,
            model=None,        # uses settings.llm.summary_model (gemini-2.0-flash)
            temperature=None,  # uses settings.llm.summary_temperature (0.4)
        )

        if not updated_summary.strip():
            logger.warning(
                "SummaryAgent received empty response — "
                "keeping previous summary."
            )
            return self._previous_summary

        self._previous_summary = updated_summary
        logger.debug(
            f"Summary updated | "
            f"chars={len(updated_summary)} | "
            f"preview='{updated_summary[:80]}...'"
        )

        return updated_summary

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_user_prompt(self, recent_context: str) -> str:
        """
        Build the user message combining previous summary and new context.

        Args:
            recent_context: Latest context window from the snapshot.

        Returns:
            Formatted user prompt string ready for the API.
        """
        if self._previous_summary:
            return (
                f"CURRENT SUMMARY:\n{self._previous_summary}\n\n"
                f"NEW MEETING EXCERPT:\n{recent_context}\n\n"
                "Please update the summary to incorporate any important "
                "new information from the excerpt above."
            )
        else:
            # First run — no previous summary yet
            return (
                f"MEETING EXCERPT (start of meeting):\n{recent_context}\n\n"
                "Please write an initial summary of what has been discussed so far."
            )

    def reset(self) -> None:
        """
        Clear the stored summary. Call between meetings.
        """
        self._previous_summary = ""
        logger.info("SummaryAgent reset.")
