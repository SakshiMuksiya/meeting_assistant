"""
agents/topic_agent.py — Discussion topic extraction agent.

Responsibility:
    Identify and extract the key topics currently being discussed
    in the meeting from the recent context window.

Receives:
    ContextSnapshot.recent_context — the latest conversation window

Produces:
    list[str] — short topic phrases representing current discussion themes
    e.g. ["Q3 roadmap planning", "backend migration ownership", "October deadlines"]

Design:
    - Stateless — does not track topics across ticks, only identifies
      what is being discussed RIGHT NOW in the context window
    - Returns short phrases, not full sentences
    - Parses Gemini's plain text response by splitting on newlines
    - Falls back to previous topics if API returns empty response
"""

from loguru import logger

from agents.base_agent import BaseAgent
from memory.context_store import ContextSnapshot


class TopicAgent(BaseAgent):
    """
    Extracts key discussion topics from the current context window.

    Each run() call independently analyses the recent context and
    returns a fresh list of topics. No state is carried between ticks
    — the agent only answers "what are they discussing right now?"

    Returns:
        list[str] — topic phrases, or previous topics if API fails.
    """

    SYSTEM_PROMPT = """You are an expert at identifying discussion topics \
in business meetings and conversations.

Your task is to extract the key topics currently being discussed.

Instructions:
- Identify the main themes, subjects, and topics in the conversation
- Output each topic as a SHORT PHRASE (2 to 5 words) — not full sentences
- List only topics that are genuinely discussed, not implied or assumed
- Output one topic per line — nothing else
- Do NOT number the topics, add bullet points, or any other formatting
- Do NOT include filler topics like "general discussion" or "meeting intro"
- Aim for 2 to 6 topics maximum — quality over quantity
- If the context is too short or unclear, output only the most obvious topic

Example output:
Q3 roadmap planning
Backend migration ownership
October release deadline
Budget approval process"""

    def __init__(self) -> None:
        super().__init__()
        # Stores last successful topics — returned if current run fails
        self._previous_topics: list[str] = []

    # ── Core method ───────────────────────────────────────────────────────────

    def run(self, snapshot: ContextSnapshot) -> list[str]:
        """
        Extract key topics from the current context window.

        Args:
            snapshot: Current ContextSnapshot from the context store.
                      Uses snapshot.recent_context as input.

        Returns:
            List of short topic phrase strings.
            Returns previous topics if the API returns empty output.
        """
        user_prompt = (
            f"Extract the key discussion topics from this meeting excerpt:\n\n"
            f"{snapshot.recent_context}"
        )

        messages = self._build_messages(
            system=self.SYSTEM_PROMPT,
            user=user_prompt,
        )

        response = self._client.chat(
            messages,
            model=None,        # uses settings.llm.extraction_model (gemini-2.0-flash)
            temperature=None,  # uses settings.llm.extraction_temperature (0.0)
        )

        topics = self._parse_topics(response)

        if not topics:
            logger.warning(
                "TopicAgent received empty response — "
                "keeping previous topics."
            )
            return self._previous_topics

        self._previous_topics = topics
        logger.debug(f"Topics extracted: {topics}")

        return topics

    # ── Private helpers ───────────────────────────────────────────────────────

    def _parse_topics(self, response: str) -> list[str]:
        """
        Parse Gemini's plain text response into a clean list of topics.

        Splits on newlines, strips whitespace, removes empty lines,
        and removes any accidental bullet points or numbering that
        Gemini may add despite instructions.

        Args:
            response: Raw string response from the API.

        Returns:
            List of clean topic strings. Empty list if nothing valid.
        """
        if not response.strip():
            return []

        topics = []
        for line in response.strip().splitlines():
            # Strip whitespace
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Remove common accidental prefixes: "- ", "• ", "1. ", "* "
            for prefix in ["- ", "• ", "* ", "· "]:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break

            # Remove leading numbering like "1.", "2.", "10."
            if len(line) > 2 and line[0].isdigit() and line[1] in ".):":
                line = line[2:].strip()
            elif len(line) > 3 and line[:2].isdigit() and line[2] in ".):":
                line = line[3:].strip()

            # Skip if still empty after cleaning
            if not line:
                continue

            topics.append(line)

        return topics

    def reset(self) -> None:
        """
        Clear stored topics. Call between meetings.
        """
        self._previous_topics = []
        logger.info("TopicAgent reset.")
