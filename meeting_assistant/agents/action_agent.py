"""
agents/action_agent.py — Action item extraction agent.

Responsibility:
    Extract structured action items (tasks, owners, deadlines) from
    the meeting conversation using Gemini's JSON mode to guarantee
    reliable, parseable output every time.

Receives:
    ContextSnapshot.recent_context — the latest conversation window

Produces:
    list[dict] — structured action items, each with:
        {
            "person":   str — who is responsible (or "Unassigned")
            "task":     str — what needs to be done
            "deadline": str — when it is due (or "Not specified")
        }

Design:
    - Uses chat_json() instead of chat() to guarantee valid JSON output
    - Merges new action items with previously found ones — never loses
      items that scroll out of the context window
    - Deduplicates by task description to avoid repeats
    - Falls back to previous action items if the API call fails
"""

from loguru import logger

from agents.base_agent import BaseAgent
from memory.context_store import ContextSnapshot


class ActionAgent(BaseAgent):
    """
    Extracts structured action items from the current meeting context.

    Unlike TopicAgent which is stateless, ActionAgent accumulates items
    across ticks — tasks mentioned early in the meeting are not lost
    when they scroll out of the context window.

    Returns:
        list[dict] — list of action item dicts with person/task/deadline,
                     or previous items if the current run produces nothing.
    """

    SYSTEM_PROMPT = """You are an expert at extracting action items and tasks \
from business meeting conversations.

Your task is to identify every action item, task, or commitment mentioned.

Instructions:
- Extract ONLY explicitly mentioned tasks — do not infer or assume
- For each action item identify:
    person:   who is responsible (use exact name if mentioned, else "Unassigned")
    task:     what needs to be done (clear, concise description)
    deadline: when it is due (exact date/time if mentioned, else "Not specified")
- If no action items are found, return an empty list
- Output ONLY a valid JSON object in exactly this format:

{
  "action_items": [
    {
      "person": "Alice",
      "task": "Send the Q3 report to stakeholders",
      "deadline": "Friday"
    },
    {
      "person": "Bob",
      "task": "Review the backend migration plan",
      "deadline": "Not specified"
    }
  ]
}

Rules:
- The JSON must always have the key "action_items" containing a list
- Each item must have exactly the keys: person, task, deadline
- Do not add extra keys or fields
- Do not include commentary, explanation, or markdown — JSON only"""

    def __init__(self) -> None:
        super().__init__()
        # Accumulated list of all action items found so far in the meeting.
        # New items are merged in on each run — old items are never discarded.
        self._all_action_items: list[dict] = []

    # ── Core method ───────────────────────────────────────────────────────────

    def run(self, snapshot: ContextSnapshot) -> list[dict]:
        """
        Extract action items from the current context window and merge
        them with all previously found items.

        Uses chat_json() to guarantee valid JSON — no string parsing needed.

        Args:
            snapshot: Current ContextSnapshot from the context store.
                      Uses snapshot.recent_context as input.

        Returns:
            Deduplicated list of all action items found so far in the
            meeting. Returns previous items if the API call fails or
            returns no new items.
        """
        user_prompt = (
            f"Extract all action items from this meeting excerpt:\n\n"
            f"{snapshot.recent_context}"
        )

        messages = self._build_messages(
            system=self.SYSTEM_PROMPT,
            user=user_prompt,
        )

        # chat_json() guarantees a dict back — no JSONDecodeError possible
        response_dict = self._client.chat_json(
            messages,
            model=None,        # uses settings.llm.extraction_model (gemini-2.0-flash)
            temperature=None,  # uses settings.llm.extraction_temperature (0.0)
        )

        new_items = self._parse_response(response_dict)

        if new_items:
            self._merge(new_items)
            logger.debug(
                f"ActionAgent | new={len(new_items)} | "
                f"total={len(self._all_action_items)}"
            )
        else:
            logger.debug("ActionAgent | no new action items in this excerpt.")

        return list(self._all_action_items)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _parse_response(self, response_dict: dict) -> list[dict]:
        """
        Validate and extract action items from the JSON response dict.

        Handles malformed responses gracefully — if the model returns
        unexpected structure, logs a warning and returns empty list
        rather than crashing the pipeline.

        Args:
            response_dict: Parsed JSON dict from chat_json().

        Returns:
            List of validated action item dicts. Empty list if the
            response is malformed or contains no items.
        """
        raw_items = response_dict.get("action_items", [])

        if not isinstance(raw_items, list):
            logger.warning(
                f"ActionAgent: 'action_items' is not a list — "
                f"got {type(raw_items).__name__}. Skipping."
            )
            return []

        validated = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue

            # Ensure all required keys are present with string values
            person   = str(item.get("person",   "Unassigned")).strip()
            task     = str(item.get("task",     "")).strip()
            deadline = str(item.get("deadline", "Not specified")).strip()

            # Skip items with no meaningful task description
            if not task:
                continue

            validated.append({
                "person":   person   or "Unassigned",
                "task":     task,
                "deadline": deadline or "Not specified",
            })

        return validated

    def _merge(self, new_items: list[dict]) -> None:
        """
        Merge new action items into the accumulated list, deduplicating
        by task description (case-insensitive).

        If a task already exists in the accumulated list, it is NOT
        added again — prevents duplicates when the same task appears
        in overlapping context windows across multiple ticks.

        Args:
            new_items: Validated action items from the current tick.
        """
        existing_tasks = {
            item["task"].lower().strip()
            for item in self._all_action_items
        }

        for item in new_items:
            task_key = item["task"].lower().strip()
            if task_key not in existing_tasks:
                self._all_action_items.append(item)
                existing_tasks.add(task_key)
                logger.debug(f"New action item added: {item}")

    def reset(self) -> None:
        """
        Clear all accumulated action items. Call between meetings.
        """
        self._all_action_items = []
        logger.info("ActionAgent reset.")
