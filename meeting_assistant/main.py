"""
main.py — Entry point for the Meeting Assistant.

Run with:
    python main.py

Before running:
    1. Copy .env.example to .env and add your GEMINI_API_KEY
    2. pip install -r requirements.txt
    3. (Optional) To launch the Streamlit dashboard at the same time:
         streamlit run frontend/app.py
       in a separate terminal.
"""

from dotenv import load_dotenv
from loguru import logger
import sys

# Load .env file before importing anything that reads settings
load_dotenv()

from config import settings  # noqa: E402 — must come after load_dotenv()


def main() -> None:
    logger.info("Starting Meeting Assistant...")

    # Fail fast on misconfiguration
    try:
        settings.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Import here (not at top) so config is fully loaded first
    from pipeline.runner import PipelineRunner

    runner = PipelineRunner()

    try:
        runner.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user — saving final notes and exiting.")
        runner.shutdown()


if __name__ == "__main__":
    main()
