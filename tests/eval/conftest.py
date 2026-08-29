from __future__ import annotations

import os

# Force the in-memory mock calendar for eval runs regardless of what
# CALENDAR_BACKEND is set to in .env -- must happen before anything
# imports booking_crew.tools/calendar_backend, so eval runs never
# silently depend on (or write to) a real Supabase project.
os.environ["CALENDAR_BACKEND"] = "mock"

import pytest

from booking_crew.mock_calendar import CALENDAR

requires_llm = pytest.mark.skipif(
    not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")),
    reason="Needs a real LLM key in .env to run agents against a live model.",
)


@pytest.fixture(autouse=True)
def reset_calendar():
    """
    CALENDAR is a module-level singleton (see mock_calendar.py) shared by
    every tool call in the app. Without this, one eval scenario's bookings
    would leak into the next test's expectations.
    """
    CALENDAR._bookings.clear()
    yield
    CALENDAR._bookings.clear()
