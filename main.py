"""
Entry point. Seeds the mock calendar with an existing booking so the demo
request collides with it on the first attempt -- that's what triggers the
self-correction loop (Scheduler fails -> Resolver proposes an alternative
-> Scheduler retries) instead of it just booking cleanly on try #1.

Usage:
    cp .env.example .env   # then fill in OPENAI_API_KEY
    pip install -r requirements.txt
    python main.py
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from booking_crew.flow import BookingFlow  # noqa: E402  (after load_dotenv)
from booking_crew.mock_calendar import CALENDAR  # noqa: E402

DEMO_REQUEST = (
    "Hi, I'd like to book a 30 minute dentist appointment on 2026-08-25 "
    "at 2pm. If that's not free, anytime in the afternoon works."
)


def main():
    # Pre-book 14:00 so the first attempt at the demo request conflicts.
    CALENDAR.seed_conflict("2026-08-25", "14:00", title="Existing meeting")

    flow = BookingFlow()
    flow.kickoff(inputs={"raw_request": DEMO_REQUEST})

    print("\n" + "=" * 60)
    print("SELF-CORRECTION LOG")
    print("=" * 60)
    for line in flow.state.log:
        print("-", line)

    print("\n" + "=" * 60)
    print(f"FINAL STATUS: {flow.state.final_status.upper()}")
    if flow.state.final_status == "booked":
        print(f"Booked: {flow.state.date} {flow.state.time} - {flow.state.title}")
    print("=" * 60)


if __name__ == "__main__":
    main()
