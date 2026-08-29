"""
CrewAI tools wrapping the active calendar backend (see
calendar_backend.py -- mock or Supabase, chosen by CALENDAR_BACKEND).
Every tool returns a JSON string with a `status` field rather than
raising on a normal business failure (slot taken, outside hours). That's
the key design choice for making self-correction possible: the agent
gets a structured signal to reason over ("status": "conflict") instead
of an opaque error it can only apologize for.
"""

from __future__ import annotations

import json

from crewai.tools import tool

from booking_crew.calendar_backend import CALENDAR


@tool("check_availability")
def check_availability(date: str, time: str) -> str:
    """
    Check whether a calendar slot is free.

    Args:
        date: Date in YYYY-MM-DD format.
        time: Time in 24-hour HH:MM format.

    Returns a JSON string: {"available": true|false}
    """
    return json.dumps({"available": CALENDAR.is_available(date, time)})


@tool("book_slot")
def book_slot(date: str, time: str, title: str, duration_minutes: int = 30) -> str:
    """
    Attempt to book a calendar slot. Does NOT raise on a conflict -- it
    returns a structured result so the caller can decide how to recover.

    Args:
        date: Date in YYYY-MM-DD format.
        time: Time in 24-hour HH:MM format.
        title: Short title/description for the booking.
        duration_minutes: Length of the meeting in minutes (default 30).

    Returns a JSON string, one of:
        {"status": "booked", "date": ..., "time": ..., ...}
        {"status": "conflict", "reason": "slot_taken", "detail": "..."}
        {"status": "error", "reason": "date_in_past", "detail": "..."}
    """
    return json.dumps(CALENDAR.book(date, time, title, duration_minutes))


@tool("find_alternative_slots")
def find_alternative_slots(
    date: str, time: str, count: int = 3, window_hours: int = 4, exclude: str = ""
) -> str:
    """
    Find nearby open slots on the same day when the requested time is
    unavailable.

    Args:
        date: Date in YYYY-MM-DD format.
        time: The originally requested time, used as the anchor point --
            results are ordered by closeness to this time.
        count: Max number of alternatives to return.
        window_hours: Only consider slots within this many hours of `time`.
        exclude: Comma-separated "HH:MM" times to skip (e.g. slots already
            proposed and rejected earlier in this conversation), so the
            same rejected time is never suggested twice.

    Returns a JSON string: {"alternatives": ["HH:MM", ...]}
    """
    excluded = [t.strip() for t in exclude.split(",") if t.strip()]
    alts = CALENDAR.find_nearby_available(
        date, time, count=count, window_hours=window_hours, exclude=excluded
    )
    return json.dumps({"alternatives": alts})
