"""
A tiny in-memory calendar standing in for a real backend (Google Calendar,
Cal.com, etc). `seed_conflict` pre-books a slot so the "slot taken"
self-correction path can be demoed on demand without a real calendar
account; it's opt-in, not run automatically, so it won't collide with
real bookings made during normal use.

State lives only in `_bookings` for the life of the process -- restarting
the server (including a `--reload` auto-restart on code changes) clears
every booking, seeded or real.

Rejects any date before today (`_is_before_today`) as a non-correctable
"error" -- it's not something the Resolver Agent should try to fix with a
different time on the same day, so the flow fails fast instead of
looping. There's no business-hours restriction: any HH:MM is bookable.

Swap this module out for a real API client later -- the tools in tools.py
only depend on the four methods below, so nothing upstream (agents, flow)
needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

SLOT_MINUTES = 30


@dataclass
class Booking:
    date: str  # "YYYY-MM-DD"
    time: str  # "HH:MM", 24h
    duration_minutes: int
    title: str


class MockCalendar:
    def __init__(self) -> None:
        self._bookings: dict[tuple[str, str], Booking] = {}

    # ---- seeding -----------------------------------------------------

    def seed_conflict(self, date: str, time: str, title: str = "Existing meeting") -> None:
        """Pre-book a slot so a demo request against it triggers a conflict."""
        self._bookings[(date, time)] = Booking(date, time, SLOT_MINUTES, title)

    # ---- inspection -----------------------------------------------------

    def all_bookings(self) -> list[dict]:
        """Every booking currently held, sorted by date then time."""
        return [
            {
                "date": b.date,
                "time": b.time,
                "duration_minutes": b.duration_minutes,
                "title": b.title,
            }
            for b in sorted(self._bookings.values(), key=lambda b: (b.date, b.time))
        ]

    # ---- core operations ----------------------------------------------

    def is_available(self, date: str, time: str) -> bool:
        if self._is_before_today(date):
            return False
        if not self._is_valid_time(time):
            return False
        return (date, time) not in self._bookings

    def book(self, date: str, time: str, title: str, duration_minutes: int = SLOT_MINUTES) -> dict:
        """
        Returns a structured result -- never raises for a plain conflict.
        This structure is what makes the failure "correctable": the caller
        (an LLM tool call) gets a machine-readable reason, not just an
        exception it has to guess about.
        """
        if self._is_before_today(date):
            return {
                "status": "error",
                "reason": "date_in_past",
                "detail": f"{date} is in the past "
                f"(today is {datetime.now().date().isoformat()}).",
            }
        if not self._is_valid_time(time):
            return {
                "status": "error",
                "reason": "invalid_time",
                "detail": f"'{time}' isn't a valid 24-hour HH:MM time.",
            }
        if not self.is_available(date, time):
            existing = self._bookings[(date, time)]
            return {
                "status": "conflict",
                "reason": "slot_taken",
                "detail": f"{date} {time} is already booked ({existing.title}).",
            }
        self._bookings[(date, time)] = Booking(date, time, duration_minutes, title)
        return {
            "status": "booked",
            "date": date,
            "time": time,
            "duration_minutes": duration_minutes,
            "title": title,
        }

    def cancel(self, date: str, time: str) -> bool:
        """Removes a booking if one exists at (date, time). Returns whether
        anything was actually removed."""
        return self._bookings.pop((date, time), None) is not None

    def find_nearby_available(
        self,
        date: str,
        time: str,
        count: int = 3,
        window_hours: int = 4,
        exclude: list[str] | None = None,
    ) -> list[str]:
        """
        Returns up to `count` free "HH:MM" slots on `date`, ordered by
        closeness to the requested `time`, skipping anything in `exclude`
        (e.g. slots already proposed/tried in this correction loop, so we
        never suggest the same rejected slot twice).
        """
        exclude = set(exclude or [])
        anchor = datetime.strptime(time, "%H:%M")
        candidates: list[tuple[float, str]] = []

        # No business-hours restriction -- search the full day (00:00-23:30).
        cursor = datetime.strptime("00:00", "%H:%M")
        end = cursor + timedelta(days=1)
        while cursor < end:
            slot = cursor.strftime("%H:%M")
            distance_hours = abs((cursor - anchor).total_seconds()) / 3600
            if (
                slot not in exclude
                and self.is_available(date, slot)
                and distance_hours <= window_hours
            ):
                candidates.append((distance_hours, slot))
            cursor += timedelta(minutes=SLOT_MINUTES)

        candidates.sort(key=lambda pair: pair[0])
        return [slot for _, slot in candidates[:count]]

    # ---- helpers --------------------------------------------------------

    def _is_valid_time(self, time: str) -> bool:
        try:
            datetime.strptime(time, "%H:%M")
        except ValueError:
            return False
        return True

    def _is_before_today(self, date: str) -> bool:
        try:
            d = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            return False
        return d < datetime.now().date()


# Module-level singleton so every tool instance (each Agent/Task creates
# its own Tool object) reads and writes the same calendar state.
CALENDAR = MockCalendar()
