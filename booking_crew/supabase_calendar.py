"""
Real, persisted, multi-user calendar backed by Supabase Postgres -- the
production alternative to mock_calendar.MockCalendar. Implements the same
four methods (is_available, book, find_nearby_available, cancel) plus
all_bookings, so tools.py doesn't change at all; only which CALENDAR
instance calendar_backend.py hands it changes.

Per-user scoping is the one thing MockCalendar never had to deal with.
CrewAI's @tool-decorated functions in tools.py have a fixed signature the
LLM fills in -- there's no clean way to make "whose calendar" an LLM tool
argument without risking a caller naming someone else's user_id. Instead,
`current_user_id` is a contextvar that server.py sets once per
authenticated request (from the verified JWT, never from the LLM), and
every method here reads it implicitly. CrewAI's flow runtime executes
each step via `asyncio.to_thread(ctx.run, method, ...)` with a captured
`contextvars.Context`, which is exactly the mechanism that makes
contextvars propagate correctly into that worker thread -- so this stays
correctly isolated per concurrent request without any explicit plumbing
through agents/tasks/tools.

The Row Level Security policy on the `bookings` table (auth.uid() =
user_id) is the actual enforcement boundary; the service-role client used
here bypasses RLS by design (the backend, not the end user, is the
trusted caller), so `current_user_id` filtering below is what keeps one
user's queries scoped to their own rows.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from datetime import datetime, timedelta

from supabase import Client, create_client

current_user_id: ContextVar[str | None] = ContextVar("current_user_id", default=None)

SLOT_MINUTES = 30

_client: Client | None = None


def _get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    return _client


def _require_user_id() -> str:
    user_id = current_user_id.get()
    if not user_id:
        raise RuntimeError(
            "No authenticated user set for this calendar operation -- "
            "current_user_id must be set before running BookingFlow."
        )
    return user_id


class SupabaseCalendar:
    def is_available(self, date: str, time: str) -> bool:
        if self._is_before_today(date):
            return False
        if not self._is_valid_time(time):
            return False
        resp = (
            _get_client()
            .table("bookings")
            .select("id")
            .eq("user_id", _require_user_id())
            .eq("date", date)
            .eq("time", time)
            .execute()
        )
        return len(resp.data) == 0

    def book(self, date: str, time: str, title: str, duration_minutes: int = SLOT_MINUTES) -> dict:
        if self._is_before_today(date):
            return {
                "status": "error",
                "reason": "date_in_past",
                "detail": f"{date} is in the past (today is {datetime.now().date().isoformat()}).",
            }
        if not self._is_valid_time(time):
            return {
                "status": "error",
                "reason": "invalid_time",
                "detail": f"'{time}' isn't a valid 24-hour HH:MM time.",
            }

        user_id = _require_user_id()
        client = _get_client()

        existing = (
            client.table("bookings")
            .select("title")
            .eq("user_id", user_id)
            .eq("date", date)
            .eq("time", time)
            .execute()
        )
        if existing.data:
            return {
                "status": "conflict",
                "reason": "slot_taken",
                "detail": f"{date} {time} is already booked ({existing.data[0]['title']}).",
            }

        client.table("bookings").insert(
            {
                "user_id": user_id,
                "date": date,
                "time": time,
                "duration_minutes": duration_minutes,
                "title": title,
            }
        ).execute()
        return {
            "status": "booked",
            "date": date,
            "time": time,
            "duration_minutes": duration_minutes,
            "title": title,
        }

    def cancel(self, date: str, time: str) -> bool:
        resp = (
            _get_client()
            .table("bookings")
            .delete()
            .eq("user_id", _require_user_id())
            .eq("date", date)
            .eq("time", time)
            .execute()
        )
        return len(resp.data) > 0

    def all_bookings(self) -> list[dict]:
        resp = (
            _get_client()
            .table("bookings")
            .select("date, time, duration_minutes, title")
            .eq("user_id", _require_user_id())
            .order("date")
            .order("time")
            .execute()
        )
        return resp.data

    def find_nearby_available(
        self,
        date: str,
        time: str,
        count: int = 3,
        window_hours: int = 4,
        exclude: list[str] | None = None,
    ) -> list[str]:
        """Same algorithm as MockCalendar's, but the "already booked"
        lookup is one query for the whole day instead of a dict check
        per candidate slot."""
        exclude_set = set(exclude or [])
        booked_times = {
            row["time"]
            for row in (
                _get_client()
                .table("bookings")
                .select("time")
                .eq("user_id", _require_user_id())
                .eq("date", date)
                .execute()
                .data
            )
        }

        anchor = datetime.strptime(time, "%H:%M")
        candidates: list[tuple[float, str]] = []

        cursor = datetime.strptime("00:00", "%H:%M")
        end = cursor + timedelta(days=1)
        while cursor < end:
            slot = cursor.strftime("%H:%M")
            distance_hours = abs((cursor - anchor).total_seconds()) / 3600
            if slot not in exclude_set and slot not in booked_times and distance_hours <= window_hours:
                candidates.append((distance_hours, slot))
            cursor += timedelta(minutes=SLOT_MINUTES)

        candidates.sort(key=lambda pair: pair[0])
        return [slot for _, slot in candidates[:count]]

    # ---- helpers, identical to MockCalendar's --------------------------

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
