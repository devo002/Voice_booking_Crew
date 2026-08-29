"""
Golden datasets for the eval suite. Expected dates are computed relative to
`date.today()` at collection time, not hardcoded -- the Intake Agent's
relative-date resolution (see flow.py's build_intake_task) is instructed
relative to "today", so a fixed golden date would silently go stale the
day after it was written.
"""

from __future__ import annotations

from datetime import date, timedelta

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _next_occurrence(today: date, target_weekday: int, include_today: bool = True) -> date:
    days_ahead = (target_weekday - today.weekday()) % 7
    if days_ahead == 0 and not include_today:
        days_ahead = 7
    return today + timedelta(days=days_ahead)


def _following_week_occurrence(today: date, target_weekday: int) -> date:
    """Matches the "'next <weekday>' skips ahead to the following week's
    occurrence" instruction in flow.py's Intake task prompt."""
    nearest = _next_occurrence(today, target_weekday, include_today=False)
    return nearest + timedelta(days=7)


def build_intake_cases() -> list[dict]:
    """
    Each case: a raw caller utterance, plus what a correct extraction
    looks like. `expected` fields are checked for exact match;
    `title_keywords`/`constraints_keywords` are checked as
    case-insensitive substrings, since free-text phrasing legitimately
    varies (see tests/eval/metrics.py:SlotFieldAccuracy).
    """
    today = date.today()
    tomorrow = today + timedelta(days=1)
    this_friday = _next_occurrence(today, WEEKDAYS.index("Friday"), include_today=True)
    next_tuesday = _following_week_occurrence(today, WEEKDAYS.index("Tuesday"))
    exact_date = today + timedelta(days=200)  # far enough out to never be "in the past"

    return [
        {
            "name": "exact_date_and_time",
            "raw_request": (
                f"Hi, I'd like to book a 30 minute dentist appointment on "
                f"{exact_date.isoformat()} at 2pm."
            ),
            "expected": {"date": exact_date.isoformat(), "time": "14:00", "duration_minutes": 30},
            "title_keywords": ["dentist"],
            "constraints_keywords": [],
        },
        {
            "name": "relative_tomorrow",
            "raw_request": "Book a haircut tomorrow at 10am.",
            "expected": {"date": tomorrow.isoformat(), "time": "10:00", "duration_minutes": 30},
            "title_keywords": ["haircut"],
            "constraints_keywords": [],
        },
        {
            "name": "bare_weekday",
            "raw_request": "Schedule a call with the accountant this Friday at 3pm.",
            "expected": {"date": this_friday.isoformat(), "time": "15:00", "duration_minutes": 30},
            "title_keywords": ["accountant"],
            "constraints_keywords": [],
        },
        {
            "name": "next_weekday_skips_nearest",
            "raw_request": "Set up a 20 minute checkup next Tuesday at 11am.",
            "expected": {
                "date": next_tuesday.isoformat(),
                "time": "11:00",
                "duration_minutes": 20,
            },
            "title_keywords": ["checkup"],
            "constraints_keywords": [],
        },
        {
            "name": "constraints_captured",
            "raw_request": (
                f"Book a 45 minute team sync on {exact_date.isoformat()}, sometime "
                "in the morning -- doesn't need to be exact."
            ),
            "expected": {"date": exact_date.isoformat(), "duration_minutes": 45},
            "title_keywords": ["team", "sync"],
            "constraints_keywords": ["morning"],
        },
        {
            "name": "default_duration_when_unstated",
            "raw_request": f"Book a call with Alex on {exact_date.isoformat()} at 9am.",
            "expected": {"date": exact_date.isoformat(), "time": "09:00", "duration_minutes": 30},
            "title_keywords": ["alex"],
            "constraints_keywords": [],
        },
    ]


def build_task_scenarios() -> list[dict]:
    """
    End-to-end BookingFlow scenarios. `expected_status` is the *correct*
    terminal state for that scenario -- not always "booked": rejecting an
    invalid request correctly is as much a success as booking a valid one.
    """
    today = date.today()
    future_date = (today + timedelta(days=200)).isoformat()
    past_date = (today - timedelta(days=5)).isoformat()

    return [
        {
            "name": "clean_booking",
            "inputs": {
                "date": future_date,
                "time": "10:00",
                "duration_minutes": 30,
                "title": "Eval test booking",
                "constraints": "",
            },
            "seed_conflict": None,
            "expected_status": "booked",
            "expect_correction": False,
        },
        {
            "name": "conflict_triggers_self_correction",
            "inputs": {
                "date": future_date,
                "time": "11:00",
                "duration_minutes": 30,
                "title": "Eval test booking (conflict)",
                "constraints": "",
            },
            "seed_conflict": {"date": future_date, "time": "11:00"},
            "expected_status": "booked",
            "expect_correction": True,
        },
        {
            "name": "past_date_rejected_without_retry",
            "inputs": {
                "date": past_date,
                "time": "10:00",
                "duration_minutes": 30,
                "title": "Eval test booking (past)",
                "constraints": "",
            },
            "seed_conflict": None,
            "expected_status": "failed",
            "expect_correction": False,
        },
        {
            "name": "any_hour_is_bookable",
            "inputs": {
                "date": future_date,
                "time": "22:30",
                "duration_minutes": 30,
                "title": "Eval test booking (late)",
                "constraints": "",
            },
            "seed_conflict": None,
            "expected_status": "booked",
            "expect_correction": False,
        },
    ]
