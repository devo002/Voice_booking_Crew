"""
Three agents, each with one job. Splitting these up (rather than one agent
with all three tools) is what makes the system easy to grow later: adding
a new failure type usually means adding a new tool to the Resolver, or a
new specialist agent, without touching how Intake or Scheduler work.
"""

from __future__ import annotations

import os

from crewai import Agent

from booking_crew.tools import book_slot, check_availability, find_alternative_slots

MODEL = os.getenv("MODEL", "gpt-4o-mini")


def build_intake_agent() -> Agent:
    return Agent(
        role="Booking Intake Specialist",
        goal=(
            "Turn a caller's natural-language scheduling request into a "
            "precise, structured booking request: date, time, duration, "
            "title, and any timing constraints they stated."
        ),
        backstory=(
            "You listen carefully to what a caller actually asked for. "
            "You never invent a date/time that wasn't stated or clearly "
            "implied, and you capture constraints like 'must be in the "
            "afternoon' or 'anytime this week' verbatim in a `constraints` "
            "field so later steps can respect them. When the caller gives "
            "a relative day (e.g. 'Sunday', 'next Tuesday', 'tomorrow') "
            "instead of an exact date, you resolve it to an absolute "
            "YYYY-MM-DD date using the current date given in the task -- "
            "never a date before today."
        ),
        llm=MODEL,
        verbose=True,
    )


def build_scheduler_agent() -> Agent:
    return Agent(
        role="Scheduling Agent",
        goal=(
            "Attempt to book the requested calendar slot and report exactly "
            "what happened -- booked, conflict, or error -- as the tool "
            "returns it."
        ),
        backstory=(
            "You call book_slot with the exact date/time/title you're "
            "given. You do not decide what to do about a conflict "
            "yourself -- that's someone else's job. You just report the "
            "tool's result faithfully, as JSON, and nothing else."
        ),
        tools=[book_slot],
        llm=MODEL,
        verbose=True,
    )


def build_resolver_agent() -> Agent:
    return Agent(
        role="Conflict Resolution Agent",
        goal=(
            "When a requested slot is taken, find the best available "
            "alternative on the same day, respecting the caller's stated "
            "constraints, and explain why you picked it."
        ),
        backstory=(
            "You are called in only after a booking conflict. You use "
            "find_alternative_slots to see what's actually open, you never "
            "re-propose a time that's already been tried and rejected in "
            "this conversation, and you weigh the caller's constraints "
            "(e.g. 'must stay in the afternoon') over simple proximity to "
            "the original time when the two disagree."
        ),
        tools=[find_alternative_slots],
        llm=MODEL,
        verbose=True,
    )
