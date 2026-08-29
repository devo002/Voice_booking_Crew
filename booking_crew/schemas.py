"""Structured shapes passed between flow steps. Using Pydantic models (via
Task.output_json) instead of free-text agent output is what lets the flow's
control logic (the router) branch reliably on `status` rather than parsing
prose.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ParsedRequest(BaseModel):
    date: str = Field(description="YYYY-MM-DD")
    time: str = Field(description="HH:MM, 24-hour")
    duration_minutes: int = Field(default=30)
    title: str = Field(description="Short label for the booking")
    constraints: str = Field(
        default="",
        description="Any timing constraints the caller stated, e.g. "
        "'must stay in the afternoon' or 'none'.",
    )


class BookingAttemptResult(BaseModel):
    status: str = Field(description="one of: booked, conflict, error")
    date: str = ""
    time: str = ""
    reason: str = ""
    detail: str = ""


class ResolutionProposal(BaseModel):
    chosen_time: str = Field(description="HH:MM of the alternative to try next")
    reasoning: str = Field(
        description="Why this alternative was chosen over the others, "
        "with respect to the caller's stated constraints"
    )
