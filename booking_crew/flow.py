"""
The orchestration layer. This is where "self-healing" actually lives at
the system level: try to book -> if it fails with a *correctable* reason,
get an alternative and try again -> stop after a bounded number of
attempts rather than looping forever.

We use a CrewAI Flow for the entry point (explicit, typed `BookingState`,
inspectable in CrewAI's tracing/observability tooling) but the retry loop
itself is a plain Python `while` inside one `@start` method, not a chain
of `@listen`/`@router` steps calling each other in a cycle. That's a
deliberate choice, not a stylistic one: CrewAI's Flow engine runs each
decorated method at most once per flow execution -- `@listen`/`@router`
model a one-shot DAG, so a decorated method can't be re-entered to form a
loop. A `while` loop that calls each agent's Crew.kickoff() on every
iteration is the documented way to get bounded, variable-length retries
inside a Flow. If you later split "try booking" and "resolve conflict"
across process boundaries (e.g. a human approval step in between), revisit
this with `kickoff_for_each` or a queue instead of a plain loop.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Callable

from crewai import Crew, Process, Task
from crewai.flow.flow import Flow, start
from pydantic import BaseModel

from booking_crew.agents import (
    build_intake_agent,
    build_resolver_agent,
    build_scheduler_agent,
)
from booking_crew.guardrails import booking_output_guardrail
from booking_crew.schemas import BookingAttemptResult, ParsedRequest, ResolutionProposal


def build_intake_task(agent) -> Task:
    """
    Factored out of BookingFlow._intake so the eval suite
    (tests/eval/test_intake_accuracy.py) can run the exact same prompt
    against the Intake Agent that production uses, instead of a copy that
    could silently drift out of sync.
    """
    return Task(
        description=(
            "Today's date is {today} ({weekday}).\n\n"
            "A caller said:\n\n\"{raw_request}\"\n\n"
            "Extract the booking they want as structured fields. If "
            "the caller gave a relative or bare day instead of an "
            "exact date (e.g. 'Sunday', 'next Tuesday', 'tomorrow'), "
            "resolve it to an absolute YYYY-MM-DD date using today's "
            "date above: a bare weekday name means the next occurrence "
            "of that day (today counts if today is that day); "
            "'next <weekday>' skips ahead to the following week's "
            "occurrence. Never resolve to a date before today. If no "
            "duration was stated, default to 30 minutes. If no "
            "constraints were stated, set constraints to an empty "
            "string."
        ),
        expected_output="A ParsedRequest JSON object.",
        agent=agent,
        output_pydantic=ParsedRequest,
    )


def intake_inputs(raw_request: str, today: date) -> dict:
    return {
        "raw_request": raw_request,
        "today": today.isoformat(),
        "weekday": today.strftime("%A"),
    }


class BookingState(BaseModel):
    raw_request: str = ""

    date: str = ""
    time: str = ""
    duration_minutes: int = 30
    title: str = ""
    constraints: str = ""

    attempts: int = 0
    max_attempts: int = 3
    tried_times: list[str] = []

    last_result: dict = {}
    final_status: str = ""  # "booked" | "failed"
    log: list[str] = []


class BookingFlow(Flow[BookingState]):
    """
    `notify` is the seam for surfacing corrections to the caller as they
    happen, rather than only in the end-of-run log. Text-only for now (it
    defaults to printing), but it's the exact hook a future voice adapter
    replaces with TTS -- the flow shouldn't need to change, just what's
    passed in here.
    """

    def __init__(self, notify: Callable[[str], None] | None = None, **kwargs):
        super().__init__(**kwargs)
        self._notify = notify or (lambda message: print(f"[assistant] {message}"))

    @start()
    def run(self):
        # A caller's free text needs the Intake Agent to turn it into
        # structured fields. A structured front door (e.g. a web form)
        # already has date/time/title -- skip the extra LLM call and go
        # straight to attempting the booking.
        if self.state.raw_request:
            self._intake()
        while True:
            self._attempt_booking()
            status = self.state.last_result.get("status")

            if status == "booked":
                self._finalize_success()
                return

            correctable = status == "conflict"  # add more correctable
            # reasons here as you extend this system -- e.g. a future
            # "double_booked_elsewhere" check could auto-shift to the
            # nearest free slot instead of asking the Resolver for
            # alternatives.

            if correctable and self.state.attempts < self.state.max_attempts:
                self._resolve_conflict()
                continue

            self._finalize_failure()
            return

    # ---- step 1: understand the request --------------------------------

    def _intake(self):
        agent = build_intake_agent()
        today = date.today()
        task = build_intake_task(agent)
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff(inputs=intake_inputs(self.state.raw_request, today))
        parsed: ParsedRequest = result.pydantic

        self.state.date = parsed.date
        self.state.time = parsed.time
        self.state.duration_minutes = parsed.duration_minutes
        self.state.title = parsed.title
        self.state.constraints = parsed.constraints
        self.state.log.append(
            f"Intake parsed: {parsed.date} {parsed.time} '{parsed.title}' "
            f"(constraints: {parsed.constraints or 'none'})"
        )

    # ---- step 2: try to book the current candidate slot -----------------

    def _attempt_booking(self):
        self.state.attempts += 1
        self.state.tried_times.append(self.state.time)

        agent = build_scheduler_agent()
        task = Task(
            description=(
                "Book this exact slot using the book_slot tool:\n"
                "date={date}, time={time}, title={title}, "
                "duration_minutes={duration_minutes}\n\n"
                "Call the tool once. Your final answer must be exactly the "
                "JSON the tool returned -- nothing else."
            ),
            expected_output="The raw JSON result from book_slot.",
            agent=agent,
            guardrail=booking_output_guardrail,
            guardrail_max_retries=2,
        )
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff(
            inputs={
                "date": self.state.date,
                "time": self.state.time,
                "title": self.state.title,
                "duration_minutes": self.state.duration_minutes,
            }
        )
        data = json.loads(result.raw)
        self.state.last_result = data
        self.state.log.append(
            f"Attempt {self.state.attempts}: booking {self.state.date} "
            f"{self.state.time} -> {data.get('status')}"
            + (f" ({data.get('detail')})" if data.get("detail") else "")
        )

    # ---- step 3: find and adopt an alternative -------------------------

    def _resolve_conflict(self):
        rejected_time = self.state.time
        agent = build_resolver_agent()
        task = Task(
            description=(
                "The requested slot {date} {time} was unavailable "
                "(reason: {reason} - {detail}). Caller constraints: "
                "{constraints}\n\n"
                "Already-tried times today (never re-propose these): "
                "{tried_times}\n\n"
                "Use find_alternative_slots to see what's open on {date}, "
                "then choose the single best alternative given the "
                "caller's constraints."
            ),
            expected_output="A ResolutionProposal JSON object.",
            agent=agent,
            output_pydantic=ResolutionProposal,
        )
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff(
            inputs={
                "date": self.state.date,
                "time": self.state.time,
                "reason": self.state.last_result.get("reason", ""),
                "detail": self.state.last_result.get("detail", ""),
                "constraints": self.state.constraints or "none",
                "tried_times": ", ".join(self.state.tried_times),
            }
        )
        proposal: ResolutionProposal = result.pydantic

        self.state.log.append(
            f"Resolver proposes {proposal.chosen_time} -- {proposal.reasoning}"
        )
        self._notify(
            f"{rejected_time} is already booked. Booking {proposal.chosen_time} instead."
        )
        self.state.time = proposal.chosen_time

    # ---- terminal states -------------------------------------------------

    def _finalize_success(self):
        self.state.final_status = "booked"
        self.state.log.append(
            f"DONE: booked {self.state.date} {self.state.time} "
            f"after {self.state.attempts} attempt(s)."
        )
        confirmation = f"You're booked for {self.state.date} at {self.state.time}."
        if self.state.attempts > 1:
            confirmation += " Let me know if that doesn't work and I'll find another time."
        self._notify(confirmation)

    def _finalize_failure(self):
        self.state.final_status = "failed"
        self.state.log.append(
            f"DONE: could not book after {self.state.attempts} attempt(s) "
            f"-- last result: {self.state.last_result}."
        )
        if self.state.last_result.get("reason") == "date_in_past":
            self._notify(f"{self.state.date} is in the past -- what date did you mean?")
            return
        self._notify(
            f"I couldn't find a slot for {self.state.date} after "
            f"{self.state.attempts} attempt(s). Want to try a different day "
            "or loosen the constraints?"
        )
