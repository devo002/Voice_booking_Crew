"""
Unit tests for BookingFlow's orchestration logic (the retry loop, terminal
states, and hand-offs between steps) with no LLM involved.

Crew and the agent builders are replaced by scripted fakes inside
booking_crew.flow, so the real run() / _attempt_booking() / _resolve_conflict()
/ finalizer code all executes -- only the model calls are faked.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from booking_crew import flow as flow_module
from booking_crew.flow import BookingFlow
from booking_crew.schemas import ParsedRequest, ResolutionProposal

BOOKED_10 = {"status": "booked", "date": "2026-10-26", "time": "10:00"}
BOOKED_0930 = {"status": "booked", "date": "2026-10-26", "time": "09:30"}
CONFLICT = {"status": "conflict", "reason": "slot_taken", "detail": "slot is already booked."}
PAST = {"status": "error", "reason": "date_in_past", "detail": "2026-01-01 is in the past."}
BAD_TIME = {"status": "error", "reason": "invalid_time", "detail": "'25:99' isn't valid."}

FORM_INPUTS = {
    "date": "2026-10-26",
    "time": "10:00",
    "title": "visa appointment",
    "constraints": "must be in the morning",
}


class Script:
    """Scripted model outputs plus a record of every Crew.kickoff call."""

    def __init__(self):
        self.intake: ParsedRequest | None = None
        self.bookings: list[dict] = []
        self.proposals: list[ResolutionProposal] = []
        self.calls: list[tuple[str, dict]] = []

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.calls]

    def inputs_for(self, kind: str) -> list[dict]:
        return [inputs for k, inputs in self.calls if k == kind]


@pytest.fixture
def script(monkeypatch) -> Script:
    s = Script()

    class FakeCrew:
        def __init__(self, agents, tasks, process=None, **kwargs):
            self.task = tasks[0]

        def kickoff(self, inputs=None):
            expected = self.task.expected_output
            if "ParsedRequest" in expected:
                s.calls.append(("intake", inputs))
                return SimpleNamespace(pydantic=s.intake)
            if "book_slot" in expected:
                s.calls.append(("scheduler", inputs))
                return SimpleNamespace(raw=json.dumps(s.bookings.pop(0)))
            if "ResolutionProposal" in expected:
                s.calls.append(("resolver", inputs))
                return SimpleNamespace(pydantic=s.proposals.pop(0))
            raise AssertionError(f"unexpected task: {expected!r}")

    monkeypatch.setattr(flow_module, "Crew", FakeCrew)
    for builder in ("build_intake_agent", "build_scheduler_agent", "build_resolver_agent"):
        monkeypatch.setattr(flow_module, builder, lambda: None)
    return s


def run_flow(inputs: dict) -> tuple[BookingFlow, list[str]]:
    messages: list[str] = []
    flow = BookingFlow(notify=messages.append)
    flow.kickoff(inputs=inputs)
    return flow, messages


# ---- happy path -----------------------------------------------------------


def test_free_slot_books_on_first_attempt(script):
    script.bookings = [BOOKED_10]

    flow, messages = run_flow(FORM_INPUTS)

    assert flow.state.final_status == "booked"
    assert flow.state.attempts == 1
    assert script.kinds() == ["scheduler"]
    assert messages == ["You're booked for 2026-10-26 at 10:00."]


def test_form_input_skips_intake(script):
    script.bookings = [BOOKED_10]

    run_flow(FORM_INPUTS)

    assert "intake" not in script.kinds()


def test_scheduler_receives_state_fields(script):
    script.bookings = [BOOKED_10]

    run_flow({**FORM_INPUTS, "duration_minutes": 45})

    assert script.inputs_for("scheduler") == [
        {"date": "2026-10-26", "time": "10:00", "title": "visa appointment", "duration_minutes": 45}
    ]


# ---- intake ---------------------------------------------------------------


def test_free_text_runs_intake_then_books(script):
    script.intake = ParsedRequest(
        date="2026-10-26", time="10:00", title="visa appointment", constraints="must be in the morning"
    )
    script.bookings = [BOOKED_10]

    flow, _ = run_flow({"raw_request": "book my visa appointment on 26.10 at 10 in the morning"})

    assert script.kinds() == ["intake", "scheduler"]
    assert flow.state.date == "2026-10-26"
    assert flow.state.title == "visa appointment"
    assert flow.state.constraints == "must be in the morning"
    assert flow.state.final_status == "booked"


def test_missing_time_asks_caller_and_never_books(script):
    script.intake = ParsedRequest(date="2026-10-26", time="", title="visa appointment")

    flow, messages = run_flow({"raw_request": "visa appointment on the 26th"})

    assert flow.state.final_status == "failed"
    assert flow.state.attempts == 0
    assert script.kinds() == ["intake"]
    assert messages == ["What time on 2026-10-26 works for you?"]


def test_missing_date_and_time_asks_for_both(script):
    script.intake = ParsedRequest(date="", time="", title="appointment")

    flow, messages = run_flow({"raw_request": "book me something"})

    assert flow.state.final_status == "failed"
    assert script.kinds() == ["intake"]
    assert messages == ["What date and time works for you?"]


# ---- conflict -> resolver -> retry ---------------------------------------


def test_conflict_then_resolver_then_success(script):
    script.bookings = [CONFLICT, BOOKED_0930]
    script.proposals = [ResolutionProposal(chosen_time="09:30", reasoning="closest morning slot")]

    flow, messages = run_flow(FORM_INPUTS)

    assert flow.state.final_status == "booked"
    assert flow.state.attempts == 2
    assert flow.state.time == "09:30"
    assert flow.state.tried_times == ["10:00", "09:30"]
    assert script.kinds() == ["scheduler", "resolver", "scheduler"]
    assert messages == [
        "10:00 is already booked. Booking 09:30 instead.",
        "You're booked for 2026-10-26 at 09:30. Let me know if that doesn't work and I'll find another time.",
    ]


def test_second_scheduler_attempt_uses_resolved_time(script):
    script.bookings = [CONFLICT, BOOKED_0930]
    script.proposals = [ResolutionProposal(chosen_time="09:30", reasoning="r")]

    run_flow(FORM_INPUTS)

    times = [inputs["time"] for inputs in script.inputs_for("scheduler")]
    assert times == ["10:00", "09:30"]


def test_resolver_is_told_the_conflict_constraints_and_tried_times(script):
    script.bookings = [CONFLICT, BOOKED_0930]
    script.proposals = [ResolutionProposal(chosen_time="09:30", reasoning="r")]

    run_flow(FORM_INPUTS)

    [resolver_inputs] = script.inputs_for("resolver")
    assert resolver_inputs["date"] == "2026-10-26"
    assert resolver_inputs["time"] == "10:00"
    assert resolver_inputs["reason"] == "slot_taken"
    assert resolver_inputs["detail"] == "slot is already booked."
    assert resolver_inputs["constraints"] == "must be in the morning"
    assert resolver_inputs["tried_times"] == "10:00"


def test_resolver_gets_none_when_no_constraints(script):
    script.bookings = [CONFLICT, BOOKED_0930]
    script.proposals = [ResolutionProposal(chosen_time="09:30", reasoning="r")]

    run_flow({"date": "2026-10-26", "time": "10:00", "title": "t"})

    [resolver_inputs] = script.inputs_for("resolver")
    assert resolver_inputs["constraints"] == "none"


def test_tried_times_accumulate_across_multiple_conflicts(script):
    script.bookings = [CONFLICT, CONFLICT, BOOKED_0930]
    script.proposals = [
        ResolutionProposal(chosen_time="09:30", reasoning="r1"),
        ResolutionProposal(chosen_time="09:00", reasoning="r2"),
    ]

    flow, _ = run_flow(FORM_INPUTS)

    resolver_tried = [i["tried_times"] for i in script.inputs_for("resolver")]
    assert resolver_tried == ["10:00", "10:00, 09:30"]
    assert flow.state.attempts == 3


# ---- bounded retries ------------------------------------------------------


def test_stops_after_max_attempts(script):
    script.bookings = [CONFLICT, CONFLICT, CONFLICT]
    script.proposals = [
        ResolutionProposal(chosen_time="09:30", reasoning="r1"),
        ResolutionProposal(chosen_time="09:00", reasoning="r2"),
    ]

    flow, messages = run_flow(FORM_INPUTS)

    assert flow.state.final_status == "failed"
    assert flow.state.attempts == 3
    assert script.kinds().count("scheduler") == 3
    assert script.kinds().count("resolver") == 2
    assert script.kinds()[-1] == "scheduler"
    assert "I couldn't find a slot for 2026-10-26 after 3 attempt(s)" in messages[-1]


def test_max_attempts_of_one_never_calls_resolver(script):
    script.bookings = [CONFLICT]

    flow, _ = run_flow({**FORM_INPUTS, "max_attempts": 1})

    assert flow.state.final_status == "failed"
    assert flow.state.attempts == 1
    assert "resolver" not in script.kinds()


# ---- non-correctable errors ----------------------------------------------


def test_past_date_fails_without_retry(script):
    script.bookings = [PAST]

    flow, messages = run_flow({**FORM_INPUTS, "date": "2026-01-01"})

    assert flow.state.final_status == "failed"
    assert flow.state.attempts == 1
    assert script.kinds() == ["scheduler"]
    assert messages == ["2026-01-01 is in the past -- what date did you mean?"]


def test_other_errors_fail_without_retry(script):
    script.bookings = [BAD_TIME]

    flow, messages = run_flow({**FORM_INPUTS, "time": "25:99"})

    assert flow.state.final_status == "failed"
    assert flow.state.attempts == 1
    assert "resolver" not in script.kinds()
    assert "I couldn't find a slot" in messages[-1]


# ---- log ------------------------------------------------------------------


def test_log_records_each_step(script):
    script.bookings = [CONFLICT, BOOKED_0930]
    script.proposals = [ResolutionProposal(chosen_time="09:30", reasoning="closest morning slot")]

    flow, _ = run_flow(FORM_INPUTS)

    log = flow.state.log
    assert log[0].startswith("Attempt 1: booking 2026-10-26 10:00 -> conflict")
    assert log[1] == "Resolver proposes 09:30 -- closest morning slot"
    assert log[2].startswith("Attempt 2: booking 2026-10-26 09:30 -> booked")
    assert log[3] == "DONE: booked 2026-10-26 09:30 after 2 attempt(s)."
