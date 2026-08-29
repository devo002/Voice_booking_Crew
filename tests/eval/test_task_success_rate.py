"""
Task success rate: does BookingFlow end up in the *correct* terminal state
for each scenario -- not just "booked" every time, but correctly rejecting
an invalid request too? Calls real LLMs (three agents' worth per
self-correction scenario), so marked "eval" and excluded from a plain
`pytest` run -- run explicitly with:

    pytest tests/eval/test_task_success_rate.py -v
    pytest -m eval -v          # both eval suites
"""

from __future__ import annotations

import json

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from booking_crew.flow import BookingFlow
from booking_crew.mock_calendar import CALENDAR

from .conftest import requires_llm
from .datasets import build_task_scenarios
from .metrics import TaskOutcomeMatch

pytestmark = pytest.mark.eval


def _run_scenario(scenario: dict) -> dict:
    if scenario["seed_conflict"]:
        CALENDAR.seed_conflict(scenario["seed_conflict"]["date"], scenario["seed_conflict"]["time"])

    flow = BookingFlow()
    flow.kickoff(inputs=scenario["inputs"])
    return {"final_status": flow.state.final_status, "attempts": flow.state.attempts}


@requires_llm
@pytest.mark.parametrize("scenario", build_task_scenarios(), ids=lambda s: s["name"])
def test_task_success_rate(scenario: dict):
    actual = _run_scenario(scenario)

    test_case = LLMTestCase(
        input=json.dumps(scenario["inputs"]),
        actual_output=json.dumps(actual),
        expected_output=json.dumps(scenario),
    )
    assert_test(test_case, [TaskOutcomeMatch(threshold=1.0)])
