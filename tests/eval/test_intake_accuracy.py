"""
Slot extraction accuracy: does the Intake Agent turn a caller's free text
into the right structured ParsedRequest? Calls a real LLM per case (see
conftest.requires_llm), so it's marked "eval" and excluded from a plain
`pytest` run -- run explicitly with:

    pytest tests/eval/test_intake_accuracy.py -v
    pytest -m eval -v          # both eval suites
"""

from __future__ import annotations

import json

import pytest
from crewai import Crew, Process
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from booking_crew.agents import build_intake_agent
from booking_crew.flow import build_intake_task, intake_inputs

from .conftest import requires_llm
from .datasets import build_intake_cases
from .metrics import SlotFieldAccuracy

pytestmark = pytest.mark.eval


def _run_intake(raw_request: str) -> dict:
    from datetime import date

    agent = build_intake_agent()
    task = build_intake_task(agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
    result = crew.kickoff(inputs=intake_inputs(raw_request, date.today()))
    return result.pydantic.model_dump()


@requires_llm
@pytest.mark.parametrize("case", build_intake_cases(), ids=lambda c: c["name"])
def test_slot_extraction_accuracy(case: dict):
    actual = _run_intake(case["raw_request"])

    test_case = LLMTestCase(
        input=case["raw_request"],
        actual_output=json.dumps(actual),
        expected_output=json.dumps(case),
    )
    assert_test(test_case, [SlotFieldAccuracy(threshold=0.8)])
