"""
Custom DeepEval metrics for this project. Both are plain field comparisons
rather than LLM-as-judge (GEval) -- the things being checked (dates, times,
status strings) have one objectively correct answer, so spending an extra
LLM call to "judge" them would be slower and less reliable than just
comparing values directly. DeepEval's role here is the test-case/metric
harness and reporting, not the judging.
"""

from __future__ import annotations

import json

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class SlotFieldAccuracy(BaseMetric):
    """
    Fraction of expected criteria the Intake Agent's extraction satisfied.
    `test_case.expected_output` is a golden-case dict (see datasets.py)
    with:
      - "expected": {field: exact_value, ...} -- checked for exact match
      - "title_keywords" / "constraints_keywords": lists of substrings
        that must all appear (case-insensitive) in the corresponding
        actual field; an empty list means that field must itself be
        empty/falsy (used to assert "no constraints stated" is honored).
    `test_case.actual_output` is the Intake Agent's ParsedRequest, as JSON.

    LLMTestCase requires actual_output/expected_output to be plain
    strings, so both are passed in JSON-encoded and decoded back here.
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        golden = json.loads(test_case.expected_output)
        actual = json.loads(test_case.actual_output)

        checks: list[bool] = []
        for field, expected_value in golden["expected"].items():
            checks.append(str(actual.get(field, "")).strip() == str(expected_value).strip())

        for field, keyword_list in (
            ("title", golden.get("title_keywords", [])),
            ("constraints", golden.get("constraints_keywords", [])),
        ):
            actual_value = str(actual.get(field, "")).strip().lower()
            if keyword_list:
                checks.append(all(kw.lower() in actual_value for kw in keyword_list))
            else:
                checks.append(actual_value == "")

        self.score = sum(checks) / len(checks) if checks else 0.0
        self.success = self.score >= self.threshold
        self.reason = f"{sum(checks)}/{len(checks)} extraction criteria matched"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "Slot Field Accuracy"


class TaskOutcomeMatch(BaseMetric):
    """
    Did BookingFlow reach the *correct* terminal state for the scenario?
    "Correct" isn't always "booked" -- rejecting an invalid request (e.g.
    a past date) is the right outcome for that scenario, so this checks
    against each scenario's own `expected_status`/`expect_correction`
    rather than a single pass/fail shape.

    Same JSON-encoded-string convention as SlotFieldAccuracy above.
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        golden = json.loads(test_case.expected_output)
        actual = json.loads(test_case.actual_output)

        status_ok = actual["final_status"] == golden["expected_status"]
        correction_ok = (actual["attempts"] > 1) == golden["expect_correction"]

        checks = [status_ok, correction_ok]
        self.score = sum(checks) / len(checks)
        self.success = self.score >= self.threshold
        self.reason = (
            f"status: got {actual['final_status']!r}, expected {golden['expected_status']!r} "
            f"({'ok' if status_ok else 'MISMATCH'}); "
            f"self-correction: got {actual['attempts'] > 1}, "
            f"expected {golden['expect_correction']} "
            f"({'ok' if correction_ok else 'MISMATCH'})"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "Task Outcome Match"
