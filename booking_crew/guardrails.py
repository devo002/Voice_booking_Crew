"""
This is CrewAI's *native* self-correction primitive: a Task can carry a
`guardrail` callable. If it returns (False, feedback), CrewAI re-runs the
same task against the same agent with that feedback appended, up to
`guardrail_max_retries` -- no custom loop code required.

We use it here for a narrow, low-level kind of self-healing: making sure
the scheduling agent's output is actually well-formed JSON matching our
schema (an LLM occasionally returns prose instead of calling the tool, or
paraphrases the tool result instead of passing it through). That's a
different failure class from "the slot was taken", which is a *business*
conflict handled one level up, in flow.py, by looping across agents.

Both are "self-correction" -- they just operate at different layers:
- guardrail: is this single task's output usable at all?
- flow router: given a usable output, was the booking itself successful,
  and if not, is it worth trying again?
"""

import json
from typing import Any

from crewai.tasks.task_output import TaskOutput

VALID_STATUSES = {"booked", "conflict", "error"}


def booking_output_guardrail(output: TaskOutput) -> tuple[bool, Any]:
    raw = output.raw.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return (
            False,
            "Your final answer must be ONLY the raw JSON object returned by "
            "the book_slot tool -- no prose, no markdown fences. Call the "
            "tool, then output exactly what it returned.",
        )

    status = data.get("status")
    if status not in VALID_STATUSES:
        return (
            False,
            f"The JSON you returned is missing a valid 'status' field "
            f"(must be one of {sorted(VALID_STATUSES)}). Re-check the "
            f"book_slot tool result and return it verbatim as JSON.",
        )

    return True, data
