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
import re
from typing import Any

from crewai.tasks.task_output import TaskOutput

VALID_STATUSES = {"booked", "conflict", "error"}

# LLMs habitually wrap JSON in ```json ... ``` fences even when told not to.
# book_slot has already run (and mutated the calendar) by the time this
# guardrail sees the output, so treating a fenced-but-otherwise-correct
# answer as a hard failure is costly, not just cosmetic: CrewAI's retry
# re-runs the whole task, which calls the non-idempotent book_slot tool
# again and self-conflicts with the booking the first call already made.
# Stripping the fence here avoids that retry in the common case instead of
# only asking the agent more firmly not to add one.
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def booking_output_guardrail(output: TaskOutput) -> tuple[bool, Any]:
    raw = output.raw.strip()
    fenced = _FENCE_RE.match(raw)
    if fenced:
        raw = fenced.group(1).strip()
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
