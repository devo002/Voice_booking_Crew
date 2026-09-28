from __future__ import annotations

import os

# Must happen before anything imports booking_crew.tools/calendar_backend,
# so unit runs never depend on (or write to) a real Supabase project.
os.environ["CALENDAR_BACKEND"] = "mock"
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

# CREWAI_TRACING_ENABLED can only force tracing on; a machine that once
# accepted CrewAI's tracing prompt stays opted in via a stored consent file,
# which makes every Flow.kickoff() upload a trace (~2s of HTTP each). That
# file is looked up under this name, so pointing at an unused one gives every
# run a clean, opted-out state regardless of the developer's machine.
os.environ["CREWAI_STORAGE_DIR"] = "voice_booking_crew_unit_tests"
