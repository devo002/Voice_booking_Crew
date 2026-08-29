"""
The single point tools.py imports CALENDAR from, so the rest of the app
(agents, flow, tools) never needs to know or care which backend is live:

  - "mock" (default): in-memory MockCalendar. Zero setup, no network, no
    Supabase project needed -- what main.py's CLI demo and the eval suite
    use, so they keep working offline exactly as before.
  - "supabase": real, persisted, multi-user SupabaseCalendar. What
    server.py uses once SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY are set.

Selected via the CALENDAR_BACKEND env var, read once at import time --
set it before anything imports booking_crew.tools (server.py does this
via .env; tests/eval/conftest.py pins it to "mock" explicitly so eval
runs never depend on or touch a real Supabase project).
"""

from __future__ import annotations

import os

if os.getenv("CALENDAR_BACKEND") == "supabase":
    from booking_crew.supabase_calendar import SupabaseCalendar

    CALENDAR = SupabaseCalendar()
else:
    from booking_crew.mock_calendar import CALENDAR
