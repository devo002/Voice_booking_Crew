# Self-Healing Booking Crew

A voice-first appointment-booking assistant built around one idea: when a
requested calendar slot is taken, the system doesn't just return an error --
it finds and proposes an alternative automatically, and retries, up to a
bounded number of attempts. You can speak the request, type it, or post it
as structured fields; all three paths run through the same self-correcting
core.

Started as a text-only demo against a mock in-memory calendar. Now also
supports real speech in/out, per-user accounts, and a persisted Postgres
backend (Supabase) -- the mock calendar still exists and is what the CLI
demo and the eval suite run against, so nothing here requires a Supabase
project just to explore the correction logic.

## Architecture

Three agents, each with one job:

- **Intake Agent** -- turns the caller's free-text (typed or spoken)
  request into structured fields (date, time, duration, title,
  constraints). Also resolves relative/bare day references ("Sunday",
  "next Tuesday", "tomorrow") to an absolute date, given today's actual
  date injected into the prompt -- an LLM has no built-in notion of "now".
- **Scheduling Agent** -- calls `book_slot` and reports exactly what
  happened. It does not decide how to recover from a conflict; that's not
  its job.
- **Conflict Resolution Agent** -- only runs after a conflict. Calls
  `find_alternative_slots` and picks the best alternative given the
  caller's stated constraints (e.g. "must stay in the afternoon"), never
  re-proposing a time already tried in this conversation.

These are wired together in `booking_crew/flow.py` as a CrewAI `Flow`.
The retry loop itself is a plain Python `while` inside one `@start`
method, not a chain of `@listen`/`@router` calls forming a cycle --
CrewAI's Flow engine runs each decorated method at most once per
execution, so a decorated method can't be re-entered to build a loop.
(This was empirically checked against crewai 1.15, not assumed --
`@listen`/`@router` model a one-shot DAG.) The `while` loop is the
correct way to get bounded, variable-length retries inside a `Flow`.

## Why CrewAI

The thing this project needed wasn't "call an LLM" -- it was three
narrow, single-responsibility roles that hand off to each other with
different tools and different failure-handling responsibilities, plus a
control layer that has to branch reliably on *what actually happened*.
A handful of CrewAI's pieces map onto that directly, rather than being
generic framework boilerplate:

- **`Task(output_pydantic=...)`** forces each agent's output into a typed
  Pydantic model (`ParsedRequest`, `ResolutionProposal`) instead of free
  text. The router in `flow.py` branches on `status == "conflict"`, not
  on parsing a sentence -- that's only possible because the schema is
  enforced at the framework level, not hoped for in a prompt.
- **`Task(guardrail=..., guardrail_max_retries=...)`** is a *native*
  self-correction primitive: if an agent's output doesn't match the
  expected shape, CrewAI automatically re-runs that task against the
  same agent with feedback about what was wrong. No custom "retry until
  valid JSON" loop needed for that layer (see Guardrails below for a real
  caveat this surfaced).
- **`@tool`-decorated functions** give agents a constrained action space
  (`book_slot`, `find_alternative_slots`) instead of open-ended
  generation -- the LLM can only do what a tool lets it do, and every
  tool returns structured JSON the agent (and our code) can reason over.
- **`Flow[BookingState]`** gives an explicit, typed state object across
  the whole multi-step interaction (`attempts`, `tried_times`,
  `last_result`, ...) rather than threading state through prompt context
  by hand, plus a hook (`@start`) that's inspectable in CrewAI's
  tracing/observability tooling.

None of this required agents to be "smart" about self-correction --
the framework's structure is what makes correction mechanical and
inspectable instead of another thing to prompt-engineer.

## Guardrails and self-correction

**Self-correction happens at two layers, deliberately kept separate:**

1. **Format-level, inside a single task** -- `booking_crew/guardrails.py`
   uses CrewAI's `Task.guardrail` mechanism described above. If the
   Scheduling Agent's output isn't valid JSON matching the expected
   shape, CrewAI re-runs that task against the same agent with feedback
   about what was wrong, up to `guardrail_max_retries` (2).
2. **Business-level, across agents** -- the `Flow`'s `while` loop in
   `flow.py`. If booking fails with a *correctable* reason (`conflict`),
   the Resolver is brought in to propose an alternative, and the
   Scheduler retries with the new time. Bounded by `max_attempts`
   (default 3) so it can't loop forever.

The distinction matters: "the agent said something malformed" and "the
underlying action failed for a real-world reason" are different failure
classes and usually deserve different repair strategies.

### Known issue: guardrail retries aren't side-effect-safe

The eval suite (see below) caught this, and a dedicated isolation test
confirmed it: **a guardrail retry can silently double-book a slot.**
`book_slot` has a side effect (it writes to the calendar) but isn't
idempotent. When the Scheduling Agent's first response doesn't match the
guardrail's expected format, CrewAI retries the *same task* -- which
calls `book_slot` again for the same date/time. The first (hidden) call
already booked it successfully; the retried call then sees that slot as
taken and reports a *self-inflicted* conflict, which the `Flow` reads as
a real one and kicks off the Resolver -- leaving the caller with two real
bookings (the original time and the "corrected" one) while only ever
being told about the second. Not yet fixed; the fix is to make `book_slot`
idempotent (e.g. check-then-book keyed so a retry with identical
arguments is a no-op) or move the guardrail's retry outside the
side-effecting call entirely.

### Why tool results are structured, not exceptions

`booking_crew/mock_calendar.py`, `supabase_calendar.py`, and `tools.py`
never raise on an ordinary business failure -- `book_slot` returns
`{"status": "conflict", "reason": "slot_taken", "detail": "..."}`. That's
what makes the failure *correctable* by an LLM: it has a machine-readable
signal to reason over, instead of an opaque error string it can only
apologize for. This is the single most important design choice in the
whole project -- if you add a new tool later, give it the same shape.

### CrewAI gotcha: guardrail return-type annotations

`Task(guardrail=...)` construction failed at runtime with:

```
Value error, If return type is annotated, it must be Tuple[bool, Any]
```

Two compounding issues, both in `booking_crew/guardrails.py`:

1. CrewAI's `Task` validator inspects the guardrail function's return-type
   annotation and only accepts a narrow allowlist for the second tuple
   element (`Any`, `str`, `TaskOutput`, or `str | TaskOutput`). The
   function was annotated `-> tuple[bool, str | dict]` -- accurate to
   what it actually returns, but `str | dict` isn't on that allowlist.
2. The file also had `from __future__ import annotations` (PEP 563) at
   the top, which makes Python store *all* annotations in that module as
   plain strings rather than live type objects. CrewAI's validator reads
   `inspect.signature(fn).return_annotation` directly without resolving
   PEP 563 strings back into types, so even a compliant annotation would
   have been unreadable to it as long as that import was present --
   confirmed by checking `get_origin()` on the annotation before and
   after removing it.

Fix: loosened the annotation to `tuple[bool, Any]` (on CrewAI's
allowlist) *and* removed `from __future__ import annotations`, so the
annotation resolves to a real `tuple[bool, typing.Any]` at runtime
instead of a string. Worth remembering before adding
`from __future__ import annotations` to any other file that defines a
`guardrail=` callable for a CrewAI `Task`.

## Voice: Whisper, not the browser's Speech API

The first voice implementation used the browser's built-in
`SpeechRecognition` (`webkitSpeechRecognition`) -- free, and it streams
transcription while you talk instead of waiting for a full recording.
It was dropped for a real reason, not a hypothetical one: **that API
isn't actually local** -- Chrome streams your audio to Google's own
speech servers to do the transcription. In a network environment where
that specific dependency was blocked, it failed with a bare, undebuggable
`"network"` error, even though the app's own server was completely
reachable.

The fix was server-side transcription via OpenAI's Whisper API
(`/api/book-voice` in `server.py`): the browser now just records raw
audio (`MediaRecorder`) and uploads it, and Whisper transcribes it. This
trades a small amount of latency (recording finishes before transcription
starts, instead of overlapping) and a per-request API cost for not
depending on an undocumented, third-party, browser-vendor-specific
network path. It also means voice booking only needs the same kind of
outbound HTTPS access the booking agents already use for their own LLM
calls -- `OPENAI_API_KEY` is required for this regardless of which
provider `MODEL` points at, since Anthropic has no transcription API.

Text-to-speech (`SpeechSynthesis`) stayed client-side -- there's no
equivalent reachability problem there, since it's speaking text the
browser already has, not sending anything out.

## Multi-user, persisted: Supabase

`booking_crew/calendar_backend.py` is the single indirection point
`tools.py` imports `CALENDAR` from, chosen by the `CALENDAR_BACKEND`
env var:

- **`mock`** (default) -- the original in-memory `MockCalendar`. No
  setup, no network, single implicit user. What `main.py`'s CLI demo and
  the eval suite use, so neither needs a Supabase project.
- **`supabase`** -- `SupabaseCalendar`, backed by a real `bookings`
  table in Postgres, scoped per-user. `server.py` requires a valid
  Supabase session (`Authorization: Bearer <access_token>`) on every
  booking endpoint when this is active; `static/index.html` handles
  sign-up/sign-in via `supabase-js` and attaches the token to every
  request.

Neither `tools.py` nor the agents know or care which backend is active --
both implementations expose the same four methods
(`is_available`/`book`/`find_nearby_available`/`cancel`, plus
`all_bookings`), which was the whole point of the original mock
calendar's docstring ("swap this module out for a real API client
later").

**Per-user scoping without an LLM-controlled parameter.** CrewAI's
`@tool`-decorated functions have a fixed signature the LLM fills in --
there's no safe way to make "whose calendar" an LLM tool argument
without risking a caller naming someone else's user id. Instead,
`supabase_calendar.current_user_id` is a `contextvars.ContextVar` that
`server.py` sets once per authenticated request (from the verified JWT,
never from the LLM), and every calendar method reads it implicitly.
CrewAI's flow runtime executes each step via
`asyncio.to_thread(ctx.run, method, ...)` with a captured
`contextvars.Context`, which is exactly the mechanism that makes
contextvars propagate correctly into that worker thread -- confirmed
empirically with a real two-user isolation test (see below), not
assumed.

**Row-Level Security is the actual enforcement boundary** on the
`bookings` table (`auth.uid() = user_id`); the app's own queries use the
service-role key, which bypasses RLS by design (the backend, not the end
user, is the trusted caller talking to Postgres), so the explicit
`current_user_id` filtering in `supabase_calendar.py` is what keeps one
user's queries scoped to their own rows day to day. RLS is the backstop
if that ever has a bug or a table is ever queried directly with the anon
key from the browser.

Verified end-to-end against a real project, not assumed: user A can book
a slot; user B booking the *identical* date/time succeeds independently
(no false cross-user conflict); user B cannot see or cancel user A's
booking (empty list, 404 respectively); an unauthenticated request is
rejected before touching any data.

## Eval suite

`tests/eval/` uses DeepEval to check two independent things a change to a
prompt or agent config could silently break:

- **Slot extraction accuracy** (`test_intake_accuracy.py`) -- does the
  Intake Agent correctly turn free text into the right structured
  fields? Custom `SlotFieldAccuracy` metric: exact match on
  date/time/duration, keyword containment on title/constraints (free-text
  phrasing legitimately varies there). Golden dates are computed relative
  to `date.today()` at collection time, not hardcoded, so cases testing
  "next Tuesday" etc. never go stale.
- **Task success rate** (`test_task_success_rate.py`) -- does the full
  `BookingFlow` reach the *correct* terminal state for a scenario? Not
  always "booked" -- correctly rejecting a past date is as much a success
  as booking a valid slot. Custom `TaskOutcomeMatch` metric checks both
  the final status and whether self-correction fired when it should have.

Both suites run the actual production code paths, not copies:
`build_intake_task`/`intake_inputs` (`flow.py`) are the same functions
`BookingFlow._intake` calls, factored out specifically so the eval can't
silently drift from what production runs.

Every case calls a real LLM, so the suite is marked and excluded by
default (`pytest.ini`: `addopts = -m "not eval"`) -- run it explicitly:

```bash
pip install -r requirements-dev.txt
pytest -m eval -v                              # everything
pytest -m eval -k conflict_triggers_self_correction -v   # one case
```

`tests/eval/conftest.py` force-sets `CALENDAR_BACKEND=mock` before
anything imports `booking_crew.tools`, so eval runs never depend on or
write to a real Supabase project regardless of what `.env` says.

On Windows, `deepeval test run` (its own CLI, richer report than plain
pytest) crashes *after* printing results unless UTF-8 mode is forced --
a `rich`/console encoding bug in that library, not this code:

```bash
$env:PYTHONUTF8=1; deepeval test run tests/eval -m eval   # PowerShell
```

## Running it

### CLI demo (no setup beyond an LLM key)

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY or OPENAI_API_KEY
python main.py
```

`main.py` pre-books 14:00 on the demo date so the first attempt collides
with it on purpose -- that's what triggers the correction loop instead of
booking cleanly on try #1. Watch the console: you'll see the Scheduler
hit a conflict, the Resolver get called in, and the Scheduler succeed on
a later attempt, with the log printed at the end.

### Web app (voice + form, mock calendar by default)

```bash
uvicorn server:app --reload
```

Open `http://127.0.0.1:8000/`. With `CALENDAR_BACKEND` unset (or `mock`),
there's no login -- it behaves like the original single-user demo, just
with voice added: speak or type a request, and repeat the same date/time
to see the self-correction path trigger on purpose.

### Web app with real accounts and persistence

Set in `.env` (see `.env.example` for the full list):
`CALENDAR_BACKEND=supabase`, `SUPABASE_URL`, `SUPABASE_ANON_KEY`,
`SUPABASE_SERVICE_ROLE_KEY`, plus `OPENAI_API_KEY` for Whisper. Requires
a Supabase project with the `bookings` table + RLS policy created (see
`booking_crew/supabase_calendar.py`'s docstring for the schema). Once
set, `/` requires sign-up/sign-in before showing the booking UI, and
`/bookings` shows a calendar view plus a cancellable list of your own
bookings, scoped by the account you're signed in as.

## Roadmap

Originally-planned items now done: voice (Whisper STT + browser TTS), a
real calendar backend (Supabase/Postgres, chosen over CrewAI's
`google_calendar` app integration for this iteration), and an eval
harness (DeepEval). Open items, roughly in priority order:

1. **Fix the guardrail/idempotency bug** above -- the most important
   remaining item; it's a real data-integrity issue, not polish.
2. **Reminders.** Deliberately dropped for now (see the multi-user
   section) in favor of shipping accounts/persistence first. Needs a
   scheduled check (Render Cron Job hitting an endpoint that queries
   Postgres for due bookings is enough for a first version -- no Celery/
   Redis needed at this scale) plus an email/SMS delivery provider,
   since nothing currently sends anything on its own.
3. **More failure types.** Only `conflict` and `date_in_past` are
   currently correctable/terminal-with-a-specific-message. Ambiguous
   dates from Intake could loop back to Intake with a clarifying question
   instead of attempting a bogus booking.
4. **Deploy.** Render is the intended host; needs the Supabase env vars
   set there and HTTPS (required for `MediaRecorder`/mic access outside
   `localhost` anyway).
5. **Rate limiting / abuse controls.** Once this isn't just local testing,
   every voice request costs a Whisper call plus 1-3 LLM calls -- worth
   limiting per authenticated user before it's exposed publicly.

## Files

```
booking_crew/
  mock_calendar.py     in-memory calendar, deterministic, seedable conflicts
  supabase_calendar.py real per-user calendar backed by Postgres (RLS-protected)
  calendar_backend.py  picks mock vs supabase via CALENDAR_BACKEND, so tools.py
                        never needs to know which one is active
  tools.py              CrewAI tools wrapping the active calendar (structured results)
  schemas.py            Pydantic models passed between flow steps
  agents.py              the three agents
  guardrails.py          native Task-level self-correction (output validation)
  flow.py                 orchestration: intake -> attempt -> resolve -> retry
main.py                CLI demo entry point (free-text request, mock calendar)
server.py              FastAPI backend: form + voice booking, auth, config
static/
  index.html           booking form + voice UI + sign-in/sign-up
  bookings.html         calendar view + cancellable list of your own bookings
tests/eval/
  datasets.py           golden test cases (dates computed relative to "today")
  metrics.py             custom DeepEval metrics (SlotFieldAccuracy, TaskOutcomeMatch)
  test_intake_accuracy.py     slot extraction accuracy eval
  test_task_success_rate.py   end-to-end outcome-correctness eval
  conftest.py             pins CALENDAR_BACKEND=mock, resets calendar between tests
requirements.txt       runtime dependencies
requirements-dev.txt   pytest + deepeval, eval-suite only
pytest.ini             excludes the (billable, LLM-calling) eval suite by default
```
