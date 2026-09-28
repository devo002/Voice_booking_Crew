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
- **`Task(max_retries=...)`** is a *native*
  self-correction primitive: if an agent's output doesn't match the
  expected shape, CrewAI automatically re-runs that task against the
  same agent with feedback about what was wrong. No custom "retry until
  valid JSON" loop needed for that layer .
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
   uses CrewAI's  mechanism described above. If the
   Scheduling Agent's output isn't valid JSON matching the expected
   shape, CrewAI re-runs that task against the same agent with feedback
   about what was wrong, up to `_max_retries` (2).
2. **Business-level, across agents** -- the `Flow`'s `while` loop in
   `flow.py`. If booking fails with a *correctable* reason (`conflict`),
   the Resolver is brought in to propose an alternative, and the
   Scheduler retries with the new time. Bounded by `max_attempts`
   (default 3) so it can't loop forever.

The distinction matters: "the agent said something malformed" and "the
underlying action failed for a real-world reason" are different failure
classes and usually deserve different repair strategies.

### Known issue: retries aren't side-effect-safe

The eval suite (see below) caught this, and a dedicated isolation test
confirmed it: **a g retry can silently double-book a slot.**
`book_slot` has a side effect (it writes to the calendar) but isn't
idempotent. When the Scheduling Agent's first response doesn't match the expected format, CrewAI retries the *same task* -- which
calls `book_slot` again for the same date/time. The first (hidden) call
already booked it successfully; the retried call then sees that slot as
taken and reports a *self-inflicted* conflict, which the `Flow` reads as
a real one and kicks off the Resolver -- leaving the caller with two real
bookings (the original time and the "corrected" one) while only ever
being told about the second. Not yet fixed; the fix is to make `book_slot`
idempotent (e.g. check-then-book keyed so a retry with identical
arguments is a no-op) or move the retry outside the
side-effecting call entirely.

### Missing required fields: refuse, don't invent

`ParsedRequest.time` (`schemas.py`) used to be a required string with no
default -- the Intake Agent had to put *something* there even when the
caller never stated a time, despite its own backstory promising it would
"never invent a date/time that wasn't stated." Confirmed empirically, not
assumed: asking it to parse "book a dentist appointment next Tuesday"
(no time at all) came back with `time: "09:00"`, invented from nothing.
Worse, non-deterministically it could instead leave the field
null/missing, which -- since there was no default -- raised an unhandled
Pydantic `ValidationError` deep inside the flow, surfacing to the browser
as a raw, non-JSON `"Internal Server Error"` (a `fetch().json()` parse
failure, not a useful error message).

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

1. **Fix the idempotency bug** above -- the most important
   remaining item; it's a real data-integrity issue, not polish.
2. **Reminders.** Deliberately dropped for now in favor of shipping
   accounts/persistence first. Needs a
   scheduled check (Render Cron Job hitting an endpoint that queries
   Postgres for due bookings is enough for a first version -- no Celery/
   Redis needed at this scale) plus an email/SMS delivery provider,
   since nothing currently sends anything on its own.
3. **More failure types.** `conflict`, `date_in_past`, and a missing
   time are now handled with specific, correct responses. `date` has the
   same "could get invented" gap `time` had and
   is the next candidate -- loop back to Intake with a clarifying
   question instead of attempting a bogus booking.
4. **Deploy.** Render is the intended host; needs the Supabase env vars
   set there and HTTPS (required for `MediaRecorder`/mic access outside
   `localhost` anyway).
5. **Rate limiting / abuse controls.** Once this isn't just local testing,
   every voice request costs a Whisper call plus 1-3 LLM calls -- worth
   limiting per authenticated user before it's exposed publicly.
