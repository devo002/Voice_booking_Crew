"""
Minimal HTTP front door for BookingFlow. A user fills a booking form (or
speaks) in the browser; the backend runs the same self-correcting flow the
CLI demo uses (Scheduler -> conflict -> Resolver -> retry) and returns
either a clean success or a "your time changed, here's why" result.

This reuses the `notify` callback already built into BookingFlow for
real-time narration (see flow.py) -- here it's collected into a list and
returned in the response instead of printed to a terminal. Same seam a
future voice adapter would use for TTS.

Voice booking (/api/book-voice) transcribes the caller's uploaded audio
via OpenAI's Whisper API server-side -- deliberately not the browser's
built-in SpeechRecognition, which depends on Google's speech service being
reachable and fails hard (a bare "network" error) when it isn't. Whisper
only needs the same kind of outbound HTTPS access the booking agents
already use for their own LLM calls.

Auth: when CALENDAR_BACKEND=supabase (see calendar_backend.py), every
booking endpoint requires a valid Supabase session (Authorization: Bearer
<access_token>, obtained client-side via supabase-js -- see static/
index.html). The verified user id is bound to
supabase_calendar.current_user_id for the duration of the request; with
CALENDAR_BACKEND unset/"mock", auth is skipped entirely (the in-memory
calendar has no concept of users) so local/CLI development needs no
Supabase project at all.

Run:
    uvicorn server:app --reload
Then open http://127.0.0.1:8000/

Note: with the mock backend, calendar state is in-memory only -- every
booking is lost on restart, including a --reload auto-restart triggered
by editing this file. The Supabase backend persists normally.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from openai import AsyncOpenAI, OpenAIError  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from supabase import Client as SupabaseClient  # noqa: E402
from supabase import create_client  # noqa: E402

from booking_crew.calendar_backend import CALENDAR  # noqa: E402
from booking_crew.flow import BookingFlow  # noqa: E402
from booking_crew.supabase_calendar import current_user_id  # noqa: E402

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Booking Crew API")

AUTH_REQUIRED = os.getenv("CALENDAR_BACKEND") == "supabase"

_transcriber: AsyncOpenAI | None = None
_auth_client: SupabaseClient | None = None


def _get_transcriber() -> AsyncOpenAI:
    # Built lazily, on first voice request, rather than at import time --
    # AsyncOpenAI() raises immediately if OPENAI_API_KEY isn't set, which
    # would otherwise crash the whole server (including the plain-form
    # /api/book path, which doesn't need OpenAI at all) just for missing a
    # key that's only required for voice.
    global _transcriber
    if _transcriber is None:
        try:
            _transcriber = AsyncOpenAI()
        except OpenAIError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Voice booking needs OPENAI_API_KEY set in .env: {e}",
            ) from e
    return _transcriber


def _get_auth_client() -> SupabaseClient:
    # Anon-key client used only to verify a caller's JWT (auth.get_user) --
    # distinct from supabase_calendar.py's service-role client, which
    # bypasses RLS to read/write bookings once the caller is verified.
    global _auth_client
    if _auth_client is None:
        _auth_client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_ANON_KEY"])
    return _auth_client


async def get_current_user(authorization: str | None = Header(default=None)) -> str | None:
    if not AUTH_REQUIRED:
        return None
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header.")
    token = authorization.removeprefix("Bearer ")
    try:
        response = _get_auth_client().auth.get_user(token)
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid or expired session.") from e
    if not response or not response.user:
        raise HTTPException(status_code=401, detail="Invalid or expired session.")
    return response.user.id


@contextmanager
def _bound_user(user_id: str | None):
    """Binds the verified caller to supabase_calendar.current_user_id for
    the duration of one request. A no-op under the mock backend (user_id
    is always None there, since get_current_user skips verification)."""
    if user_id is None:
        yield
        return
    token = current_user_id.set(user_id)
    try:
        yield
    finally:
        current_user_id.reset(token)


class BookingRequest(BaseModel):
    date: str = Field(description="YYYY-MM-DD")
    time: str = Field(description="HH:MM, 24-hour")
    duration_minutes: int = 30
    title: str
    constraints: str = ""


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/bookings")
def bookings_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "bookings.html")


@app.get("/api/config")
def get_config() -> dict:
    """Public config for the frontend to initialize supabase-js. The anon
    key is designed to be exposed in browser code -- RLS is what actually
    protects the data, not keeping this key secret."""
    if not AUTH_REQUIRED:
        return {"auth_enabled": False}
    return {
        "auth_enabled": True,
        "supabase_url": os.environ["SUPABASE_URL"],
        "supabase_anon_key": os.environ["SUPABASE_ANON_KEY"],
    }


@app.get("/api/bookings")
def list_bookings(user_id: str | None = Depends(get_current_user)) -> dict:
    with _bound_user(user_id):
        return {"bookings": CALENDAR.all_bookings()}


@app.delete("/api/bookings")
def cancel_booking(date: str, time: str, user_id: str | None = Depends(get_current_user)) -> dict:
    with _bound_user(user_id):
        cancelled = CALENDAR.cancel(date, time)
    if not cancelled:
        raise HTTPException(status_code=404, detail=f"No booking at {date} {time}.")
    return {"cancelled": True, "date": date, "time": time}


async def _run_flow(inputs: dict) -> dict:
    messages: list[str] = []
    flow = BookingFlow(notify=messages.append)
    await flow.kickoff_async(inputs=inputs)
    return {
        "status": flow.state.final_status,
        "date": flow.state.date,
        "time": flow.state.time,
        "title": flow.state.title,
        "attempts": flow.state.attempts,
        "corrected": flow.state.attempts > 1,
        "messages": messages,
    }


@app.post("/api/book")
async def book(request: BookingRequest, user_id: str | None = Depends(get_current_user)) -> dict:
    with _bound_user(user_id):
        return await _run_flow(
            {
                "date": request.date,
                "time": request.time,
                "duration_minutes": request.duration_minutes,
                "title": request.title,
                "constraints": request.constraints,
            }
        )


@app.post("/api/book-voice")
async def book_voice(
    audio: UploadFile = File(...), user_id: str | None = Depends(get_current_user)
) -> dict:
    transcriber = _get_transcriber()
    transcription = await transcriber.audio.transcriptions.create(
        model="whisper-1",
        file=(audio.filename or "speech.webm", await audio.read(), audio.content_type),
    )
    heard = transcription.text.strip()

    if not heard:
        return {
            "status": "failed",
            "date": "",
            "time": "",
            "title": "",
            "attempts": 0,
            "corrected": False,
            "messages": ["I didn't catch that -- could you try again?"],
            "heard": "",
        }

    with _bound_user(user_id):
        result = await _run_flow({"raw_request": heard})
    result["heard"] = heard
    return result
