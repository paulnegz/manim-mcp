"""SSE progress broadcasting for animation renders.

Provides a pub/sub system where the pipeline publishes progress events
and HTTP clients subscribe via Server-Sent Events (SSE).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ── Friendly progress stages ──────────────────────────────────────────

PROGRESS_STAGES: Dict[int, tuple[str, str]] = {
    0:   ("\U0001f9e0", "Understanding your idea..."),
    5:   ("\U0001f9e0", "Understanding your idea..."),
    10:  ("\u270d\ufe0f", "Writing the animation code..."),
    25:  ("\U0001f50d", "Checking everything looks right..."),
    30:  ("\U0001f3ac", "Rendering your animation..."),
    55:  ("\U0001f3a8", "Adding the finishing touches..."),
    60:  ("\U0001f399\ufe0f", "Recording the narration..."),
    85:  ("\U0001f3b5", "Mixing audio and video..."),
    92:  ("\U0001f4f8", "Creating the thumbnail..."),
    95:  ("\u2728", "Almost there..."),
    100: ("\U0001f389", "Your animation is ready!"),
}


def _stage_for_pct(pct: int) -> tuple[str, str]:
    """Return the (emoji, message) pair for the given percentage.

    Finds the highest defined stage whose key is <= pct.
    """
    best_key = 0
    for key in sorted(PROGRESS_STAGES.keys()):
        if key <= pct:
            best_key = key
        else:
            break
    return PROGRESS_STAGES[best_key]


# ── Event types ───────────────────────────────────────────────────────

@dataclass
class ProgressEvent:
    """A single progress update."""
    event_type: str  # "progress", "thumbnail", "complete", "error"
    data: Dict[str, Any] = field(default_factory=dict)


# ── Render tracker ────────────────────────────────────────────────────

@dataclass
class _RenderState:
    """Internal state for a single active render."""
    queue: asyncio.Queue[ProgressEvent]
    created_at: float = field(default_factory=time.time)
    completed: bool = False
    last_event: Optional[ProgressEvent] = None


class ProgressBroadcaster:
    """Central hub for SSE progress events.

    Usage:
        broadcaster = ProgressBroadcaster()

        # Producer side (pipeline):
        broadcaster.create_render("abc123")
        broadcaster.send_progress("abc123", 30, "rendering")
        broadcaster.send_thumbnail("abc123", "https://...")
        broadcaster.send_complete("abc123", url="https://...")

        # Consumer side (SSE endpoint):
        async for event in broadcaster.subscribe("abc123"):
            yield format_sse(event)
    """

    def __init__(self, max_renders: int = 200, ttl_seconds: int = 600) -> None:
        self._renders: Dict[str, _RenderState] = {}
        self._max_renders = max_renders
        self._ttl_seconds = ttl_seconds

    # ── Producer API ──────────────────────────────────────────────

    def create_render(self, render_id: str) -> None:
        """Register a new render for progress tracking."""
        self._evict_stale()
        self._renders[render_id] = _RenderState(queue=asyncio.Queue())
        logger.debug("Progress tracking started for render %s", render_id)

    def send_progress(self, render_id: str, pct: int, stage: str = "") -> None:
        """Push a progress event to all subscribers."""
        state = self._renders.get(render_id)
        if not state or state.completed:
            return

        emoji, message = _stage_for_pct(pct)
        event = ProgressEvent(
            event_type="progress",
            data={
                "pct": pct,
                "emoji": emoji,
                "message": message,
                "stage": stage,
                "render_id": render_id,
            },
        )
        state.last_event = event
        state.queue.put_nowait(event)

    def send_thumbnail(self, render_id: str, url: str) -> None:
        """Push an early thumbnail event."""
        state = self._renders.get(render_id)
        if not state or state.completed:
            return

        event = ProgressEvent(
            event_type="thumbnail",
            data={"url": url, "render_id": render_id},
        )
        state.last_event = event
        state.queue.put_nowait(event)

    def send_complete(
        self,
        render_id: str,
        url: str | None = None,
        thumbnail_url: str | None = None,
        result: Dict[str, Any] | None = None,
    ) -> None:
        """Push the final completion event and mark the render as done."""
        state = self._renders.get(render_id)
        if not state:
            return

        emoji, message = PROGRESS_STAGES[100]
        data: Dict[str, Any] = {
            "pct": 100,
            "emoji": emoji,
            "message": message,
            "render_id": render_id,
        }
        if url:
            data["url"] = url
        if thumbnail_url:
            data["thumbnail_url"] = thumbnail_url
        if result:
            data["result"] = result

        event = ProgressEvent(event_type="complete", data=data)
        state.last_event = event
        state.completed = True
        state.queue.put_nowait(event)

    def send_error(self, render_id: str, error: str) -> None:
        """Push an error event and mark the render as done."""
        state = self._renders.get(render_id)
        if not state:
            return

        event = ProgressEvent(
            event_type="error",
            data={
                "render_id": render_id,
                "error": error,
            },
        )
        state.last_event = event
        state.completed = True
        state.queue.put_nowait(event)

    # ── Consumer API ──────────────────────────────────────────────

    def has_render(self, render_id: str) -> bool:
        """Check if a render is being tracked."""
        return render_id in self._renders

    def is_completed(self, render_id: str) -> bool:
        """Check if a render has finished (complete or error)."""
        state = self._renders.get(render_id)
        return state.completed if state else False

    async def subscribe(self, render_id: str):
        """Async generator that yields ProgressEvents for a render.

        Terminates when a 'complete' or 'error' event is received,
        or after a timeout (to avoid leaked connections).
        """
        state = self._renders.get(render_id)
        if not state:
            yield ProgressEvent(
                event_type="error",
                data={"render_id": render_id, "error": "Render not found"},
            )
            return

        timeout = self._ttl_seconds
        try:
            while True:
                try:
                    event = await asyncio.wait_for(state.queue.get(), timeout=timeout)
                    yield event
                    if event.event_type in ("complete", "error"):
                        return
                except asyncio.TimeoutError:
                    # Connection timed out waiting for events
                    yield ProgressEvent(
                        event_type="error",
                        data={"render_id": render_id, "error": "Timed out waiting for progress"},
                    )
                    return
        except asyncio.CancelledError:
            logger.debug("SSE subscription cancelled for render %s", render_id)
            return

    # ── Housekeeping ──────────────────────────────────────────────

    def remove_render(self, render_id: str) -> None:
        """Remove a render from tracking."""
        self._renders.pop(render_id, None)

    def _evict_stale(self) -> None:
        """Remove completed or expired renders to stay under limits."""
        now = time.time()
        stale = [
            rid
            for rid, state in self._renders.items()
            if state.completed or (now - state.created_at > self._ttl_seconds)
        ]
        for rid in stale:
            del self._renders[rid]

        # Hard limit: drop oldest if still over max
        if len(self._renders) >= self._max_renders:
            oldest = sorted(self._renders, key=lambda r: self._renders[r].created_at)
            for rid in oldest[: len(self._renders) - self._max_renders + 1]:
                del self._renders[rid]


# ── Singleton ─────────────────────────────────────────────────────────

_broadcaster: ProgressBroadcaster | None = None


def get_broadcaster() -> ProgressBroadcaster:
    """Get or create the global ProgressBroadcaster singleton."""
    global _broadcaster
    if _broadcaster is None:
        _broadcaster = ProgressBroadcaster()
    return _broadcaster


# ── SSE formatting helper ─────────────────────────────────────────────

def format_sse(event: ProgressEvent) -> str:
    """Format a ProgressEvent as an SSE text chunk.

    Example output:
        event: progress
        data: {"pct": 30, "emoji": "...", "message": "..."}

    """
    data_str = json.dumps(event.data, ensure_ascii=False)
    return f"event: {event.event_type}\ndata: {data_str}\n\n"
