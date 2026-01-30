"""Terminal formatting: ANSI colors, JSON mode, and a spinner."""

from __future__ import annotations

import json
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any


# ── ANSI helpers ──────────────────────────────────────────────────────

_IS_TTY = sys.stdout.isatty()


def _sgr(code: str) -> str:
    return f"\033[{code}m" if _IS_TTY else ""


BOLD = _sgr("1")
DIM = _sgr("2")
RED = _sgr("31")
GREEN = _sgr("32")
YELLOW = _sgr("33")
BLUE = _sgr("34")
MAGENTA = _sgr("35")
CYAN = _sgr("36")
RESET = _sgr("0")


# ── Spinner ───────────────────────────────────────────────────────────

@contextmanager
def spinner(message: str):
    """Show a simple spinner on stderr while work is happening."""
    if not sys.stderr.isatty():
        sys.stderr.write(f"{message}...\n")
        sys.stderr.flush()
        yield
        return

    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    stop = threading.Event()

    def _spin():
        i = 0
        while not stop.is_set():
            sys.stderr.write(f"\r{CYAN}{frames[i % len(frames)]}{RESET} {message}")
            sys.stderr.flush()
            i += 1
            stop.wait(0.08)
        sys.stderr.write(f"\r{' ' * (len(message) + 4)}\r")
        sys.stderr.flush()

    t = threading.Thread(target=_spin, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()


# ── Printer ───────────────────────────────────────────────────────────

class Printer:
    """Unified output: human-friendly ANSI or machine-readable JSON."""

    def __init__(self, json_mode: bool = False) -> None:
        self.json_mode = json_mode

    # ── Low-level ─────────────────────────────────────────────────

    def _print_json(self, data: dict) -> None:
        print(json.dumps(data, default=str))

    def _write(self, text: str) -> None:
        print(text)

    # ── Status messages ───────────────────────────────────────────

    def success(self, message: str) -> None:
        if self.json_mode:
            self._print_json({"status": "success", "message": message})
        else:
            self._write(f"{GREEN}{BOLD}✓{RESET} {message}")

    def error(self, message: str) -> None:
        if self.json_mode:
            self._print_json({"status": "error", "message": message})
        else:
            self._write(f"{RED}{BOLD}✗{RESET} {RED}{message}{RESET}")

    def info(self, message: str) -> None:
        if self.json_mode:
            return  # info is suppressed in JSON mode
        self._write(f"{DIM}{message}{RESET}")

    def warn(self, message: str) -> None:
        if self.json_mode:
            return
        self._write(f"{YELLOW}⚠ {message}{RESET}")

    # ── Animation result ──────────────────────────────────────────

    def animation_result(self, result: dict) -> None:
        if self.json_mode:
            self._print_json(result)
            return

        self._write("")
        self._write(f"{GREEN}{BOLD}Animation ready{RESET}")
        self._write(f"  Render ID : {CYAN}{result['render_id']}{RESET}")
        if result.get("url"):
            self._write(f"  URL       : {BOLD}{result['url']}{RESET}")
        if result.get("format"):
            self._write(f"  Format    : {result['format']}")
        if result.get("quality"):
            self._write(f"  Quality   : {result['quality']}")
        if result.get("resolution"):
            self._write(f"  Resolution: {result['resolution']}")
        if result.get("file_size_bytes"):
            size_mb = result["file_size_bytes"] / (1024 * 1024)
            self._write(f"  Size      : {size_mb:.1f} MB")
        if result.get("render_time_seconds"):
            self._write(f"  Render    : {result['render_time_seconds']:.1f}s")
        self._write("")

    # ── Render list ───────────────────────────────────────────────

    def render_list(self, renders: list[dict], count: int) -> None:
        if self.json_mode:
            self._print_json({"renders": renders, "count": count})
            return

        if not renders:
            self._write(f"{DIM}No renders found.{RESET}")
            return

        self._write(f"\n{BOLD}Renders ({count}):{RESET}\n")
        for r in renders:
            status = r.get("status", "?")
            color = {
                "completed": GREEN,
                "failed": RED,
                "rendering": YELLOW,
                "generating": YELLOW,
                "uploading": BLUE,
                "pending": DIM,
            }.get(status, "")
            prompt = r.get("original_prompt", "") or ""
            if len(prompt) > 60:
                prompt = prompt[:57] + "..."
            self._write(
                f"  {CYAN}{r['render_id']}{RESET}  "
                f"{color}{status:12}{RESET}  "
                f"{prompt}"
            )
        self._write("")

    # ── Render detail ─────────────────────────────────────────────

    def render_detail(self, detail: dict) -> None:
        if self.json_mode:
            self._print_json(detail)
            return

        self._write("")
        self._write(f"{BOLD}Render {CYAN}{detail['render_id']}{RESET}")
        fields = [
            ("Status", "status"),
            ("Scene", "scene_name"),
            ("Quality", "quality"),
            ("Format", "format"),
            ("Prompt", "original_prompt"),
            ("URL", "presigned_url"),
            ("S3 URL", "s3_url"),
            ("Size", "file_size_bytes"),
            ("Render time", "render_time_seconds"),
            ("Created", "created_at"),
            ("Completed", "completed_at"),
            ("Error", "error_message"),
        ]
        for label, key in fields:
            val = detail.get(key)
            if val is None:
                continue
            if key == "file_size_bytes":
                val = f"{val / (1024 * 1024):.1f} MB"
            elif key == "render_time_seconds":
                val = f"{val:.1f}s"
            elif key == "status":
                color = {
                    "completed": GREEN, "failed": RED,
                    "rendering": YELLOW, "generating": YELLOW,
                }.get(val, "")
                val = f"{color}{val}{RESET}"
            self._write(f"  {label:12}: {val}")
        self._write("")

    # ── Agent output ──────────────────────────────────────────────

    def agent_text(self, text: str) -> None:
        if self.json_mode:
            self._print_json({"type": "agent_text", "text": text})
        else:
            self._write(text)

    def agent_tool_call(self, name: str, args: dict[str, Any]) -> None:
        if self.json_mode:
            self._print_json({"type": "tool_call", "name": name, "args": args})
        else:
            args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
            self._write(f"\n{MAGENTA}▶ {name}{RESET}({DIM}{args_str}{RESET})")

    def agent_tool_result(self, name: str, result: dict[str, Any]) -> None:
        if self.json_mode:
            self._print_json({"type": "tool_result", "name": name, "result": result})
        else:
            if result.get("error"):
                self._write(f"  {RED}✗ {result.get('message', 'failed')}{RESET}")
            elif name == "generate_animation" or name == "edit_animation":
                rid = result.get("render_id", "?")
                url = result.get("url", "")
                self._write(f"  {GREEN}✓{RESET} render_id={CYAN}{rid}{RESET}")
                if url:
                    self._write(f"    {url}")
            elif name == "list_renders":
                count = result.get("count", 0)
                self._write(f"  {GREEN}✓{RESET} {count} render(s)")
            elif name == "get_render":
                self._write(f"  {GREEN}✓{RESET} status={result.get('status', '?')}")
            elif name == "delete_render":
                self._write(f"  {GREEN}✓{RESET} deleted {result.get('render_id', '?')}")
            else:
                self._write(f"  {GREEN}✓{RESET} done")
