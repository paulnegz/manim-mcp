"""SQLite-based render job tracking."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import aiosqlite

from manim_mcp.exceptions import RenderNotFoundError
from manim_mcp.models import RenderMetadata, RenderStatus

if TYPE_CHECKING:
    from manim_mcp.config import ManimMCPConfig

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS renders (
    render_id TEXT PRIMARY KEY,
    parent_render_id TEXT,
    status TEXT NOT NULL,
    scene_name TEXT,
    quality TEXT,
    format TEXT,
    original_prompt TEXT,
    source_code TEXT,
    edit_instructions TEXT,
    s3_url TEXT,
    s3_object_key TEXT,
    thumbnail_s3_key TEXT,
    file_size_bytes INTEGER,
    duration_seconds REAL,
    width INTEGER,
    height INTEGER,
    fps INTEGER,
    error_message TEXT,
    local_path TEXT,
    render_time_seconds REAL,
    code_hash TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
)
"""

_MIGRATION_ADD_THUMBNAIL_KEY = """
ALTER TABLE renders ADD COLUMN thumbnail_s3_key TEXT
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_renders_status ON renders (status)",
    "CREATE INDEX IF NOT EXISTS idx_renders_created ON renders (created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_renders_parent ON renders (parent_render_id)",
    "CREATE INDEX IF NOT EXISTS idx_renders_code_hash ON renders (code_hash)",
]


class RenderTracker:
    def __init__(self, config: ManimMCPConfig) -> None:
        self.db_path = config.tracker_db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute(_CREATE_TABLE)
        for idx_sql in _CREATE_INDEXES:
            await self._db.execute(idx_sql)
        await self._run_migrations()
        await self._db.commit()

    async def _run_migrations(self) -> None:
        """Run schema migrations for existing databases."""
        cursor = await self._db.execute("PRAGMA table_info(renders)")
        columns = {row[1] for row in await cursor.fetchall()}
        if "thumbnail_s3_key" not in columns:
            try:
                await self._db.execute(_MIGRATION_ADD_THUMBNAIL_KEY)
            except Exception:
                pass

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def create_render(
        self,
        render_id: str,
        *,
        parent_render_id: str | None = None,
        scene_name: str | None = None,
        quality: str | None = None,
        fmt: str | None = None,
        original_prompt: str | None = None,
        source_code: str | None = None,
        edit_instructions: str | None = None,
        code_hash: str | None = None,
    ) -> RenderMetadata:
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """INSERT INTO renders
               (render_id, parent_render_id, status, scene_name, quality, format,
                original_prompt, source_code, edit_instructions, code_hash, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (render_id, parent_render_id, RenderStatus.pending.value, scene_name,
             quality, fmt, original_prompt, source_code, edit_instructions, code_hash, now),
        )
        await self._db.commit()
        return RenderMetadata(
            render_id=render_id,
            parent_render_id=parent_render_id,
            status=RenderStatus.pending,
            scene_name=scene_name,
            quality=quality,
            format=fmt,
            original_prompt=original_prompt,
            source_code=source_code,
            edit_instructions=edit_instructions,
            code_hash=code_hash,
            created_at=now,
        )

    async def update_render(self, render_id: str, **kwargs) -> RenderMetadata:
        row = await self._get_row(render_id)
        if row is None:
            raise RenderNotFoundError(f"Render '{render_id}' not found")

        allowed_fields = {
            "status", "parent_render_id", "scene_name", "quality", "format",
            "original_prompt", "source_code", "edit_instructions",
            "s3_url", "s3_object_key", "thumbnail_s3_key", "file_size_bytes",
            "duration_seconds", "width", "height", "fps", "error_message",
            "local_path", "render_time_seconds", "code_hash", "completed_at",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields and v is not None}

        if "status" in updates and isinstance(updates["status"], RenderStatus):
            updates["status"] = updates["status"].value

        if not updates:
            return await self.get_render(render_id)

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [render_id]
        await self._db.execute(
            f"UPDATE renders SET {set_clause} WHERE render_id = ?",
            values,
        )
        await self._db.commit()
        return await self.get_render(render_id)

    async def get_render(self, render_id: str) -> RenderMetadata:
        row = await self._get_row(render_id)
        if row is None:
            raise RenderNotFoundError(f"Render '{render_id}' not found")
        return _row_to_metadata(row)

    async def list_renders(
        self,
        limit: int = 20,
        offset: int = 0,
        status: RenderStatus | None = None,
    ) -> list[RenderMetadata]:
        if status:
            cursor = await self._db.execute(
                "SELECT * FROM renders WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (status.value, limit, offset),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM renders ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        rows = await cursor.fetchall()
        return [_row_to_metadata(r) for r in rows]

    async def delete_render(self, render_id: str) -> None:
        row = await self._get_row(render_id)
        if row is None:
            raise RenderNotFoundError(f"Render '{render_id}' not found")
        await self._db.execute("DELETE FROM renders WHERE render_id = ?", (render_id,))
        await self._db.commit()

    async def count_active_renders(self) -> int:
        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM renders WHERE status IN (?, ?, ?)",
            (RenderStatus.pending.value, RenderStatus.generating.value, RenderStatus.rendering.value),
        )
        row = await cursor.fetchone()
        return row[0]

    async def find_by_code_hash(self, code_hash: str) -> RenderMetadata | None:
        cursor = await self._db.execute(
            "SELECT * FROM renders WHERE code_hash = ? AND status = ? ORDER BY created_at DESC LIMIT 1",
            (code_hash, RenderStatus.completed.value),
        )
        row = await cursor.fetchone()
        return _row_to_metadata(row) if row else None

    async def _get_row(self, render_id: str):
        cursor = await self._db.execute(
            "SELECT * FROM renders WHERE render_id = ?", (render_id,)
        )
        return await cursor.fetchone()


def _row_to_metadata(row) -> RenderMetadata:
    # Handle both old DBs (without thumbnail_s3_key) and new ones
    thumbnail_s3_key = None
    try:
        thumbnail_s3_key = row["thumbnail_s3_key"]
    except (IndexError, KeyError):
        pass

    return RenderMetadata(
        render_id=row["render_id"],
        parent_render_id=row["parent_render_id"],
        status=RenderStatus(row["status"]),
        scene_name=row["scene_name"],
        quality=row["quality"],
        format=row["format"],
        original_prompt=row["original_prompt"],
        source_code=row["source_code"],
        edit_instructions=row["edit_instructions"],
        s3_url=row["s3_url"],
        s3_object_key=row["s3_object_key"],
        thumbnail_s3_key=thumbnail_s3_key,
        file_size_bytes=row["file_size_bytes"],
        width=row["width"],
        height=row["height"],
        fps=row["fps"],
        duration_seconds=row["duration_seconds"],
        render_time_seconds=row["render_time_seconds"],
        error_message=row["error_message"],
        local_path=row["local_path"],
        code_hash=row["code_hash"],
        created_at=row["created_at"],
        completed_at=row["completed_at"],
        presigned_url=None,
        thumbnail_url=None,
    )
