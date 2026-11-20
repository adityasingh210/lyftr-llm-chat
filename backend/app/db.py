"""SQLite persistence for chat sessions.

Uses the stdlib sqlite3 driver. Each operation opens a short-lived connection and runs in a
worker thread (via asyncio.to_thread) so the event loop is never blocked. WAL mode lets reads
proceed while a write is in progress.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    metadata    TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id, id);

CREATE TABLE IF NOT EXISTS sessions (
    session_id     TEXT PRIMARY KEY,
    system_prompt  TEXT,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
"""


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class Store:
    def __init__(self, path: str) -> None:
        self.path = path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ---- lifecycle -------------------------------------------------------------------------
    def init(self) -> None:
        if self.path != ":memory:":
            directory = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(directory, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
            conn.commit()

    def ping(self) -> bool:
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1 FROM messages LIMIT 1").fetchall()
            return True
        except sqlite3.Error:
            return False

    # ---- sync implementations --------------------------------------------------------------
    def _get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT session_id, system_prompt FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def _get_history(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if limit:
                rows = conn.execute(
                    "SELECT * FROM (SELECT id, role, content, metadata, created_at FROM messages "
                    "WHERE session_id = ? ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
                    (session_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, role, content, metadata, created_at FROM messages WHERE session_id = ? ORDER BY id ASC",
                    (session_id,),
                ).fetchall()
        history = []
        for row in rows:
            item: dict[str, Any] = {
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
            }
            if row["metadata"]:
                item["metadata"] = json.loads(row["metadata"])
            history.append(item)
        return history

    def _session_exists(self, session_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM messages WHERE session_id = ? LIMIT 1", (session_id,)).fetchone()
        return row is not None

    def _append_exchange(
        self,
        session_id: str,
        user_message: str,
        reply: str,
        metadata: dict[str, Any],
        system_prompt: str | None,
    ) -> None:
        """Persist one user/assistant turn atomically (both rows or neither)."""
        user_ts = utc_now_iso()
        with self._connect() as conn, conn:  # transaction
            conn.execute(
                "INSERT INTO sessions (session_id, system_prompt, created_at, updated_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET "
                "system_prompt = excluded.system_prompt, updated_at = excluded.updated_at",
                (session_id, system_prompt, user_ts, user_ts),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, metadata, created_at) VALUES (?, 'user', ?, NULL, ?)",
                (session_id, user_message, user_ts),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, metadata, created_at) "
                "VALUES (?, 'assistant', ?, ?, ?)",
                (session_id, reply, json.dumps(metadata), utc_now_iso()),
            )

    # ---- async facade ----------------------------------------------------------------------
    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_session, session_id)

    async def get_history(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._get_history, session_id, limit)

    async def session_exists(self, session_id: str) -> bool:
        return await asyncio.to_thread(self._session_exists, session_id)

    async def append_exchange(self, *args: Any, **kwargs: Any) -> None:
        await asyncio.to_thread(self._append_exchange, *args, **kwargs)

    async def aping(self) -> bool:
        return await asyncio.to_thread(self.ping)
