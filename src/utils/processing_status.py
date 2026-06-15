"""
Processing-status tracking for the async "upload → process → index → ready"
pipeline.

``/v1/data/process/{project_id}`` schedules a background task that walks a
project's uploaded files through extraction, chunking, embedding and
vector-DB indexing. The frontend polls ``/v1/data/status/{project_id}``
until the status reaches ``READY`` (or ``FAILED``) before it lets the user
chat.

This is a lightweight in-process store (a single ``asyncio``-safe dict).
It is intentionally simple: the background task and the polling endpoint
run inside the same FastAPI process/event loop, so no external broker is
required. For multi-replica deployments this would need to move to a
shared backend (e.g. Redis, for which connection settings already exist in
``helpers.config``) — see the note in the project README / hand-off notes.
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Dict, Optional


class ProcessingStatus(str, Enum):
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    EXTRACTING = "EXTRACTING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    INDEXING = "INDEXING"
    READY = "READY"
    FAILED = "FAILED"


# States that mean "the frontend may open the chat interface".
TERMINAL_STATES = {ProcessingStatus.READY, ProcessingStatus.FAILED}


class ProcessingStatusStore:
    """In-memory, asyncio-safe store of per-project processing status."""

    def __init__(self) -> None:
        self._store: Dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def set_status(
        self,
        project_id: str,
        status: ProcessingStatus,
        detail: str = "",
        **progress,
    ) -> None:
        async with self._lock:
            entry = self._store.setdefault(str(project_id), {})
            entry["status"] = status.value if isinstance(status, ProcessingStatus) else status
            entry["detail"] = detail
            entry["updated_at"] = time.time()
            if progress:
                entry.setdefault("progress", {}).update(progress)

    async def get_status(self, project_id: str) -> dict:
        async with self._lock:
            entry = self._store.get(str(project_id))
            if entry is None:
                return {
                    "status": None,
                    "detail": "",
                    "updated_at": None,
                    "progress": {},
                }
            return dict(entry)

    async def reset(self, project_id: str) -> None:
        async with self._lock:
            self._store.pop(str(project_id), None)

    def is_terminal(self, status: Optional[str]) -> bool:
        return status in {s.value for s in TERMINAL_STATES}


# Module-level singleton — shared across the FastAPI app's lifetime.
processing_status_store = ProcessingStatusStore()
