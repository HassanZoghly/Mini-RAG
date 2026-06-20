"""
MemoryStore — Phase 5: preference and progress tracking (item 6).

New capabilities:
- ``store_preference``   — persist a student's teaching-mode / style choice.
- ``get_preference``     — retrieve the most recent preference record for a session.
- ``update_progress``    — append newly discussed topics to the session's progress record.
- ``get_progress``       — retrieve the current progress record for a session.

Existing ``store_memory`` / ``retrieve_memories`` are unchanged so all
existing callers (routes/nlp.py, MemoryAgent) keep working.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from sqlalchemy import select

from controllers.NLPController import NLPController
from models.db_schemes.minirag.schemes.memory import MemoryModel
from stores.llm.LLMEnums import DocumentTypeEnum

from .MemorySchema import MemoryRecord, create_memory_record

logger = logging.getLogger(__name__)

_CONSOLIDATION_THRESHOLD: int = 10
_CONSOLIDATION_BATCH: int = 5

# memory_type constants
_TYPE_SHORT_TERM  = "short_term"
_TYPE_PREFERENCE  = "preference"
_TYPE_PROGRESS    = "progress"


class MemoryStore:

    def __init__(self, nlp_controller: NLPController, async_session_maker) -> None:
        self._nlp = nlp_controller
        self._async_session_maker = async_session_maker

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> List[float]:
        raw = self._nlp.embedding_client.embed_text(
            text=text,
            document_type=DocumentTypeEnum.DOCUMENT.value,
        )
        return self._nlp._flatten_vector(raw)

    def _embed_query(self, text: str) -> List[float]:
        raw = self._nlp.embedding_client.embed_text(
            text=text,
            document_type=DocumentTypeEnum.QUERY.value,
        )
        return self._nlp._flatten_vector(raw)

    # ------------------------------------------------------------------
    # Core store / retrieve (unchanged)
    # ------------------------------------------------------------------

    async def store_memory(self, memory: MemoryRecord) -> bool:
        try:
            vector = self._embed(memory["content"])

            async with self._async_session_maker() as session:
                db_memory = MemoryModel(
                    memory_id=memory["memory_id"],
                    session_id=memory["session_id"],
                    memory_type=memory["memory_type"],
                    content=memory["content"],
                    summary=memory["summary"],
                    timestamp=memory["timestamp"],
                    importance_score=memory["importance_score"],
                    embedding=vector,
                )
                session.add(db_memory)
                await session.commit()

            logger.info(
                "MemoryStore: stored memory_id=%s type=%s session=%s",
                memory["memory_id"], memory["memory_type"], memory["session_id"],
            )
            return True

        except Exception as exc:
            logger.error("MemoryStore.store_memory failed: %s", exc)
            return False

    async def retrieve_memories(
        self,
        query: str,
        session_id: str,
        k: int = 5,
        memory_type: Optional[str] = None,
    ) -> List[MemoryRecord]:
        """
        Retrieve the *k* most semantically relevant memories for *query*
        within *session_id*.  Optionally filter by *memory_type*.
        """
        try:
            query_vector = self._embed_query(query)
            if not any(query_vector):
                return []

            async with self._async_session_maker() as session:
                stmt = select(MemoryModel).where(MemoryModel.session_id == session_id)

                if memory_type:
                    stmt = stmt.where(MemoryModel.memory_type == memory_type)

                stmt = stmt.order_by(
                    MemoryModel.embedding.cosine_distance(query_vector)
                ).limit(k)

                result = await session.execute(stmt)
                records = result.scalars().all()

                memories: List[MemoryRecord] = []
                for r in records:
                    memories.append({
                        "memory_id": str(r.memory_id),
                        "session_id": r.session_id,
                        "memory_type": r.memory_type,
                        "content": r.content,
                        "summary": r.summary or "",
                        "timestamp": r.timestamp,
                        "importance_score": float(r.importance_score),
                        "preference_data": None,
                        "topics_covered": None,
                    })
                return memories

        except Exception as exc:
            logger.error("MemoryStore.retrieve_memories search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Preference storage (item 6 — teaching mode / explanation style)
    # ------------------------------------------------------------------

    async def store_preference(self, session_id: str, preference_data: dict) -> bool:
        """
        Persist a student's preference record for *session_id*.

        *preference_data* is a dict with keys such as:
            ``teaching_mode``   — "quick_review" | "full_explanation" | …
            ``language``        — "ar" | "en"
            ``detail_level``    — inferred detail preference

        The content is stored as a JSON string so it can be retrieved and
        decoded later by ``get_preference``.  A human-readable summary is
        stored in the ``summary`` field.

        This overwrites any previous preference record for the session
        by storing a new record (retrieval always returns the newest one
        via ordering on the timestamp returned from the cosine-sort or
        a direct type query).
        """
        try:
            pref_json = json.dumps(preference_data, ensure_ascii=False)
            summary_parts = []
            if preference_data.get("teaching_mode"):
                summary_parts.append(f"mode={preference_data['teaching_mode']}")
            if preference_data.get("language"):
                summary_parts.append(f"lang={preference_data['language']}")
            summary = "Student preference: " + ", ".join(summary_parts) if summary_parts else "Student preference"

            record = create_memory_record(
                session_id=session_id,
                content=f"[preference] {pref_json}",
                memory_type=_TYPE_PREFERENCE,
                summary=summary,
                importance_score=0.9,   # preferences are highly important
                preference_data=preference_data,
            )
            return await self.store_memory(record)
        except Exception as exc:
            logger.error("MemoryStore.store_preference failed: %s", exc)
            return False

    async def get_preference(self, session_id: str) -> Optional[dict]:
        """
        Retrieve the most recent preference record for *session_id*.

        Returns the decoded ``preference_data`` dict, or ``None`` if no
        preference has been stored yet.
        """
        try:
            async with self._async_session_maker() as session:
                stmt = (
                    select(MemoryModel)
                    .where(
                        MemoryModel.session_id == session_id,
                        MemoryModel.memory_type == _TYPE_PREFERENCE,
                    )
                    .order_by(MemoryModel.timestamp.desc())
                    .limit(1)
                )
                result = await session.execute(stmt)
                row = result.scalars().first()

            if row is None:
                return None

            # The content field is "[preference] <json>"
            content = row.content or ""
            prefix = "[preference] "
            if content.startswith(prefix):
                try:
                    return json.loads(content[len(prefix):])
                except json.JSONDecodeError:
                    pass
            return None

        except Exception as exc:
            logger.error("MemoryStore.get_preference failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Progress tracking (item 6 — topics covered)
    # ------------------------------------------------------------------

    async def update_progress(self, session_id: str, new_topics: List[str]) -> bool:
        """
        Append *new_topics* to the session's running list of covered topics.

        Retrieves the current progress record, merges the new topics
        (deduplicating), and stores a fresh progress record.  The old
        records remain in the DB but are superseded by the newest one.
        """
        if not new_topics:
            return True

        try:
            existing = await self.get_progress(session_id)
            all_topics = list(dict.fromkeys((existing or []) + new_topics))   # dedup, order-preserving

            topics_json = json.dumps(all_topics, ensure_ascii=False)
            record = create_memory_record(
                session_id=session_id,
                content=f"[progress] {topics_json}",
                memory_type=_TYPE_PROGRESS,
                summary=f"Topics covered ({len(all_topics)}): " + ", ".join(all_topics[:5])
                        + ("…" if len(all_topics) > 5 else ""),
                importance_score=0.7,
                topics_covered=all_topics,
            )
            return await self.store_memory(record)
        except Exception as exc:
            logger.error("MemoryStore.update_progress failed: %s", exc)
            return False

    async def get_progress(self, session_id: str) -> Optional[List[str]]:
        """
        Return the list of topics the student has already covered in
        *session_id*, or ``None`` / ``[]`` if no progress has been stored.
        """
        try:
            async with self._async_session_maker() as session:
                stmt = (
                    select(MemoryModel)
                    .where(
                        MemoryModel.session_id == session_id,
                        MemoryModel.memory_type == _TYPE_PROGRESS,
                    )
                    .order_by(MemoryModel.timestamp.desc())
                    .limit(1)
                )
                result = await session.execute(stmt)
                row = result.scalars().first()

            if row is None:
                return []

            content = row.content or ""
            prefix = "[progress] "
            if content.startswith(prefix):
                try:
                    return json.loads(content[len(prefix):])
                except json.JSONDecodeError:
                    pass
            return []

        except Exception as exc:
            logger.error("MemoryStore.get_progress failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Consolidation (placeholder — unchanged)
    # ------------------------------------------------------------------

    async def consolidate_memories(self, session_id: str) -> bool:
        pass