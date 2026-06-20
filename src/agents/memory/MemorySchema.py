from typing import Optional
from typing_extensions import TypedDict
from datetime import datetime, timezone
import uuid


class MemoryRecord(TypedDict, total=False):
    """
    Typed dictionary that represents a single memory entry in the
    semantic memory system.

    Phase 5 additions
    -----------------
    ``preference_data`` : dict | None
        Arbitrary key-value dict stored alongside ``"preference"`` type
        records.  Used to persist per-session settings such as the student's
        chosen teaching mode, detected explanation-style preference, and
        language.  Not embedded — only stored as metadata in ``summary``.

    ``topics_covered`` : list[str] | None
        Filled on ``"progress"`` type records.  Tracks the lecture topics
        the student has asked about so the tutor can acknowledge already-
        explained concepts without repeating them at length.

    All other fields are the same as before.
    """

    memory_id: str
    session_id: str
    memory_type: str
    content: str
    summary: str
    timestamp: str
    importance_score: float
    # Optional extended fields (phase 5)
    preference_data: Optional[dict]
    topics_covered: Optional[list]


def create_memory_record(
    session_id: str,
    content: str,
    memory_type: str,
    summary: str,
    importance_score: float = 0.5,
    preference_data: Optional[dict] = None,
    topics_covered: Optional[list] = None,
) -> MemoryRecord:
    """
    Factory that constructs a ``MemoryRecord`` with a fresh UUID and the
    current UTC timestamp.

    Parameters
    ----------
    session_id : str
        The conversation / session that produced this memory.
    content : str
        Full text content to be embedded and stored.
    memory_type : str
        One of ``"short_term"``, ``"long_term"``, ``"semantic"``,
        ``"episodic"``, ``"preference"``, or ``"progress"``.
    summary : str
        Condensed human-readable description of *content*.
    importance_score : float, optional
        Importance weight in ``[0.0, 1.0]``.  Defaults to ``0.5``.
    preference_data : dict, optional
        Key-value settings to persist on ``"preference"`` records.
    topics_covered : list, optional
        Topic labels to persist on ``"progress"`` records.

    Returns
    -------
    MemoryRecord
        A fully initialised memory record ready for ``MemoryStore.store_memory``.
    """
    record: MemoryRecord = {
        "memory_id": str(uuid.uuid4()),
        "session_id": session_id,
        "memory_type": memory_type,
        "content": content,
        "summary": summary,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "importance_score": importance_score,
        "preference_data": preference_data,
        "topics_covered": topics_covered,
    }
    return record