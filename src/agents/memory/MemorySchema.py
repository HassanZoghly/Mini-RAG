from typing import Optional
from typing_extensions import TypedDict
from datetime import datetime, timezone
import uuid


class MemoryRecord(TypedDict):
    """
    Typed dictionary that represents a single memory entry in the
    semantic memory system.

    Memories are stored in the vector database so that retrieval is
    driven by *semantic similarity* rather than chronological order.
    All fields are serialised to JSON and stored as the vector-DB text
    payload so that they survive a round-trip through the embedding +
    search pipeline.

    Fields
    ------
    memory_id : str
        Universally unique identifier (UUID4) for this memory record.
    session_id : str
        Identifier of the conversation or user session that produced
        this memory.  Used to namespace separate memory collections in
        the vector database.
    memory_type : str
        Category of the memory.  One of:
        * ``"short_term"``  – recent interactions not yet consolidated.
        * ``"long_term"``   – consolidated, durable knowledge.
        * ``"semantic"``    – factual / conceptual knowledge extracted
                              from documents.
        * ``"episodic"``    – event-based memories tied to a specific
                              moment in the conversation.
    content : str
        The raw, uncompressed text of the memory (e.g. the original
        question + answer pair).  This is the field that is embedded
        and searched against.
    summary : str
        A condensed, human-readable summary of *content*.  May be
        identical to *content* for short memories, or an LLM-generated
        compression for long ones.
    timestamp : str
        ISO 8601 UTC timestamp indicating when the memory was created
        (e.g. ``"2026-05-24T01:48:00+00:00"``).
    importance_score : float
        A scalar in ``[0.0, 1.0]`` reflecting how important this memory
        is considered to be.  Higher values persist through consolidation;
        lower values are pruned first.
    """

    memory_id: str
    session_id: str
    memory_type: str
    content: str
    summary: str
    timestamp: str
    importance_score: float


def create_memory_record(
    session_id: str,
    content: str,
    memory_type: str,
    summary: str,
    importance_score: float = 0.5,
) -> MemoryRecord:
    """
    Factory function that constructs a ``MemoryRecord`` with a fresh UUID
    and the current UTC timestamp.

    Parameters
    ----------
    session_id : str
        The conversation / session that produced this memory.
    content : str
        Full text content to be embedded and stored.
    memory_type : str
        One of ``"short_term"``, ``"long_term"``, ``"semantic"``,
        or ``"episodic"``.
    summary : str
        Condensed human-readable description of *content*.
    importance_score : float, optional
        Importance weight in ``[0.0, 1.0]``.  Defaults to ``0.5``.

    Returns
    -------
    MemoryRecord
        A fully initialised memory record ready to be passed to
        ``MemoryStore.store_memory``.
    """
    return MemoryRecord(
        memory_id=str(uuid.uuid4()),
        session_id=session_id,
        memory_type=memory_type,
        content=content,
        summary=summary,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        importance_score=importance_score,
    )
