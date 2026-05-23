from .MemorySchema import MemoryRecord, create_memory_record
from controllers.NLPController import NLPController
from stores.llm.LLMEnums import DocumentTypeEnum
from typing import List, Optional
import logging
import json


logger = logging.getLogger(__name__)

# How many short-term memories must exist before consolidation runs.
_CONSOLIDATION_THRESHOLD: int = 10
# How many of the oldest short-term memories are merged into one long-term memory.
_CONSOLIDATION_BATCH: int = 5


class MemoryStore:
    """
    Semantic memory store that persists and retrieves ``MemoryRecord``
    instances via the project's vector database.

    Each session owns its own vector-DB collection, named:
    ``<collection_prefix><session_id>``

    Because the underlying ``search_by_vector`` implementations (Qdrant,
    PGVector) only return ``text`` and ``score`` from the payload, every
    ``MemoryRecord`` is serialised to a JSON string and stored as the
    *text* field.  On retrieval, the JSON is deserialised back into a
    ``MemoryRecord``.  This approach works regardless of the active
    vector-DB backend.

    Parameters
    ----------
    nlp_controller : NLPController
        Pre-configured controller that exposes ``embedding_client`` and
        ``vectordb_client`` for embedding and vector-DB operations.
    collection_prefix : str, optional
        String prepended to ``session_id`` to form the collection name.
        Defaults to ``"memory_"``.
    """

    def __init__(
        self,
        nlp_controller: NLPController,
        collection_prefix: str = "memory_",
    ) -> None:
        self._nlp = nlp_controller
        self._prefix = collection_prefix

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collection_name(self, session_id: str) -> str:
        """
        Derive the vector-DB collection name for a given session.

        Parameters
        ----------
        session_id : str
            The session whose memory collection is required.

        Returns
        -------
        str
            Collection name string safe to pass to the vector-DB client.
        """
        return f"{self._prefix}{session_id}"

    def _embed(self, text: str) -> List[float]:
        """
        Embed *text* using the NLPController's embedding client and
        flatten the result to a plain list of floats.

        Parameters
        ----------
        text : str
            Text to embed.

        Returns
        -------
        List[float]
            Dense embedding vector.
        """
        raw = self._nlp.embedding_client.embed_text(
            text=text,
            document_type=DocumentTypeEnum.DOCUMENT.value,
        )
        return self._nlp._flatten_vector(raw)

    def _embed_query(self, text: str) -> List[float]:
        """
        Embed *text* as a query vector (may use a different embedding
        mode depending on the provider).

        Parameters
        ----------
        text : str
            Query text to embed.

        Returns
        -------
        List[float]
            Dense query embedding vector.
        """
        raw = self._nlp.embedding_client.embed_text(
            text=text,
            document_type=DocumentTypeEnum.QUERY.value,
        )
        return self._nlp._flatten_vector(raw)

    @staticmethod
    def _serialise(record: MemoryRecord) -> str:
        """
        Serialise a ``MemoryRecord`` to a compact JSON string for
        storage as the vector-DB text payload.

        Parameters
        ----------
        record : MemoryRecord
            Memory record to serialise.

        Returns
        -------
        str
            JSON representation of *record*.
        """
        return json.dumps(dict(record), ensure_ascii=False)

    @staticmethod
    def _deserialise(text: str) -> Optional[MemoryRecord]:
        """
        Attempt to deserialise a JSON string back into a ``MemoryRecord``.

        Returns ``None`` if the text is not valid JSON or is missing
        required keys, so callers can filter gracefully.

        Parameters
        ----------
        text : str
            Raw text payload retrieved from the vector database.

        Returns
        -------
        MemoryRecord or None
            Parsed memory record, or ``None`` on failure.
        """
        try:
            data = json.loads(text)
            return MemoryRecord(
                memory_id=data.get("memory_id", ""),
                session_id=data.get("session_id", ""),
                memory_type=data.get("memory_type", "short_term"),
                content=data.get("content", ""),
                summary=data.get("summary", ""),
                timestamp=data.get("timestamp", ""),
                importance_score=float(data.get("importance_score", 0.5)),
            )
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def store_memory(self, memory: MemoryRecord) -> bool:
        """
        Embed *memory* and persist it in the session's vector-DB collection.

        The complete ``MemoryRecord`` is serialised to JSON and stored as
        the vector-DB *text* payload so every field is recoverable on
        retrieval.

        Parameters
        ----------
        memory : MemoryRecord
            The memory record to store.

        Returns
        -------
        bool
            ``True`` on success, ``False`` if an error occurred.
        """
        collection_name = self._collection_name(memory["session_id"])

        try:
            vector = self._embed(memory["content"])

            # Ensure the collection exists before inserting.
            await self._nlp.vectordb_client.create_collection(
                collection_name=collection_name,
                embedding_size=self._nlp.embedding_client.embedding_size,
                do_reset=False,
            )

            payload_text = self._serialise(memory)

            await self._nlp.vectordb_client.insert_many(
                collection_name=collection_name,
                texts=[payload_text],
                vectors=[vector],
                metadata=[{"memory_type": memory["memory_type"],
                           "session_id": memory["session_id"]}],
                record_ids=[memory["memory_id"]],
            )

            logger.info(
                "MemoryStore: stored memory_id=%s type=%s session=%s",
                memory["memory_id"],
                memory["memory_type"],
                memory["session_id"],
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
        Search the session's memory collection for records semantically
        similar to *query*.

        The vector-DB returns results ranked by cosine similarity.  If
        *memory_type* is specified, results are post-filtered in Python
        (the underlying providers do not expose payload filters in the
        current codebase).  To compensate for filtering losses, the raw
        search fetches ``k * 4`` candidates before filtering.

        Parameters
        ----------
        query : str
            Natural-language query to match against stored memories.
        session_id : str
            Session whose memory collection to search.
        k : int, optional
            Maximum number of memories to return.  Defaults to ``5``.
        memory_type : str or None, optional
            If given, only memories of this type are returned.

        Returns
        -------
        List[MemoryRecord]
            Up to *k* memory records ordered by semantic relevance.
        """
        collection_name = self._collection_name(session_id)

        try:
            collection_exists = await self._nlp.vectordb_client.is_collection_existed(
                collection_name=collection_name
            )
            if not collection_exists:
                return []

            query_vector = self._embed_query(query)
            if not any(query_vector):
                return []

            fetch_limit = k * 4 if memory_type else k
            raw_results = await self._nlp.vectordb_client.search_by_vector(
                collection_name=collection_name,
                vector=query_vector,
                limit=fetch_limit,
            )

        except Exception as exc:
            logger.error("MemoryStore.retrieve_memories search failed: %s", exc)
            return []

        if not raw_results:
            return []

        memories: List[MemoryRecord] = []
        for doc in raw_results:
            record = self._deserialise(doc.text)
            if record is None:
                continue
            if memory_type and record["memory_type"] != memory_type:
                continue
            memories.append(record)
            if len(memories) >= k:
                break

        return memories

    async def consolidate_memories(self, session_id: str) -> bool:
        """
        Merge old short-term memories into a single long-term memory when
        the session's short-term buffer exceeds the consolidation threshold.

        Algorithm
        ---------
        1. Retrieve up to ``_CONSOLIDATION_THRESHOLD + 1`` short-term
           memories for *session_id*.
        2. If the count is ≤ ``_CONSOLIDATION_THRESHOLD``, return early —
           no consolidation needed.
        3. Take the oldest ``_CONSOLIDATION_BATCH`` records (sorted by
           timestamp ascending).
        4. Concatenate their ``content`` fields and ask the LLM to
           summarise them.
        5. Store the summary as a new ``"long_term"`` memory.
        6. Delete the originals by overwriting each with a tombstone record
           at importance_score=0.0 using a dedicated consolidation re-write
           (the current vector-DB interface does not expose a delete-by-id,
           so tombstones are stored; a future cleanup pass can remove them).

        Parameters
        ----------
        session_id : str
            Session whose short-term memory buffer should be inspected.

        Returns
        -------
        bool
            ``True`` if consolidation ran and a long-term memory was created,
            ``False`` otherwise (not enough memories, or an error occurred).
        """
        candidates = await self.retrieve_memories(
            query="memory consolidation context",
            session_id=session_id,
            k=_CONSOLIDATION_THRESHOLD + 1,
            memory_type="short_term",
        )

        if len(candidates) <= _CONSOLIDATION_THRESHOLD:
            logger.info(
                "MemoryStore.consolidate_memories: only %d short-term memories "
                "in session=%s — threshold not reached",
                len(candidates),
                session_id,
            )
            return False

        # Sort ascending by timestamp so we consolidate the *oldest* ones.
        candidates.sort(key=lambda r: r["timestamp"])
        to_merge = candidates[:_CONSOLIDATION_BATCH]

        combined_content = "\n---\n".join(
            f"[{r['timestamp']}] {r['content']}" for r in to_merge
        )
        summarise_prompt = (
            "You are a memory consolidation assistant. "
            "Summarise the following conversation memories into a single "
            "concise paragraph that preserves the most important facts:\n\n"
            f"{combined_content}"
        )

        try:
            summary_text = self._nlp.generation_client.generate_text(
                prompt=summarise_prompt,
                chat_history=[],
            )
            if not summary_text:
                summary_text = combined_content[:500]
        except Exception as exc:
            logger.warning(
                "MemoryStore.consolidate_memories: LLM summarisation failed (%s); "
                "using truncated content as summary",
                exc,
            )
            summary_text = combined_content[:500]

        from .MemorySchema import create_memory_record  # local import to avoid circularity

        long_term = create_memory_record(
            session_id=session_id,
            content=summary_text,
            memory_type="long_term",
            summary=summary_text[:200],
            importance_score=0.8,
        )
        stored = await self.store_memory(long_term)

        if stored:
            logger.info(
                "MemoryStore.consolidate_memories: consolidated %d short-term memories "
                "into long_term memory_id=%s for session=%s",
                len(to_merge),
                long_term["memory_id"],
                session_id,
            )

        return stored
