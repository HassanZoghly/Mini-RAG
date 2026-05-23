from agents.base import BaseAgent, AgentState
from .MemoryStore import MemoryStore
from .MemorySchema import create_memory_record
from typing import List


class MemoryAgent(BaseAgent):
    """
    Agent that loads semantically relevant memories into the pipeline
    state before answer generation, and persists completed Q&A
    interactions as new short-term memories afterward.

    The agent queries the ``MemoryStore`` using the user's natural-language
    query as a similarity search key, so results are ranked by *semantic
    relevance* rather than recency.

    Parameters
    ----------
    memory_store : MemoryStore
        Configured store instance backed by the project's vector database.
    """

    def __init__(self, memory_store: MemoryStore) -> None:
        self._memory_store = memory_store

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def agent_name(self) -> str:
        """Return the display name used in logs and agent trace entries."""
        return "MemoryAgent"

    async def execute(self, state: AgentState) -> AgentState:
        """
        Retrieve semantically relevant memories for the current query and
        populate ``state["memory_context"]``.

        Steps
        -----
        1. Validate that ``query`` and ``metadata`` are present in *state*.
        2. Read ``session_id`` from ``state["metadata"]``; fall back to
           ``"default"`` when absent.
        3. Call ``memory_store.retrieve_memories`` with the current query.
        4. Normalise each returned ``MemoryRecord`` to a lightweight dict
           ``{content, memory_type, importance_score, summary, timestamp}``
           and write the list to ``state["memory_context"]``.
        5. Append a trace entry and return the updated state.

        Parameters
        ----------
        state : AgentState
            Current pipeline state.  Must contain ``query`` and
            ``metadata``.

        Returns
        -------
        AgentState
            Updated state with ``memory_context`` populated.
        """
        self.validate_state(state, ["query", "metadata"])

        session_id: str = state["metadata"].get("session_id", "default")
        query: str = state["query"]

        self.log_step(
            f"retrieving memories for query='{query[:60]}' session='{session_id}'"
        )

        try:
            records = await self._memory_store.retrieve_memories(
                query=query,
                session_id=session_id,
                k=5,
            )
        except Exception as exc:
            self.log_step(f"memory retrieval raised an exception: {exc}")
            records = []

        memory_context: List[dict] = [
            {
                "content": r["content"],
                "memory_type": r["memory_type"],
                "importance_score": r["importance_score"],
                "summary": r["summary"],
                "timestamp": r["timestamp"],
            }
            for r in records
        ]

        state["memory_context"] = memory_context
        state["agent_trace"].append(
            f"{self.agent_name}: loaded {len(memory_context)} memories"
        )

        self.log_step(f"loaded {len(memory_context)} memories from session='{session_id}'")
        return state

    # ------------------------------------------------------------------
    # Post-interaction persistence
    # ------------------------------------------------------------------

    async def save_interaction(self, state: AgentState) -> None:
        """
        Persist the completed Q&A interaction as a short-term memory so
        it is available for future semantic lookups.

        This method should be called *after* the final response has been
        written to ``state["final_response"]``.  It is a fire-and-forget
        operation; errors are logged but not re-raised so they never
        disrupt the user-facing response.

        The stored content concatenates the query and response in a
        natural dialogue format:
        ``"Q: <query>\\nA: <final_response>"``

        Parameters
        ----------
        state : AgentState
            Pipeline state that must have ``query``, ``metadata``, and
            (ideally) ``final_response`` populated.
        """
        query: str = state.get("query", "")
        response: str = state.get("final_response", "")
        session_id: str = (state.get("metadata") or {}).get("session_id", "default")

        if not query:
            return

        content = f"Q: {query}\nA: {response}" if response else f"Q: {query}"
        summary = query[:200]

        record = create_memory_record(
            session_id=session_id,
            content=content,
            memory_type="short_term",
            summary=summary,
            importance_score=0.5,
        )

        try:
            success = await self._memory_store.store_memory(record)
            if success:
                self.log_step(
                    f"saved short_term memory memory_id={record['memory_id']} "
                    f"for session='{session_id}'"
                )
            else:
                self.log_step("store_memory returned False — interaction not persisted")
        except Exception as exc:
            self.log_step(f"save_interaction failed silently: {exc}")
