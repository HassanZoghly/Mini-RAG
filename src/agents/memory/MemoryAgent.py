"""
MemoryAgent — Phase 5: preference + progress tracking (item 6).

Changes vs original:
- ``execute`` now also loads the session's teaching-mode preference and
  progress record, and writes them into the state so downstream agents
  (``ResponseFormatterAgent``) can adapt tone/depth automatically.
  • ``state["teaching_mode"]`` is set from stored preference when the
    current request doesn't carry an explicit one.
  • A context note about already-covered topics is prepended to
    ``state["memory_context"]`` so the LLM knows what to skip.

- ``save_interaction`` now additionally:
  • Calls ``update_progress`` with topics extracted from the new
    interaction (coarse topic extraction from the query).
  • Calls ``store_preference`` whenever ``state["teaching_mode"]``
    is set, so the student's mode choice persists across messages.

All original behaviour (semantic short-term memory, k=5 retrieval) is
kept unchanged.
"""

from __future__ import annotations

import re
from typing import List, Optional

from agents.base import BaseAgent, AgentState
from .MemoryStore import MemoryStore
from .MemorySchema import create_memory_record


class MemoryAgent(BaseAgent):
    """
    Agent responsible for loading and saving per-session memory.

    Execution order in the graph (Phase 4):
        router → **memory** → query_rewrite → retrieval → reasoning → response

    Memory is now loaded BEFORE query rewriting so the rewriter has
    access to recent conversation history.
    """

    def __init__(self, memory_store: MemoryStore) -> None:
        self._memory_store = memory_store

    @property
    def agent_name(self) -> str:
        return "MemoryAgent"

    # ------------------------------------------------------------------
    # Read — called at the start of every request
    # ------------------------------------------------------------------

    async def execute(self, state: AgentState) -> AgentState:
        """
        Load:
        1. Semantically relevant short-term memories (k=5).
        2. The session's teaching-mode / style preference (if stored).
        3. The session's learning-progress record (topics already covered).

        Writes into state:
        - ``state["memory_context"]``  — list of short-term memory dicts.
        - ``state["teaching_mode"]``   — restored from preference when not
                                         explicitly set by the current request.
        """
        self.validate_state(state, ["query", "metadata"])

        session_id: str = state["metadata"].get("session_id", "default")
        query: str = state["query"]

        self.log_step(
            f"loading memories: query='{query[:60]}' session='{session_id}'"
        )

        # ── 1. Short-term semantic memories ─────────────────────────────
        try:
            records = await self._memory_store.retrieve_memories(
                query=query,
                session_id=session_id,
                k=5,
            )
        except Exception as exc:
            self.log_step(f"memory retrieval raised: {exc}")
            records = []

        memory_context: List[dict] = [
            {
                "content":          r["content"],
                "memory_type":      r["memory_type"],
                "importance_score": r["importance_score"],
                "summary":          r["summary"],
                "timestamp":        r["timestamp"],
            }
            for r in records
        ]



        # ── 3. Progress (topics covered) ─────────────────────────────────
        try:
            covered_topics = await self._memory_store.get_progress(session_id)
            if covered_topics:
                topics_str = ", ".join(covered_topics[:10])
                if len(covered_topics) > 10:
                    topics_str += f" … (+{len(covered_topics) - 10} more)"
                progress_entry = {
                    "content": (
                        f"[Session progress] Topics already explained to this student: "
                        f"{topics_str}. Acknowledge prior coverage briefly instead of "
                        f"re-explaining from scratch."
                    ),
                    "memory_type":      "progress",
                    "importance_score": 0.8,
                    "summary":          f"Covered topics: {topics_str}",
                    "timestamp":        "",
                }
                # Prepend so it's visible to the query rewriter and LLM
                memory_context = [progress_entry] + memory_context
                self.log_step(f"loaded {len(covered_topics)} covered topics")
        except Exception as exc:
            self.log_step(f"progress load failed (non-fatal): {exc}")
            covered_topics = []

        # ── 4. Current Topic extraction (Intent Handling) ────────────────
        current_topics = self._extract_topics(query)
        if current_topics:
            if "metadata" not in state: state["metadata"] = {}
            state["metadata"]["current_topic"] = current_topics[0]
            self.log_step(f"extracted new current_topic: {current_topics[0]}")
        elif covered_topics:
            if "metadata" not in state: state["metadata"] = {}
            state["metadata"]["current_topic"] = covered_topics[-1]
            self.log_step(f"kept previous current_topic: {covered_topics[-1]}")
        else:
            if "metadata" not in state: state["metadata"] = {}
            state["metadata"]["current_topic"] = ""

        state["memory_context"] = memory_context
        state["agent_trace"].append(
            f"{self.agent_name}: loaded {len(memory_context)} memory entries "
            f"(incl. preference + progress) for session='{session_id}'"
        )
        return state

    # ------------------------------------------------------------------
    # Write — called after the final response is ready
    # ------------------------------------------------------------------

    async def save_interaction(self, state: AgentState) -> None:
        """
        Persist:
        1. The completed Q&A as a short-term memory.
        2. The teaching-mode preference (if set on this turn).
        3. Updated learning-progress topics extracted from the query.

        This is a fire-and-forget method — errors are logged but never
        re-raised so they don't disrupt the response the student receives.
        """
        query:      str = state.get("query", "")
        response:   str = state.get("final_response", "")
        session_id: str = (state.get("metadata") or {}).get("session_id", "default")


        if not query:
            return

        # ── 1. Short-term Q&A memory ─────────────────────────────────────
        content = f"Q: {query}\nA: {response}" if response else f"Q: {query}"
        record = create_memory_record(
            session_id=session_id,
            content=content,
            memory_type="short_term",
            summary=query[:200],
            importance_score=0.5,
        )
        try:
            success = await self._memory_store.store_memory(record)
            if success:
                self.log_step(
                    f"saved short_term memory for session='{session_id}'"
                )
            else:
                self.log_step("store_memory returned False — not persisted")
        except Exception as exc:
            self.log_step(f"save short_term failed: {exc}")



        # ── 3. Progress — extract topics from query ──────────────────────
        topics = self._extract_topics(query)
        if topics:
            try:
                await self._memory_store.update_progress(
                    session_id=session_id,
                    new_topics=topics,
                )
                self.log_step(f"updated progress with topics: {topics}")
            except Exception as exc:
                self.log_step(f"update progress failed (non-fatal): {exc}")

    # ------------------------------------------------------------------
    # Lightweight topic extractor
    # ------------------------------------------------------------------

    # Stop-words to filter out when extracting topic tokens from a query.
    _STOP_WORDS: frozenset = frozenset({
        "what", "how", "why", "when", "where", "who", "which", "is", "are",
        "the", "a", "an", "and", "or", "of", "in", "on", "to", "for",
        "explain", "describe", "tell", "me", "about", "please", "can", "you",
        # Arabic equivalents
        "ما", "كيف", "لماذا", "متى", "أين", "من", "هو", "هي",
        "اشرح", "وضح", "فسر", "اذكر", "ما", "هل",
    })

    def _extract_topics(self, query: str, max_topics: int = 3) -> List[str]:
        """
        Extract candidate topic labels from *query* using a simple
        heuristic: keep capitalised words / meaningful tokens after
        removing stop-words.  Returns at most *max_topics* labels.

        This is intentionally simple — a lightweight noun-phrase extractor
        rather than a full NLP pipeline so it doesn't add dependencies.
        """
        # Remove punctuation and split
        tokens = re.sub(r"[^\w\s\u0600-\u06FF]", " ", query).split()

        candidates: List[str] = []
        for tok in tokens:
            tok_clean = tok.strip()
            if not tok_clean or len(tok_clean) < 3:
                continue
            if tok_clean.lower() in self._STOP_WORDS:
                continue
            # Prefer capitalised English terms (likely proper-noun topics)
            if tok_clean[0].isupper() or len(tok_clean) >= 5:
                candidates.append(tok_clean)

        # Deduplicate preserving order
        seen: set = set()
        topics: List[str] = []
        for c in candidates:
            key = c.lower()
            if key not in seen:
                seen.add(key)
                topics.append(c)
            if len(topics) >= max_topics:
                break

        return topics
