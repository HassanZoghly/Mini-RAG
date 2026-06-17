"""
ReasoningAgent — Phase 3 improvements (items 1, 4B/C, 8).

Changes vs original:
- Removed the hardcoded ``_MAX_CHUNKS = 5`` cap. Chunks are now used in
  full as returned by RetrievalAgent (which already applies dynamic
  top_n sizing per task type via RETRIEVAL_SIZES).
- Summary requests store the raw ordered chunks on the state for
  SummaryGenerator (item 1) rather than dumping them into a flat string.
- ``_build_citations`` assembles a ``citations`` list from chunk metadata
  (source/page/section) for the Sources section (item 8).
- ``assemble_multimodal_context`` no longer injects raw file texts —
  the reworked pipeline uses indexed chunks instead (item 3). The method
  is kept for backward compatibility but delegates to the indexed path
  when retrieved_chunks are available.
- ``execute_multimodal`` populates ``state["citations"]`` for item 8.
"""

from __future__ import annotations

from typing import List

from agents.base import BaseAgent, AgentState
from agents.base.intent_utils import TASK_SUMMARY, classify_task_type

# Maximum memory entries to include in the context string.
_MAX_MEMORIES: int = 5


class ReasoningAgent(BaseAgent):

    def __init__(self, llm_provider, template_parser) -> None:
        self._llm = llm_provider
        self._template_parser = template_parser

    @property
    def agent_name(self) -> str:
        return "ReasoningAgent"

    # ------------------------------------------------------------------
    # Primary (indexed-RAG) pipeline
    # ------------------------------------------------------------------

    async def execute(self, state: AgentState) -> AgentState:
        self.validate_state(state, ["query", "retrieved_chunks"])

        query: str = state["query"]
        route: str = state.get("metadata", {}).get("route", "")
        task_type = classify_task_type(query, route)
        chunks: List[dict] = state.get("retrieved_chunks") or []

        # ── SUMMARY: store chunks for SummaryGenerator, skip text dump ──
        if task_type == TASK_SUMMARY:
            # SummaryGenerator (called from ResponseFormatterAgent) needs
            # the ORM objects; they're stored in the chunks list under "_orm".
            state["reasoning_context"] = "__SUMMARY__"   # sentinel for ResponseFormatterAgent
            state["citations"] = []
            state["agent_trace"].append(
                f"{self.agent_name}: summary task — {len(chunks)} ordered chunks "
                f"ready for SummaryGenerator"
            )
            return state

        sections: List[str] = []
        source_count: int = 0

        # ── 1. Retrieved document chunks (full list, no hard cap) ────────
        if chunks:
            chunk_lines = []
            for idx, chunk in enumerate(chunks, start=1):
                text = chunk.get("text", "").strip()
                if not text:
                    continue
                meta = chunk.get("metadata", {})
                label_parts = []
                if meta.get("source"):
                    label_parts.append(f"src={meta['source']}")
                if meta.get("page"):
                    label_parts.append(f"p={meta['page']}")
                if meta.get("section"):
                    label_parts.append(f"§={meta['section']}")
                label = f" [{', '.join(label_parts)}]" if label_parts else ""
                chunk_lines.append(f"### Document Chunk {idx}{label}\n{text}")

            if chunk_lines:
                sections.append("## Retrieved Knowledge\n" + "\n\n".join(chunk_lines))
                source_count += 1

        # ── 2. Memory context ────────────────────────────────────────────
        memories: List[dict] = state.get("memory_context") or []
        if memories:
            memory_lines = []
            for mem in memories[:_MAX_MEMORIES]:
                content = mem.get("content", "").strip()
                mem_type = mem.get("memory_type", "memory")
                if content:
                    memory_lines.append(f"[{mem_type}] {content}")
            if memory_lines:
                sections.append(
                    "## Relevant Past Context\n" + "\n\n".join(memory_lines)
                )
                source_count += 1

        # ── 3. OCR text ──────────────────────────────────────────────────
        ocr_text: str = (state.get("ocr_text") or "").strip()
        if ocr_text:
            sections.append(f"## Extracted Image Text (OCR)\n{ocr_text}")
            source_count += 1

        # ── 4. Vision description ────────────────────────────────────────
        vision_desc: str = (state.get("vision_description") or "").strip()
        if vision_desc:
            sections.append(f"## Visual Description\n{vision_desc}")
            source_count += 1

        state["reasoning_context"] = "\n\n".join(sections)
        state["citations"] = self._build_citations(chunks)

        state["agent_trace"].append(
            f"{self.agent_name}: assembled context from {source_count} source(s), "
            f"{len(chunks)} chunks, {len(state['citations'])} citations"
        )
        self.log_step(
            f"reasoning_context: {source_count} sources, "
            f"{len(state['reasoning_context'])} chars"
        )
        return state

    # ------------------------------------------------------------------
    # Multimodal pipeline (kept for backward compat, now prefers indexed)
    # ------------------------------------------------------------------

    def assemble_multimodal_context(self, state: AgentState) -> str:
        """
        Assemble context for the multimodal pipeline.

        Priority: indexed retrieved_chunks (from the vector DB) > raw
        file_texts (legacy re-upload path).  Using indexed chunks avoids
        the 16k-char truncation problem that plagued the original design.
        """
        parts = []
        query: str = state.get("query", "")
        route: str = state.get("metadata", {}).get("route", "")
        task_type = classify_task_type(query, route)

        # 1. Indexed chunks (preferred — from vector DB)
        retrieved = state.get("retrieved_chunks") or []
        if retrieved:
            parts.append("=== 📚 Retrieved Lecture Chunks ===")
            for i, chunk in enumerate(retrieved):
                meta = chunk.get("metadata", {})
                label_parts = []
                if meta.get("source"):
                    label_parts.append(meta["source"])
                if meta.get("page"):
                    label_parts.append(f"p.{meta['page']}")
                if meta.get("section"):
                    label_parts.append(meta["section"])
                label = f" [{' | '.join(label_parts)}]" if label_parts else ""
                parts.append(f"--- Chunk {i+1}{label} ---\n{chunk.get('text', '')}")

        # 2. Raw file texts (legacy fallback — only when no indexed chunks)
        elif state.get("file_texts"):
            file_texts = state["file_texts"]
            parts.append("\n=== 📄 Uploaded File Content ===")
            if task_type != TASK_SUMMARY:
                parts.append(
                    "INSTRUCTION: Search the following text ONLY to extract the "
                    "answer to the student's question. Do NOT summarize the files."
                )
            query_lower = query.lower()
            query_terms = {
                w for w in query_lower.replace("?", " ").replace(".", " ").split()
                if len(w) > 2
            }

            def relevance(ft):
                return sum(1 for t in query_terms if t in ft.get("text", "").lower())

            for ft in sorted(file_texts, key=relevance, reverse=True):
                fname = ft.get("file_name", "unknown")
                parts.append(f"--- START OF FILE: {fname} ---")
                parts.append(ft.get("text", ""))
                parts.append(f"--- END OF FILE: {fname} ---\n")

        # 3. Memory
        memory = state.get("memory_context") or []
        if memory:
            parts.append("\n=== 🧠 Session Memory ===")
            for m in memory[:_MAX_MEMORIES]:
                if "role" in m:
                    parts.append(f"{m['role'].upper()}: {m['content']}")
                else:
                    parts.append(m.get("content", ""))

        # 4. Vision
        vision = (state.get("vision_description") or "").strip()
        if vision:
            parts.append(f"\n=== 👁️ Visual Analysis ===\n{vision}")

        # 5. OCR
        ocr = (state.get("ocr_text") or "").strip()
        if ocr:
            parts.append(f"\n=== 📝 OCR Text ===\n{ocr}")

        return "\n\n".join(parts)

    async def execute_multimodal(self, state: AgentState) -> AgentState:
        self.validate_state(state, [])
        context = self.assemble_multimodal_context(state)
        state["reasoning_context"] = context

        sources_used = []
        if state.get("retrieved_chunks"):
            sources_used.append("retrieved_chunks")
        if state.get("memory_context"):
            sources_used.append("memory")
        if state.get("file_texts") and not state.get("retrieved_chunks"):
            sources_used.append("file_texts")
        if state.get("vision_description"):
            sources_used.append("vision")
        if state.get("ocr_text"):
            sources_used.append("ocr_text")

        state["sources_used"] = sources_used
        state["citations"] = self._build_citations(state.get("retrieved_chunks") or [])

        state["agent_trace"].append(
            f"{self.agent_name}: multimodal context — sources: {sources_used}, "
            f"{len(state['citations'])} citations"
        )
        return state

    # ------------------------------------------------------------------
    # Citation builder (item 8)
    # ------------------------------------------------------------------

    def _build_citations(self, chunks: List[dict]) -> List[dict]:
        """
        Build a deduplicated list of source references from chunk metadata.

        Each entry: ``{source, page, section}`` — used by
        ``ResponseFormatterAgent`` to append a "Sources" block.
        """
        seen: set = set()
        citations: List[dict] = []

        for chunk in chunks:
            meta = chunk.get("metadata", {})
            source = meta.get("source", "")
            page = str(meta.get("page", ""))
            section = meta.get("section", "")

            if not source:
                continue

            key = (source, page, section)
            if key in seen:
                continue
            seen.add(key)

            entry: dict = {"source": source}
            if page:
                entry["page"] = page
            if section:
                entry["section"] = section
            citations.append(entry)

        return citations
