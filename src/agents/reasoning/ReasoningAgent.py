from agents.base import BaseAgent, AgentState
from typing import List


# Maximum number of retrieved chunks to include in the reasoning context.
_MAX_CHUNKS: int = 5
# Maximum number of memory context entries to include.
_MAX_MEMORIES: int = 3


class ReasoningAgent(BaseAgent):
    """
    Agent that assembles a structured, labelled context string from all
    available evidence sources before answer generation.

    This agent performs **no LLM call** — it is a pure data-assembly step
    whose output drives the prompt built by ``ResponseAgent``.  Separating
    context assembly from generation makes each concern independently
    testable and keeps prompt-construction logic out of the LLM call path.

    Sources assembled (in order, skipping empty ones):

    1. **Retrieved document chunks** — top-k results from the vector DB.
    2. **Memory context** — semantically relevant past interactions.
    3. **OCR text** — text extracted from attached images by ``OCRAgent``.
    4. **Vision description** — image understanding from ``VisionAgent``.

    Parameters
    ----------
    llm_provider : object
        Reserved for future use (e.g. chain-of-thought reasoning).
        Accepted but not used in the current implementation.
    template_parser : object
        ``TemplateParser`` instance.  Reserved for future prompt-template
        integration.  Accepted but not used in the current implementation.
    """

    def __init__(self, llm_provider, template_parser) -> None:
        self._llm = llm_provider
        self._template_parser = template_parser

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def agent_name(self) -> str:
        """Return the display name used in logs and agent trace entries."""
        return "ReasoningAgent"

    async def execute(self, state: AgentState) -> AgentState:
        """
        Assemble a structured context string from all evidence sources.

        Steps
        -----
        1. Validate that ``query`` and ``retrieved_chunks`` are present.
        2. Build a section for each non-empty evidence source.
        3. Join sections with double newlines and write to
           ``state["reasoning_context"]``.
        4. Count the total number of contributing sources and append a
           trace entry.
        5. Return the updated state.

        Parameters
        ----------
        state : AgentState
            Current pipeline state.  Must contain ``query`` and
            ``retrieved_chunks``.

        Returns
        -------
        AgentState
            Updated state with ``reasoning_context`` populated.
        """
        self.validate_state(state, ["query", "retrieved_chunks"])

        sections: List[str] = []
        source_count: int = 0

        # -- 1. Retrieved document chunks ----------------------------------
        chunks: List[dict] = state.get("retrieved_chunks") or []
        if chunks:
            chunk_lines = []
            for idx, chunk in enumerate(chunks[:_MAX_CHUNKS], start=1):
                text = chunk.get("text", "").strip()
                if text:
                    chunk_lines.append(f"### Document Chunk {idx}\n{text}")
            if chunk_lines:
                sections.append(
                    "## Retrieved Knowledge\n" + "\n\n".join(chunk_lines)
                )
                source_count += 1

        # -- 2. Memory context ---------------------------------------------
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

        # -- 3. OCR text ---------------------------------------------------
        ocr_text: str = (state.get("ocr_text") or "").strip()
        if ocr_text:
            sections.append(f"## Extracted Image Text (OCR)\n{ocr_text}")
            source_count += 1

        # -- 4. Vision description -----------------------------------------
        vision_desc: str = (state.get("vision_description") or "").strip()
        if vision_desc:
            sections.append(f"## Visual Description\n{vision_desc}")
            source_count += 1

        state["reasoning_context"] = "\n\n".join(sections)

        state["agent_trace"].append(
            f"{self.agent_name}: assembled context from {source_count} source(s)"
        )
        self.log_step(
            f"assembled reasoning_context from {source_count} source(s), "
            f"{len(state['reasoning_context'])} chars total"
        )
        return state

    def assemble_multimodal_context(self, state: AgentState) -> str:
        """
        Builds labeled context from ALL available sources.
        Only includes sections that have content.
        """
        parts = []

        retrieved = state.get("retrieved_chunks", [])
        if retrieved:
            parts.append("=== Retrieved Document Chunks ===")
            for i, chunk in enumerate(retrieved):
                parts.append(f"--- Chunk {i+1} ---\n{chunk.get('text', '')}")

        memory = state.get("memory_context", [])
        if memory:
            parts.append("\n=== Session Memory ===")
            for m in memory:
                parts.append(f"{m['role'].upper()}: {m['content']}")

        file_texts = state.get("file_texts", [])
        if file_texts:
            parts.append("\n=== Uploaded File Content ===")
            for ft in file_texts:
                parts.append(f"--- Source: {ft.get('file_name', 'unknown')} ---\n{ft.get('text', '')}")

        vision = state.get("vision_description", "").strip()
        if vision:
            parts.append(f"\n=== Visual Analysis ===\n{vision}")

        ocr = state.get("ocr_text", "").strip()
        if ocr:
            parts.append(f"\n=== OCR Text ===\n{ocr}")

        return "\n".join(parts)

    async def execute_multimodal(self, state: AgentState) -> AgentState:
        """
        Calls assemble_multimodal_context().
        Sets state["reasoning_context"].
        Sets state["sources_used"] = list of non-empty section names.
        Trace: "ReasoningAgent: multimodal context — sources: {sources_used}"
        """
        self.validate_state(state, [])
        context = self.assemble_multimodal_context(state)
        state["reasoning_context"] = context
        
        sources_used = []
        if state.get("retrieved_chunks"): sources_used.append("retrieved_chunks")
        if state.get("memory_context"): sources_used.append("memory")
        if state.get("file_texts"): sources_used.append("file_texts")
        if state.get("vision_description"): sources_used.append("vision")
        if state.get("ocr_text"): sources_used.append("ocr_text")
            
        state["sources_used"] = sources_used
        state["agent_trace"].append(
            f"{self.agent_name}: multimodal context — sources: {sources_used}"
        )
        return state
