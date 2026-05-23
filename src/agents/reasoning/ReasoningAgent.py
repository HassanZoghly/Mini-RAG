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
