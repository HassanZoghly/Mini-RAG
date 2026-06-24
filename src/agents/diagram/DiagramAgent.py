"""
DiagramAgent — generates a Mermaid diagram from lecture chunks.

Flow
----
1. Receive ordered chunks for the project.
2. Run a lightweight concept-extraction pass over batches (map step).
3. Send the extracted concepts + relationships to the LLM with a
   strict Mermaid-output prompt (reduce step).
4. Return a dict: {title, diagram_type, content (mermaid code)}.

The agent is stateless and synchronous, called directly from the
diagram route handler — no LangGraph node needed.

Supported diagram types (chosen automatically based on content shape):
- flowchart TD  (default — works for most hierarchical/process content)
- mindmap       (good for flat topic lists)

The frontend renders the Mermaid code using the Mermaid JS library
embedded in the Streamlit component.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

_CONCEPT_BATCH_CHARS = 10_000   # per concept-extraction call
_CONCEPT_MAX_TOKENS  = 800      # keep extraction tight
_DIAGRAM_MAX_TOKENS  = 1_200    # Mermaid code is compact


class DiagramAgent:
    """
    Generates a Mermaid-format concept diagram from lecture chunks.

    Parameters
    ----------
    generation_client : LLM provider
    template_parser   : Template parser instance
    language          : "en" | "ar"  (affects prompt language only;
                        Mermaid node labels are always in the lecture language)
    """

    def __init__(self, generation_client, template_parser, language: str = "en") -> None:
        self._llm             = generation_client
        self._template_parser = template_parser
        self._language        = language if language in ("ar", "en") else "en"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, chunks: list) -> dict:
        """
        Generate a Mermaid diagram from *chunks*.

        Returns
        -------
        dict with keys:
            title        — short title derived from content
            diagram_type — "flowchart"
            content      — raw Mermaid code string
        """
        if not chunks:
            return self._error_diagram("No lecture content provided.")

        # Step 1 — concept extraction (map over batches)
        content_blocks = self._build_content_blocks(chunks)
        concepts_text  = self._extract_concepts(content_blocks)

        if not concepts_text:
            return self._error_diagram("Could not extract concepts from the lecture.")

        # Step 2 — diagram generation (reduce)
        mermaid_code = self._generate_mermaid(concepts_text)
        if not mermaid_code:
            return self._error_diagram("Could not generate diagram code.")

        mermaid_code = self._clean_mermaid(mermaid_code)
        title = self._extract_title(mermaid_code, chunks)

        return {
            "title":        title,
            "diagram_type": "flowchart",
            "content":      mermaid_code,
        }

    # ------------------------------------------------------------------
    # Step 1 — concept extraction
    # ------------------------------------------------------------------

    def _build_content_blocks(self, chunks: list) -> list[str]:
        """Split chunks into character-capped batches."""
        batches: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for chunk in chunks:
            if hasattr(chunk, "chunk_text"):
                text = (chunk.chunk_text or "").strip()
                meta = chunk.chunk_metadata or {}
            else:
                text = (chunk.get("text", "")).strip()
                meta = chunk.get("metadata", {})

            if not text:
                continue

            # Prepend section heading as a hint for concept extraction
            section = meta.get("section", "")
            block = f"[{section}]\n{text}" if section else text
            block_len = len(block)

            if current_parts and current_len + block_len > _CONCEPT_BATCH_CHARS:
                batches.append("\n\n".join(current_parts))
                current_parts = [block]
                current_len = block_len
            else:
                current_parts.append(block)
                current_len += block_len

        if current_parts:
            batches.append("\n\n".join(current_parts))

        return batches

    def _extract_concepts(self, batches: list[str]) -> str:
        """
        Run a concept-extraction prompt over each batch, collecting
        bullet-point concept/relationship lists.
        """
        system = self._template_parser.get("rag", "diagram_concept_system")
        chat_history = [self._llm.construct_prompt(prompt=system, role=self._llm.enums.SYSTEM.value)]

        all_concepts: list[str] = []
        for i, batch in enumerate(batches, 1):
            try:
                user_prompt = self._template_parser.get("rag", "diagram_concept_user", {"content": batch})
                result = self._llm.generate_text(
                    prompt=user_prompt, 
                    chat_history=chat_history,
                    max_output_tokens=800
                )
                if result:
                    all_concepts.append(f"--- Part {i} ---\n{result.strip()}")
            except Exception as exc:
                logger.warning("DiagramAgent: concept extraction batch %d failed: %s", i, exc)

        return "\n\n".join(all_concepts)

    # ------------------------------------------------------------------
    # Step 2 — Mermaid generation
    # ------------------------------------------------------------------

    def _generate_mermaid(self, concepts_text: str) -> Optional[str]:
        system = self._template_parser.get("rag", "diagram_mermaid_system")
        user_prompt = self._template_parser.get("rag", "diagram_mermaid_user", {"concepts_text": concepts_text})

        chat_history = [
            self._llm.construct_prompt(
                prompt=system, role=self._llm.enums.SYSTEM.value
            )
        ]

        try:
            return self._llm.generate_text(
                prompt=user_prompt,
                chat_history=chat_history,
                max_output_tokens=_DIAGRAM_MAX_TOKENS,
            )
        except Exception as exc:
            logger.error("DiagramAgent: Mermaid generation failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clean_mermaid(self, raw: str) -> str:
        """Strip markdown fences and normalise whitespace."""
        cleaned = re.sub(r"```(?:mermaid)?", "", raw)
        cleaned = cleaned.strip("`").strip()

        # Ensure it starts with a valid Mermaid header
        if not re.match(r"^\s*(graph|flowchart|mindmap|sequenceDiagram)", cleaned, re.I):
            cleaned = "graph TD\n" + cleaned

        return cleaned

    def _extract_title(self, mermaid_code: str, chunks: list) -> str:
        """Derive a short title from the first subgraph label or first chunk metadata."""
        # Try subgraph label
        m = re.search(r"subgraph\s+(.+)", mermaid_code)
        if m:
            return m.group(1).strip().strip('"').strip("'")

        # Try first chunk source
        for chunk in chunks[:3]:
            if hasattr(chunk, "chunk_metadata"):
                meta = chunk.chunk_metadata or {}
            else:
                meta = chunk.get("metadata", {})
            src = meta.get("source", "")
            if src:
                # Strip file extension
                return re.sub(r"\.\w+$", "", src).replace("_", " ").replace("-", " ")

        return "Lecture Concept Map"

    def _error_diagram(self, message: str) -> dict:
        return {
            "title":        "Error",
            "diagram_type": "flowchart",
            "content":      f"graph TD\n    A[\"{message}\"]",
        }