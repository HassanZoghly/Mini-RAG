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
    language          : "en" | "ar"  (affects prompt language only;
                        Mermaid node labels are always in the lecture language)
    """

    def __init__(self, generation_client, language: str = "en") -> None:
        self._llm      = generation_client
        self._language = language if language in ("ar", "en") else "en"

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
        if self._language == "ar":
            system = (
                "أنت محلل محتوى. استخرج المفاهيم والمواضيع الرئيسية والعلاقات بينها "
                "من النص المُقدَّم. أعد قائمة منظمة من النقاط فقط — لا شرح إضافي."
            )
            user_tmpl = (
                "استخرج من النص التالي:\n"
                "1. المواضيع الرئيسية (أقسام المحاضرة)\n"
                "2. المفاهيم الفرعية لكل موضوع\n"
                "3. العلاقات بين المفاهيم (مثال: A يؤدي إلى B، A هو نوع من B)\n\n"
                "النص:\n{content}\n\n"
                "أعد فقط قائمة نقاط منظمة:"
            )
        else:
            system = (
                "You are a content analyst. Extract the main topics, concepts, and "
                "their relationships from the provided text. Return a structured "
                "bullet-point list only — no additional explanation."
            )
            user_tmpl = (
                "Extract from the following text:\n"
                "1. Main topics (lecture sections)\n"
                "2. Sub-concepts under each topic\n"
                "3. Relationships between concepts (e.g., A leads to B, A is a type of B)\n\n"
                "Text:\n{content}\n\n"
                "Return ONLY a structured bullet-point list:"
            )

        chat_history = [
            self._llm.construct_prompt(
                prompt=system, role=self._llm.enums.SYSTEM.value
            )
        ]

        all_concepts: list[str] = []
        for i, batch in enumerate(batches, 1):
            try:
                result = self._llm.generate_text(
                    prompt=user_tmpl.format(content=batch),
                    chat_history=chat_history,
                    max_output_tokens=_CONCEPT_MAX_TOKENS,
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
        if self._language == "ar":
            system = (
                "أنت خبير في إنشاء مخططات Mermaid. مهمتك تحويل قوائم المفاهيم "
                "والعلاقات إلى كود Mermaid صحيح يمثل هيكل المحاضرة بصرياً. "
                "أعد كود Mermaid فقط — لا شيء آخر."
            )
            user_prompt = (
                "حوّل قائمة المفاهيم والعلاقات التالية إلى مخطط Mermaid من النوع flowchart TD.\n\n"
                "القواعد:\n"
                "- ابدأ بـ: graph TD\n"
                "- استخدم معرّفات قصيرة للعقد (A, B, C, ...) مع تسميات واضحة بين قوسين مربعين\n"
                "- إذا كانت التسمية تحتوي على مسافات أو أحرف خاصة، ضعها بين علامتي اقتباس\n"
                "- استخدم --> للعلاقات العادية\n"
                "- استخدم -->|نص| للعلاقات التي تحتاج وصفاً\n"
                "- أضف subgraph للمجموعات المنطقية (أقسام المحاضرة)\n"
                "- لا تضع أي نص قبل graph TD أو بعد آخر سطر\n\n"
                f"قائمة المفاهيم:\n{concepts_text}\n\n"
                "كود Mermaid:"
            )
        else:
            system = (
                "You are a Mermaid diagram expert. Your task is to convert concept "
                "lists and relationships into valid Mermaid code that visually "
                "represents the lecture structure. Return ONLY Mermaid code — nothing else."
            )
            user_prompt = (
                "Convert the following concept list and relationships into a Mermaid flowchart TD diagram.\n\n"
                "Rules:\n"
                "- Start with: graph TD\n"
                "- Use short node IDs (A, B, C, ...) with clear labels in square brackets\n"
                "- If a label contains spaces or special chars, wrap it in double quotes\n"
                "- Use --> for standard relationships\n"
                "- Use -->|label| for labelled relationships\n"
                "- Use subgraph for logical groups (lecture sections)\n"
                "- Do NOT put any text before 'graph TD' or after the last line\n\n"
                f"Concept list:\n{concepts_text}\n\n"
                "Mermaid code:"
            )

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
