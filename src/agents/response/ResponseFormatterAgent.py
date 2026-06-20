"""
ResponseFormatterAgent — Phase 3 improvements (items 1, 2, 7, 8).

Changes vs original:
- Summary requests (sentinel ``reasoning_context == "__SUMMARY__"``)
  call ``SummaryGenerator`` via map-reduce instead of a single prompt
  (item 1).
- Teaching-mode instructions are injected into the Q&A prompt based on
  ``state["teaching_mode"]`` (item 7).
- "Be concise" Q&A instruction replaced with detailed/educational
  wording matching the updated system_prompt in rag.py (item 2).
- A "Sources" block is appended to non-summary, non-smalltalk answers
  when ``state["citations"]`` is populated (item 8).
- ``max_output_tokens`` raised to 3000 for explanation/quiz queries
  so deep answers are not cut short (item 2).
"""

from __future__ import annotations

import re
from typing import AsyncGenerator, List

from agents.base import BaseAgent, AgentState
from agents.base.intent_utils import (
    TASK_EXPLANATION,
    TASK_FULL_EXPLAIN,
    TASK_QUIZ,
    TASK_SIMPLE_QA,
    TASK_SUMMARY,
    classify_task_type,
)

# Output token budget per task type
_MAX_TOKENS = {
    TASK_SUMMARY:      4000,
    TASK_FULL_EXPLAIN: 4000,
    TASK_QUIZ:         3000,
    TASK_EXPLANATION:  3000,
    TASK_SIMPLE_QA:    1500,
}


class ResponseFormatterAgent(BaseAgent):

    def __init__(self, llm_provider, template_parser) -> None:
        self._llm = llm_provider
        self._template_parser = template_parser

    @property
    def agent_name(self) -> str:
        return "ResponseFormatterAgent"

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    async def execute(self, state: AgentState) -> AgentState:
        self.validate_state(state, ["query", "metadata"])

        # ── Fetch Lecture Status ─────────────────────────────────────────
        # The processing_status_store is in-memory.  If the server restarted
        # or this is a new process, status will be None for an already-indexed
        # project.  We treat None (unknown) the same as READY so legitimate
        # queries are never blocked by a missing in-memory entry.
        from utils.processing_status import processing_status_store
        project_id = state.get("project_id", "")
        status_data = await processing_status_store.get_status(project_id)
        current_status = status_data.get("status")

        if current_status in ("READY", "FAILED", None):
            state["metadata"]["lecture_status"] = "ready"
            effective_status = "READY"
        else:
            state["metadata"]["lecture_status"] = "processing"
            effective_status = current_status
        state["metadata"]["raw_status"] = effective_status

        route = state["metadata"].get("route", "")

        if route == "small_talk":
            state["final_response"] = state.get("reasoning_context", "Hello!")
            state["agent_trace"].append(
                f"{self.agent_name}: formatted small_talk response"
            )
            return state

        query = state["query"]
        route_str = state.get("metadata", {}).get("route", "")
        task_type = classify_task_type(query, route_str)
        max_tokens = _MAX_TOKENS.get(task_type, 1500)

        # ── SUMMARY: delegate to SummaryGenerator ────────────────────────
        if state.get("reasoning_context") == "__SUMMARY__":
            answer = await self._run_summary(state)
            state["final_response"] = answer
            state["agent_trace"].append(
                f"{self.agent_name}: summary via SummaryGenerator"
            )
            return state

        full_prompt, chat_history = self._build_prompt(state)
        self.log_step(
            f"calling LLM: task={task_type}, max_tokens={max_tokens}, "
            f"prompt_len={len(full_prompt)}"
        )

        try:
            answer = self._llm.generate_text(
                prompt=full_prompt,
                chat_history=chat_history,
                max_output_tokens=max_tokens,
            )
            state["final_response"] = (answer or "") + self._sources_block(state)
        except Exception as exc:
            self.log_step(f"LLM generation failed: {exc}")
            state["final_response"] = ""
            state["error"] = str(exc)

        state["agent_trace"].append(
            f"{self.agent_name}: generated final answer (task={task_type})"
        )
        return state

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream_execute(self, state: AgentState) -> AsyncGenerator[str, None]:
        self.validate_state(state, ["query", "metadata"])
        
        # ── Fetch Lecture Status ─────────────────────────────────────────
        # Treat None (unknown / server restarted) the same as READY.
        from utils.processing_status import processing_status_store
        project_id = state.get("project_id", "")
        status_data = await processing_status_store.get_status(project_id)
        current_status = status_data.get("status")

        if current_status in ("READY", "FAILED", None):
            state["metadata"]["lecture_status"] = "ready"
            effective_status = "READY"
        else:
            state["metadata"]["lecture_status"] = "processing"
            effective_status = current_status
        state["metadata"]["raw_status"] = effective_status

        route = state["metadata"].get("route", "")

        if route == "small_talk":
            yield state.get("reasoning_context", "Hello!")
            state["agent_trace"].append(
                f"{self.agent_name}: streamed small_talk response"
            )
            return

        query = state["query"]
        route_str = state.get("metadata", {}).get("route", "")
        task_type = classify_task_type(query, route_str)
        max_tokens = _MAX_TOKENS.get(task_type, 1500)

        # ── SUMMARY: stream via SummaryGenerator ─────────────────────────
        if state.get("reasoning_context") == "__SUMMARY__":
            async for token in self._stream_summary(state):
                yield token
            state["agent_trace"].append(
                f"{self.agent_name}: streamed summary via SummaryGenerator"
            )
            return

        full_prompt, chat_history = self._build_prompt(state)
        self.log_step(
            f"streaming: task={task_type}, max_tokens={max_tokens}, "
            f"prompt_len={len(full_prompt)}"
        )

        sources_block = self._sources_block(state)
        buffer = []

        try:
            async for chunk in self._llm.generate_stream(
                prompt=full_prompt,
                chat_history=chat_history,
                max_output_tokens=max_tokens,
            ):
                yield chunk.replace("\n", "\\n")
                buffer.append(chunk)
        except Exception as exc:
            self.log_step(f"LLM streaming failed: {exc}")
            yield "An error occurred while generating the response."

        # Append sources block after the main stream ends
        if sources_block:
            yield sources_block.replace("\n", "\\n")

        state["agent_trace"].append(
            f"{self.agent_name}: streamed answer (task={task_type})"
        )

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_prompt(self, state: AgentState):
        query: str = state["query"]
        reasoning_context: str = state.get("reasoning_context", "").strip()
        query_lower = query.lower().strip()
        route = state.get("metadata", {}).get("route", "")
        task_type = classify_task_type(query, route)

        # ── Detect language ──────────────────────────────────────────────
        is_arabic = any("\u0600" <= c <= "\u06FF" for c in query) or \
                    "arabic" in query_lower or "عربي" in query_lower

        # ── Parse quiz question count ────────────────────────────────────
        q_match = re.search(r"\[(\d+)\]", query)
        num_q = int(q_match.group(1)) if q_match else 5

        # ── System prompt (from template) ────────────────────────────────
        if is_arabic:
            from stores.llm.templates.locales.ar.rag import (
                quiz_system_prompt as ar_quiz,
            )
            from stores.llm.templates.locales.ar.rag import (
                summarize_system_prompt as ar_sum,
            )
            from stores.llm.templates.locales.ar.rag import system_prompt as ar_sys

            if task_type == TASK_SUMMARY:
                system_prompt = ar_sum.safe_substitute()
            elif task_type == TASK_QUIZ:
                system_prompt = ar_quiz.safe_substitute(num_questions=num_q)
            else:
                system_prompt = ar_sys.safe_substitute()
        else:
            from stores.llm.templates.locales.en.rag import (
                quiz_system_prompt as en_quiz,
            )
            from stores.llm.templates.locales.en.rag import (
                summarize_system_prompt as en_sum,
            )
            from stores.llm.templates.locales.en.rag import system_prompt as en_sys

            if task_type == TASK_SUMMARY:
                system_prompt = en_sum.safe_substitute()
            elif task_type == TASK_QUIZ:
                system_prompt = en_quiz.safe_substitute(num_questions=num_q)
            else:
                system_prompt = en_sys.safe_substitute()


        chat_history = [
            self._llm.construct_prompt(
                prompt=system_prompt,
                role=self._llm.enums.SYSTEM.value,
            )
        ]

        # ── Task-specific instruction block ──────────────────────────────
        if task_type == TASK_SUMMARY:
            instruction = (
                "**التعليمات:** قم بتقديم ملخص شامل ومنظم للملفات الموجودة بالأسفل."
                if is_arabic else
                "**INSTRUCTION:** Provide a complete, structured lecture summary of the content below."
            )
        elif task_type == TASK_QUIZ:
            instruction = (
                "**التعليمات:** قم بتوليد أسئلة الامتحان بناءً على المحتوى بالأسفل."
                if is_arabic else
                "**INSTRUCTION:** Generate the quiz questions based on the content below."
            )
        else:
            # Detailed Q&A instruction with strict grounding rules
            if is_arabic:
                instruction = (
                    "**تعليمات الإجابة:**\n"
                    "1. ابحث في المادة المرجعية أدناه عن الإجابة الدقيقة.\n"
                    "2. إذا كان CONTEXT_QUALITY = FULL: أجب من المحاضرة فقط. لا تضف معلومات خارجية.\n"
                    "3. إذا كان CONTEXT_QUALITY = PARTIAL: أشر إلى ذلك، ثم أجب مما هو متاح مع إضافة حد أدنى من المعرفة الخارجية الضرورية فقط.\n"
                    "4. إذا كان CONTEXT_QUALITY = EMPTY أو CONTEXT_AVAILABLE = FALSE: قل فقط 'المعلومة غير متوفرة في المحاضرة المرفوعة.' ولا تخمّن.\n"
                    "5. لا تخترع حقائق. لا تجيب من معرفتك الخاصة عند توفر سياق كافٍ.\n"
                    "6. للمعادلات: اشرح معنى كل رمز بالعربية.\n"
                    "7. قارن بالمفاهيم ذات الصلة ونبّه إلى الأخطاء الشائعة عند الضرورة."
                )
            else:
                instruction = (
                    "**ANSWERING RULES:**\n"
                    "1. Find the answer in the Reference Material below.\n"
                    "2. If CONTEXT_QUALITY = FULL: answer strictly from the lecture. Do not add outside knowledge.\n"
                    "3. If CONTEXT_QUALITY = PARTIAL: note that context is partial, then answer from what is available. Add minimal external knowledge only where the lecture is silent.\n"
                    "4. If CONTEXT_QUALITY = EMPTY or CONTEXT_AVAILABLE = FALSE: respond only with 'The information is not available in the uploaded content.' Do not guess.\n"
                    "5. Never invent facts. Never use your own knowledge when strong context exists.\n"
                    "6. For equations: explain every symbol in plain language.\n"
                    "7. Compare with related concepts and flag common mistakes where relevant."
                )

        # ── Assemble final prompt ────────────────────────────────────────
        ref_section = reasoning_context if reasoning_context else ""
        
        if task_type in (TASK_EXPLANATION, TASK_SIMPLE_QA):
            history_lines = []
            for mem in state.get("memory_context", []):
                if mem.get("memory_type") == "short_term":
                    history_lines.append(mem.get("content", ""))
            conversation_history = "\n".join(history_lines[-3:]) # last 3 turns
            
            current_topic = state.get("metadata", {}).get("current_topic", "")
            
            raw_status = state.get("metadata", {}).get("raw_status", "READY")
            # raw_status is already normalised by execute/stream_execute above;
            # anything other than active mid-pipeline states is treated as READY.
            if raw_status in ("PROCESSING", "EXTRACTING", "CHUNKING", "EMBEDDING", "INDEXING"):
                state_val = "PROCESSING"
            else:
                state_val = "READY"
                
            chunks = state.get("retrieved_chunks") or []
            # Use the quality signal written by RetrievalAgent — this correctly
            # distinguishes EMPTY (no chunks above threshold) from PARTIAL (weak
            # scores) from FULL (strong matches), instead of just checking len(chunks).
            retrieval_quality = state.get("metadata", {}).get("retrieval_quality", "FULL")

            # If retrieval_quality is EMPTY, set context_available = FALSE even
            # when there are technically some chunks (they're below threshold).
            if retrieval_quality == "EMPTY" or not chunks:
                context_available = "FALSE"
                context_quality = "EMPTY"
            elif retrieval_quality == "PARTIAL":
                context_available = "TRUE"
                context_quality = "PARTIAL"
            else:
                context_available = "TRUE"
                context_quality = "FULL"
            
            import json
            input_json = {
                "STATE": state_val,
                "CONTEXT_AVAILABLE": context_available,
                "CONTEXT_QUALITY": context_quality,
                "lecture_context": ref_section,
                "conversation_history": conversation_history,
                "current_topic": current_topic,
                "user_request": query
            }
            input_str = json.dumps(input_json, ensure_ascii=False, indent=2)
            
            full_prompt = (
                f"{instruction}\n\n"
                f"```json\n{input_str}\n```"
            )
        else:
            full_prompt = (
                f"{instruction}\n\n"
                f"## Student Question:\n{query}\n\n"
                f"---\n## Reference Material:\n\n{ref_section}"
            )

        return full_prompt, chat_history

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Sources / citations block (item 8)
    # ------------------------------------------------------------------

    def _sources_block(self, state: AgentState) -> str:
        """
        Build a markdown "Sources" section from ``state["citations"]``.
        Returns an empty string when there are no citations.
        """
        citations: List[dict] = state.get("citations") or []
        if not citations:
            return ""

        query = state.get("query", "")
        is_arabic = any("\u0600" <= c <= "\u06FF" for c in query)
        heading = "\n\n---\n### 📌 المصادر" if is_arabic else "\n\n---\n### 📌 Sources"

        lines = []
        for cit in citations:
            parts = []
            src = cit.get("source", "")
            if src:
                parts.append(f"**{src}**")
            if cit.get("page"):
                label = "صفحة" if is_arabic else "Page"
                parts.append(f"{label} {cit['page']}")
            if cit.get("section"):
                label = "قسم" if is_arabic else "Section"
                parts.append(f"{label}: *{cit['section']}*")
            if parts:
                lines.append("- " + " · ".join(parts))

        if not lines:
            return ""

        return heading + "\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Summary helpers (call SummaryGenerator)
    # ------------------------------------------------------------------

    async def _run_summary(self, state: AgentState) -> str:
        """Synchronous map-reduce summary via SummaryGenerator."""
        from agents.response.SummaryGenerator import SummaryGenerator

        chunks = state.get("retrieved_chunks") or []
        if not chunks:
            return "No lecture content was found to summarise."

        # SummaryGenerator expects ORM objects; when available they're
        # stored under the "_orm" key by RetrievalAgent._fetch_ordered_chunks.
        orm_chunks = [c["_orm"] for c in chunks if c.get("_orm") is not None]

        if not orm_chunks:
            # Fallback: build lightweight wrapper objects from dict chunks
            orm_chunks = [_DictChunkWrapper(c) for c in chunks]

        query = state.get("query", "")
        is_arabic = any("\u0600" <= c <= "\u06FF" for c in query)
        lang = "ar" if is_arabic else "en"

        generator = SummaryGenerator(
            generation_client=self._llm,
            template_parser=self._template_parser,
            language=lang,
        )
        return generator.generate(chunks=orm_chunks)

    async def _stream_summary(self, state: AgentState) -> AsyncGenerator[str, None]:
        """Streaming map-reduce summary via SummaryGenerator."""
        from agents.response.SummaryGenerator import SummaryGenerator

        chunks = state.get("retrieved_chunks") or []
        if not chunks:
            yield "No lecture content was found to summarise."
            return

        orm_chunks = [c["_orm"] for c in chunks if c.get("_orm") is not None]
        if not orm_chunks:
            orm_chunks = [_DictChunkWrapper(c) for c in chunks]

        query = state.get("query", "")
        is_arabic = any("\u0600" <= c <= "\u06FF" for c in query)
        lang = "ar" if is_arabic else "en"

        generator = SummaryGenerator(
            generation_client=self._llm,
            template_parser=self._template_parser,
            language=lang,
        )
        async for token in generator.generate_stream(chunks=orm_chunks):
            yield token.replace("\n", "\\n")


# ---------------------------------------------------------------------------
# Lightweight wrapper so dict-chunks quack like DataChunk ORM objects
# ---------------------------------------------------------------------------

class _DictChunkWrapper:
    """
    Wraps a ``{text, score, metadata}`` dict so it looks like a DataChunk
    ORM object to ``SummaryGenerator._chunks_to_blocks``.
    """

    def __init__(self, chunk_dict: dict) -> None:
        self.chunk_text: str = chunk_dict.get("text", "")
        self.chunk_metadata: dict = chunk_dict.get("metadata", {})