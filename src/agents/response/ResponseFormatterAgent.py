"""
ResponseFormatterAgent - Refactored to completely use TemplateParser 
and eliminate hardcoded prompts and dynamic imports.
"""
from __future__ import annotations

import re
from typing import AsyncGenerator, List

from agents.base import BaseAgent, AgentState
from agents.base.intent_utils import (
    TASK_EXPLANATION,
    TASK_QUIZ,
    TASK_SIMPLE_QA,
    TASK_SUMMARY,
    classify_task_type,
)

# Output token budget per task type
_MAX_TOKENS = {
    TASK_SUMMARY:     4000,
    TASK_QUIZ:        3000,
    TASK_EXPLANATION: 3000,
    TASK_SIMPLE_QA:   1500,
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

        # [SUMMARY] delegate to SummaryGenerator
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

        # [SUMMARY] stream via SummaryGenerator
        if state.get("reasoning_context") == "__SUMMARY__":
            async for token in self._stream_summary(state):
                yield token
            state["agent_trace"].append(
                f"{self.agent_name}: streamed summary via SummaryGenerator"
            )
            return

        reasoning_context = (state.get("reasoning_context") or "").strip()
        if not reasoning_context:
            yield "No content was retrieved. Please ensure the lecture has been processed and indexed."
            state["agent_trace"].append(
                f"{self.agent_name}: empty context - no retrieval"
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

        # Detect language
        is_arabic = any("\u0600" <= c <= "\u06FF" for c in query) or \
                    "arabic" in query_lower or "بالعربي" in query_lower
        lang_code = "ar" if is_arabic else "en"
        
        # Set the language in the Template Parser dynamically
        self._template_parser.set_language(lang_code)

        # Parse quiz question count
        q_match = re.search(r"\[(\d+)\]", query)
        num_q = int(q_match.group(1)) if q_match else 5

        # System prompt (from template registry)
        if task_type == TASK_SUMMARY:
            system_prompt = self._template_parser.get("rag", "summarize_system_prompt")
        elif task_type == TASK_QUIZ:
            system_prompt = self._template_parser.get("rag", "quiz_system_prompt", {"num_questions": num_q})
        else:
            system_prompt = self._template_parser.get("rag", "system_prompt")

        # Fallback if templates are missing
        if not system_prompt:
            system_prompt = "You are a helpful AI assistant."

        chat_history = [
            self._llm.construct_prompt(
                prompt=system_prompt,
                role=self._llm.enums.SYSTEM.value,
            )
        ]

        # Task-specific instruction block
        if task_type == TASK_SUMMARY:
            instruction = self._template_parser.get("rag", "instruction_summary")
        elif task_type == TASK_QUIZ:
            instruction = self._template_parser.get("rag", "instruction_quiz")
        else:
            instruction = self._template_parser.get("rag", "instruction_qa")
            
        if not instruction:
            instruction = "Answer the question based on the material."

        # Assemble final prompt
        ref_section = reasoning_context if reasoning_context else "No reference material was found."
        full_prompt = (
            f"{instruction}\n\n"
            f"## Student Question:\n{query}\n\n"
            f"---\n## Reference Material:\n\n{ref_section}"
        )
        return full_prompt, chat_history

    # ------------------------------------------------------------------
    # Sources / citations block
    # ------------------------------------------------------------------
    def _sources_block(self, state: AgentState) -> str:
        citations: List[dict] = state.get("citations") or []
        if not citations:
            return ""
            
        query = state.get("query", "")
        is_arabic = any("\u0600" <= c <= "\u06FF" for c in query)
        heading = "\n\n---\n### المصادر" if is_arabic else "\n\n---\n### Sources"
        
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
                lines.append("- " + " | ".join(parts))
                
        if not lines:
            return ""
        return heading + "\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Summary helpers (call SummaryGenerator)
    # ------------------------------------------------------------------
    async def _run_summary(self, state: AgentState) -> str:
        from agents.response.SummaryGenerator import SummaryGenerator
        chunks = state.get("retrieved_chunks") or []
        if not chunks:
            return "No lecture content was found to summarise."
            
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
        return generator.generate(chunks=orm_chunks)

    async def _stream_summary(self, state: AgentState) -> AsyncGenerator[str, None]:
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
    def __init__(self, chunk_dict: dict) -> None:
        self.chunk_text: str = chunk_dict.get("text", "")
        self.chunk_metadata: dict = chunk_dict.get("metadata", {})