from typing import AsyncGenerator
from agents.base import BaseAgent, AgentState
import re

class ResponseFormatterAgent(BaseAgent):
    """
    Agent that calls the LLM to produce the final natural-language answer.
    It constructs the prompt dynamically based on metadata (route, mode, context classification).
    """

    def __init__(self, llm_provider, template_parser) -> None:
        self._llm = llm_provider
        self._template_parser = template_parser

    @property
    def agent_name(self) -> str:
        return "ResponseFormatterAgent"

    async def execute(self, state: AgentState) -> AgentState:
        self.validate_state(state, ["query", "metadata"])

        metadata = state["metadata"]
        route = metadata.get("route", "")
        has_context = metadata.get("has_context", True)

        if route in ("smalltalk", "small_talk"):
            state["final_response"] = state.get("reasoning_context", "Hello!")
            state["agent_trace"].append(f"{self.agent_name}: formatted small_talk response")
            return state

        if not has_context and route == "retrieval":
            query = state["query"]
            query_lower = query.lower().strip()
            is_arabic = any('\u0600' <= char <= '\u06FF' for char in query) or "arabic" in query_lower or "عربي" in query_lower
            if is_arabic:
                state["final_response"] = "عذراً، لم أتمكن من العثور على هذه المعلومة في المستندات المرفوعة."
            else:
                state["final_response"] = "This information is not available in the uploaded documents."
            state["agent_trace"].append(f"{self.agent_name}: skipped generation due to missing context")
            return state

        full_prompt, chat_history = self._build_prompt(state)

        self.log_step(f"calling LLM for query='{state['query'][:60]}' ({len(full_prompt)} chars)")

        try:
            answer = self._llm.generate_text(
                prompt=full_prompt,
                chat_history=chat_history,
            )
            state["final_response"] = answer or ""
        except Exception as exc:
            self.log_step(f"LLM generation failed: {exc}")
            state["final_response"] = ""
            state["error"] = str(exc)

        state["agent_trace"].append(f"{self.agent_name}: generated final formatted answer")
        return state

    async def stream_execute(self, state: AgentState) -> AsyncGenerator[str, None]:
        self.validate_state(state, ["query", "metadata"])
        
        metadata = state["metadata"]
        route = metadata.get("route", "")
        has_context = metadata.get("has_context", True)

        if route in ("smalltalk", "small_talk"):
            yield state.get("reasoning_context", "Hello!")
            state["agent_trace"].append(f"{self.agent_name}: streamed small_talk response")
            return

        if not has_context and route == "retrieval":
            query = state["query"]
            query_lower = query.lower().strip()
            is_arabic = any('\u0600' <= char <= '\u06FF' for char in query) or "arabic" in query_lower or "عربي" in query_lower
            if is_arabic:
                yield "عذراً، لم أتمكن من العثور على هذه المعلومة في المستندات المرفوعة."
            else:
                yield "This information is not available in the uploaded documents."
            state["agent_trace"].append(f"{self.agent_name}: skipped generation due to missing context")
            return

        full_prompt, chat_history = self._build_prompt(state)

        self.log_step(f"streaming LLM response for query='{state['query'][:60]}'")

        try:
            async for chunk in self._llm.generate_stream(
                prompt=full_prompt,
                chat_history=chat_history,
            ):
                yield chunk.replace("\n", "\\n")
        except Exception as exc:
            self.log_step(f"LLM streaming failed: {exc}")
            yield "An error occurred while generating the response."

        state["agent_trace"].append(f"{self.agent_name}: generated final answer (streaming)")

    def _build_prompt(self, state: AgentState):
        query: str = state["query"]
        reasoning_context: str = state.get("reasoning_context", "").strip()
        metadata = state.get("metadata", {})
        
        query_lower = query.lower().strip()
        route = metadata.get("route", "retrieval")
        mode = metadata.get("mode", "qa")
        is_partial = metadata.get("is_partial", False)

        # 1. اكتشاف اللغة وتوجيه الـ Parser أوتوماتيكياً
        is_arabic = any('\u0600' <= char <= '\u06FF' for char in query) or "arabic" in query_lower or "عربي" in query_lower
        self._template_parser.set_language("ar" if is_arabic else "en")

        # 2. بناء System Prompt
        system_prompt = self._template_parser.get("rag", "system_prompt")
        
        if route == "retrieval":
            if is_partial:
                rule_prompt = self._template_parser.get("rag", "partial_context_rule")
            else:
                rule_prompt = self._template_parser.get("rag", "strict_context_rule")
            system_prompt = f"{system_prompt}\n\n{rule_prompt}"
            
        if mode == "summary":
            mode_prompt = self._template_parser.get("rag", "mode_summary")
        elif mode == "explain":
            mode_prompt = self._template_parser.get("rag", "mode_explain")
        else:
            mode_prompt = self._template_parser.get("rag", "mode_qa")
            
        system_prompt = f"{system_prompt}\n\n{mode_prompt}"

        footer_prompt = self._template_parser.get("rag", "footer_prompt", {"query": query})

        if reasoning_context and route == "retrieval":
            full_prompt = f"## Reference Material:\n{reasoning_context}\n\n{footer_prompt}"
        else:
            full_prompt = footer_prompt

        chat_history = [
            self._llm.construct_prompt(
                prompt=system_prompt,
                role=self._llm.enums.SYSTEM.value,
            )
        ]

        return full_prompt, chat_history