from typing import AsyncGenerator
from agents.base import BaseAgent, AgentState
import re

class ResponseFormatterAgent(BaseAgent):
    """
    Agent that calls the LLM to produce the final natural-language answer.
    It uses the TemplateParser to dynamically load prompts based on the route
    and language, ensuring clean separation of logic and prompt text.
    """

    def __init__(self, llm_provider, template_parser) -> None:
        self._llm = llm_provider
        self._template_parser = template_parser

    @property
    def agent_name(self) -> str:
        return "ResponseFormatterAgent"

    async def execute(self, state: AgentState) -> AgentState:
        self.validate_state(state, ["query", "metadata"])

        route = state["metadata"].get("route", "")

        # If it's small talk, SmallTalkAgent already generated the response.
        if route == "small_talk":
            state["final_response"] = state.get("reasoning_context", "Hello!")
            state["agent_trace"].append(f"{self.agent_name}: formatted small_talk response")
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
        route = state["metadata"].get("route", "")

        if route == "small_talk":
            yield state.get("reasoning_context", "Hello!")
            state["agent_trace"].append(f"{self.agent_name}: streamed small_talk response")
            return

        reasoning_context: str = state.get("reasoning_context", "").strip()
        if not reasoning_context:
            yield "No retrieval executed"
            state["agent_trace"].append(f"{self.agent_name}: streamed empty retrieval response")
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
        
        query_lower = query.lower().strip()
        route = state.get("metadata", {}).get("route", "")

        # 1. اكتشاف اللغة وتوجيه الـ Parser أوتوماتيكياً
        is_arabic = any('\u0600' <= char <= '\u06FF' for char in query) or "arabic" in query_lower or "عربي" in query_lower
        self._template_parser.set_language("ar" if is_arabic else "en")

        # 2. تحديد العملية
        is_summary = route == "summary" or "summarize" in query_lower or "ملخص" in query_lower
        is_quiz = route == "quiz" or "quiz" in query_lower or "امتحان" in query_lower

        q_match = re.search(r'\[(\d+)\]', query)
        num_q = int(q_match.group(1)) if q_match else 5

        # 3. سحب القوالب باستخدام الـ Template Parser النظيف
        if is_summary:
            system_prompt = self._template_parser.get("rag", "summarize_system_prompt")
            footer_prompt = self._template_parser.get("rag", "summarize_footer_prompt")
        elif is_quiz:
            system_prompt = self._template_parser.get("rag", "quiz_system_prompt", {"num_questions": num_q})
            footer_prompt = self._template_parser.get("rag", "quiz_footer_prompt", {"num_questions": num_q})
        else:
            system_prompt = self._template_parser.get("rag", "system_prompt")
            footer_prompt = self._template_parser.get("rag", "footer_prompt", {"query": query})

        # 4. بناء الـ Prompt النهائي
        if reasoning_context:
            full_prompt = f"## Reference Material:\n{reasoning_context}\n\n{footer_prompt}"
        else:
            full_prompt = f"## Reference Material:\nNo documents were found.\n\n{footer_prompt}"

        chat_history = [
            self._llm.construct_prompt(
                prompt=system_prompt,
                role=self._llm.enums.SYSTEM.value,
            )
        ]

        return full_prompt, chat_history