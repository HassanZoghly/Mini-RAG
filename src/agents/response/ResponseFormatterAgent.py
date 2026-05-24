from typing import AsyncGenerator
from agents.base import BaseAgent, AgentState

class ResponseFormatterAgent(BaseAgent):
    """
    Agent that calls the LLM to produce the final natural-language answer.
    Also handles skipping RAG formatting if the route is small_talk.
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

        # If it's small talk, SmallTalkAgent already generated the response into reasoning_context.
        # We can just use it directly without re-prompting, or just pass it to final_response.
        if route == "small_talk":
            state["final_response"] = state.get("reasoning_context", "Hello!")
            state["agent_trace"].append(f"{self.agent_name}: formatted small_talk response")
            return state

        # Otherwise, standard RAG response generation
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
                # التعديل هنا: تحويل المسافات لـ Escaped String عشان متتمسحش في الـ Network
                yield chunk.replace("\n", "\\n")
        except Exception as exc:
            self.log_step(f"LLM streaming failed: {exc}")
            yield "An error occurred while generating the response."

        state["agent_trace"].append(f"{self.agent_name}: generated final answer (streaming)")

    def _build_prompt(self, state: AgentState):
        query: str = state["query"]
        reasoning_context: str = state.get("reasoning_context", "").strip()

        # التعديل هنا: تحديد الـ Template بناءً على الـ Route أو الكلمة المفتاحية
        query_lower = query.lower().strip()
        route = state.get("metadata", {}).get("route", "")

        if route == "summary" or "summarize" in query_lower:
            system_prompt = self._template_parser.get("rag", "summarize_system_prompt")
        elif route == "quiz" or "quiz" in query_lower:
            # تمرير المتغيرات المطلوبة للـ Quiz
            system_prompt = self._template_parser.get("rag", "quiz_system_prompt", num_questions=5)
        else:
            system_prompt = self._template_parser.get("rag", "system_prompt")

        chat_history = [
            self._llm.construct_prompt(
                prompt=system_prompt,
                role=self._llm.enums.SYSTEM.value,
            )
        ]

        if reasoning_context:
            full_prompt = (
                f"## Reference Material\n\n"
                f"{reasoning_context}\n\n"
                f"---\n\n"
                f"## Student Question\n\n"
                f"{query}\n\n"
                f"## Your Answer\n\n"
                f"Respond using Markdown formatting."
            )
        else:
            full_prompt = (
                f"## Student Question\n\n"
                f"{query}\n\n"
                f"## Your Answer\n\n"
                f"No documents were found. Respond using Markdown formatting."
            )

        return full_prompt, chat_history
