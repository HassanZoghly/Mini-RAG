from typing import AsyncGenerator
from agents.base import BaseAgent, AgentState
import re

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
        import re

        query: str = state["query"]
        reasoning_context: str = state.get("reasoning_context", "").strip()

        query_lower = query.lower().strip()
        route = state.get("metadata", {}).get("route", "")

        # 1. اكتشاف اللغة
        is_arabic = any('\u0600' <= char <= '\u06FF' for char in query) or "arabic" in query_lower or "عربي" in query_lower

        # 2. تحديد العملية
        is_summary = route == "summary" or "summarize" in query_lower or "ملخص" in query_lower
        is_quiz = route == "quiz" or "quiz" in query_lower or "امتحان" in query_lower

        q_match = re.search(r'\[(\d+)\]', query)
        num_q = int(q_match.group(1)) if q_match else 5

        # 3. سحب القوالب
        if is_arabic:
            from stores.llm.templates.locales.ar.rag import summarize_system_prompt as ar_sum
            from stores.llm.templates.locales.ar.rag import quiz_system_prompt as ar_quiz
            from stores.llm.templates.locales.ar.rag import system_prompt as ar_sys

            if is_summary:
                system_prompt = ar_sum.safe_substitute()
            elif is_quiz:
                system_prompt = ar_quiz.safe_substitute(num_questions=num_q)
            else:
                system_prompt = ar_sys.safe_substitute()
        else:
            from stores.llm.templates.locales.en.rag import summarize_system_prompt as en_sum
            from stores.llm.templates.locales.en.rag import quiz_system_prompt as en_quiz
            from stores.llm.templates.locales.en.rag import system_prompt as en_sys

            if is_summary:
                system_prompt = en_sum.safe_substitute()
            elif is_quiz:
                system_prompt = en_quiz.safe_substitute(num_questions=num_q)
            else:
                system_prompt = en_sys.safe_substitute()

        chat_history = [
            self._llm.construct_prompt(
                prompt=system_prompt,
                role=self._llm.enums.SYSTEM.value,
            )
        ]

        # 4. 🔥 الحقنة الصارمة لمنع التلخيص العشوائي نهائياً
        if is_summary:
            instruction = (
                "**التعليمات:** قم بتقديم ملخص شامل ومنظم للملفات الموجودة بالأسفل." if is_arabic else "**INSTRUCTION:** Provide a structured summary of the files below."
            )
        elif is_quiz:
            instruction = (
                "**التعليمات:** قم بتوليد أسئلة الامتحان بناءً على المحتوى بالأسفل." if is_arabic else "**INSTRUCTION:** Generate the quiz based on the content below."
            )
        else:
            instruction = (
                "**تعليمات الإجابة (Q&A):**\n"
                "1. ابحث في النصوص المرفقة بالأسفل (Reference Material) عن الإجابة الدقيقة لسؤال الطالب.\n"
                "2. أجب باختصار وفي صلب الموضوع. ممنوع تماماً تلخيص المادة.\n"
                "3. ⚠️ **تجاهل الإخفاقات السابقة:** قيم هذا السؤال بشكل مستقل تماماً.\n"
                "4. إذا لم تجد الإجابة نهائياً في النص المرفق، قل 'المعلومة غير متوفرة في المحاضرة'."
                if is_arabic else
                "**Q&A INSTRUCTIONS:**\n"
                "1. Extract the exact answer from the Reference Material below.\n"
                "2. Be concise. Do NOT summarize.\n"
                "3. ⚠️ **IGNORE PAST FAILURES:** Evaluate this query independently.\n"
                "4. If the answer is truly missing, say 'This information is not present in the provided lectures.'"
            )

        # 🎯 الهندسة العكسية: التعليمات والسؤال في القمة، والمحاضرات في القاع
        if reasoning_context:
            full_prompt = (
                f"{instruction}\n\n"
                f"## Student Question:\n{query}\n\n"
                f"---\n## Reference Material:\n\n{reasoning_context}"
            )
        else:
            full_prompt = (
                f"{instruction}\n\n"
                f"## Student Question:\n{query}\n\n"
                f"---\n## Reference Material:\nNo documents were found."
            )

        return full_prompt, chat_history
