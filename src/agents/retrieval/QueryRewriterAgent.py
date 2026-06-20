from agents.base import BaseAgent, AgentState
from agents.base.intent_utils import classify_task_type, TASK_SUMMARY, TASK_QUIZ
from typing import List


# Words/phrases (English + Arabic) that suggest a follow-up question relies
# on earlier conversation context ("explain the previous part", "what about
# that?", "اشرح الجزء السابق", "كمان", "كمل").
_REFERENTIAL_TERMS: frozenset = frozenset({
    "this", "that", "these", "those", "it", "they",
    "above", "previous", "before", "earlier", "last",
    "more", "continue", "again", "further", "same",
    "ده", "دي", "ذلك", "هذا", "هذه", "السابق", "السابقة",
    "كمان", "زيادة", "كمل", "أكتر", "اكتر", "وبعدين", "تاني",
})

# Queries this short are often fragments ("example?", "and the formula?")
# that only make sense with the preceding conversation.
_SHORT_QUERY_WORD_THRESHOLD = 4


class QueryRewriterAgent(BaseAgent):
    """
    Rewrites a (possibly vague, context-dependent) student question into a
    self-contained search query before retrieval — item 4A.

    Example
    -------
    Conversation so far: "Q: Explain PCA. A: PCA reduces dimensionality by ..."
    New question: "explain the previous part again"
    Rewritten query: "Explain how PCA reduces dimensionality (intuition,
    steps, and mathematical idea)"

    The rewrite only runs when:
    - the route is NOT summary/quiz (those are intentionally broad), AND
    - there is recent ``memory_context`` to draw on, AND
    - the question *looks* like it depends on earlier context (short or
      contains referential language).

    Failures fall back silently to the original ``query`` —
    ``state["query_for_retrieval"]`` is ALWAYS set by this agent so
    downstream agents never need to special-case its absence.
    """

    def __init__(self, llm_provider) -> None:
        self._llm = llm_provider

    @property
    def agent_name(self) -> str:
        return "QueryRewriterAgent"

    async def execute(self, state: AgentState) -> AgentState:
        self.validate_state(state, ["query"])

        query: str = state["query"]
        memory_context: List[dict] = state.get("memory_context") or []
        route = state.get("metadata", {}).get("route", "")
        task_type = classify_task_type(query, route)

        # Summaries/quizzes are intentionally broad — rewriting could
        # narrow them in unhelpful ways, so leave them untouched.
        if task_type in (TASK_SUMMARY, TASK_QUIZ) or not memory_context:
            state["query_for_retrieval"] = query
            state["agent_trace"].append(
                f"{self.agent_name}: skipped (task_type={task_type}, "
                f"memory_entries={len(memory_context)})"
            )
            return state

        if not self._looks_context_dependent(query):
            state["query_for_retrieval"] = query
            state["agent_trace"].append(
                f"{self.agent_name}: query looks self-contained, no rewrite"
            )
            return state

        history_text = "\n".join(
            mem.get("content", "").strip()
            for mem in memory_context[:3]
            if mem.get("content", "").strip()
        )

        if not history_text:
            state["query_for_retrieval"] = query
            state["agent_trace"].append(f"{self.agent_name}: no usable history, no rewrite")
            return state

        system_prompt = (
            "You rewrite a student's follow-up question into a standalone "
            "search query for a lecture-retrieval system, using the recent "
            "conversation as context.\n"
            "- Keep the SAME language as the student's new question.\n"
            "- Resolve references like 'this', 'that', 'the previous part' "
            "into the actual topic/concept being discussed.\n"
            "- Return ONLY the rewritten query text — no quotes, labels, "
            "or explanation.\n"
            "- If the question is already self-contained, return it "
            "unchanged."
        )

        prompt = (
            f"Recent conversation:\n{history_text}\n\n"
            f"Student's new question: {query}\n\n"
            "Rewritten standalone search query:"
        )

        try:
            rewritten = self._llm.generate_text(
                prompt=prompt,
                chat_history=[
                    self._llm.construct_prompt(
                        prompt=system_prompt,
                        role=self._llm.enums.SYSTEM.value,
                    )
                ],
                max_output_tokens=120,
            )
            rewritten = (rewritten or "").strip().strip('"').strip("'").strip()

            if rewritten:
                state["query_for_retrieval"] = rewritten
                state["agent_trace"].append(
                    f"{self.agent_name}: rewrote query → '{rewritten[:80]}'"
                )
                self.log_step(f"rewrote '{query[:60]}' → '{rewritten[:80]}'")
            else:
                state["query_for_retrieval"] = query
                state["agent_trace"].append(
                    f"{self.agent_name}: empty rewrite result — using original query"
                )
        except Exception as exc:
            state["query_for_retrieval"] = query
            state["agent_trace"].append(
                f"{self.agent_name}: rewrite failed ({exc}) — using original query"
            )
            self.log_step(f"rewrite failed: {exc}")

        return state

    # ------------------------------------------------------------------
    # Heuristics
    # ------------------------------------------------------------------

    def _looks_context_dependent(self, query: str) -> bool:
        normalised = query.lower().strip()
        tokens = set(normalised.replace("?", " ").replace("؟", " ").split())

        if len(tokens) <= _SHORT_QUERY_WORD_THRESHOLD:
            return True

        return bool(_REFERENTIAL_TERMS & tokens)