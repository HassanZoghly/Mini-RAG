"""
Shared, lightweight task-type classification.

Several agents (``RetrievalAgent``, ``ReasoningAgent``,
``ResponseFormatterAgent``) need to agree on whether the current request
is a summary, a quiz, an "explain this in depth" question, or a simple
factual lookup — because each of those needs a different amount of
retrieved context and a different system prompt.

Previously each agent re-implemented its own ad-hoc keyword check
(``"summarize" in query_lower or "ملخص" in query_lower``, etc.), which made
it easy for the checks to drift out of sync. This module centralises that
logic.
"""

from __future__ import annotations

TASK_SUMMARY = "summary"
TASK_QUIZ = "quiz"
TASK_EXPLANATION = "explanation"
TASK_SIMPLE_QA = "simple_qa"


_SUMMARY_KEYWORDS = (
    "summarize", "summarise", "summary", "overview of the lecture",
    "ملخص", "تلخيص", "لخص", "اعمل ملخص", "اكتب ملخص",
)

_QUIZ_KEYWORDS = (
    "quiz", "mcq", "multiple choice", "exam questions", "practice questions",
    "امتحان", "اختبار", "أسئلة اختيار", "اسئلة امتحان",
)

_EXPLANATION_KEYWORDS = (
    "explain", "describe", "how does", "how do", "how is", "why does", "why is",
    "compare", "difference between", "walk me through", "derive", "derivation",
    "in detail", "step by step", "step-by-step", "what is the intuition",
    "اشرح", "فسر", "وضح", "اشرحلي", "ما الفرق", "قارن", "بالتفصيل", "خطوة بخطوة",
)

# Queries with more than this many words are treated as needing a deeper
# (explanation-style) answer even without an explicit keyword match.
_LONG_QUERY_WORD_THRESHOLD = 12


def classify_task_type(query: str, route: str = "") -> str:
    """
    Classify *query* (optionally informed by the router's ``route``) into
    one of ``TASK_SUMMARY``, ``TASK_QUIZ``, ``TASK_EXPLANATION``, or
    ``TASK_SIMPLE_QA``.
    """
    query_lower = (query or "").lower()
    route_lower = (route or "").lower()

    if route_lower == TASK_SUMMARY or any(kw in query_lower for kw in _SUMMARY_KEYWORDS):
        return TASK_SUMMARY

    if route_lower == TASK_QUIZ or any(kw in query_lower for kw in _QUIZ_KEYWORDS):
        return TASK_QUIZ

    if route_lower == "reasoning" or any(kw in query_lower for kw in _EXPLANATION_KEYWORDS):
        return TASK_EXPLANATION

    if len(query_lower.split()) > _LONG_QUERY_WORD_THRESHOLD:
        return TASK_EXPLANATION

    return TASK_SIMPLE_QA


# Retrieval sizing per task type (item 4C — dynamic retrieval size).
# ``fetch_limit`` is how many candidates to pull from the vector DB before
# reranking; ``top_n`` is how many to keep after reranking.
# ``TASK_SUMMARY`` is handled separately via full ordered-chunk retrieval
# (see RetrievalAgent / ChunkModel.get_all_chunks_ordered) and therefore has
# no fetch/top_n here.
RETRIEVAL_SIZES = {
    TASK_SIMPLE_QA: {"fetch_limit": 15, "top_n": 5},
    TASK_EXPLANATION: {"fetch_limit": 30, "top_n": 12},
    TASK_QUIZ: {"fetch_limit": 60, "top_n": 20},
}
