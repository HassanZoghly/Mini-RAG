"""
QuizAgent — generates an interactive multiple-choice quiz from lecture chunks.

Flow
----
1. Receive ordered chunks (from ChunkModel.get_all_chunks_ordered or a
   targeted vector search result).
2. Build a dense content block (map-reduce if needed, same pattern as
   SummaryGenerator but lighter — we only need concept extraction, not
   a full summary).
3. Call the LLM with a strict JSON-output prompt.
4. Parse and validate the response into a list of QuizQuestion dicts.
5. Return the list — the route handler wraps it in the API response.

The agent is stateless and synchronous (no LangGraph node needed) — it
is called directly from the quiz route handler.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import List, Optional

logger = logging.getLogger(__name__)

# Maximum characters of lecture content sent per generation call.
# Leaves headroom inside the 16 000-char process_text limit for the
# system prompt (~1 500 chars) and the JSON output (~3 000 chars).
_CONTENT_BATCH_CHARS = 11_000

# Output token budget — enough for 10 detailed questions.
_MAX_TOKENS = 3_500


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

def _make_question(
    question: str,
    options: List[str],
    correct_answer: str,
    hint: str,
    explanation: str,
) -> dict:
    return {
        "id":             str(uuid.uuid4()),
        "question":       question.strip(),
        "options":        [o.strip() for o in options],
        "correct_answer": correct_answer.strip(),
        "hint":           hint.strip(),
        "explanation":    explanation.strip(),
    }


# ---------------------------------------------------------------------------
# QuizAgent
# ---------------------------------------------------------------------------

class QuizAgent:
    """
    Generates a multiple-choice quiz from a list of DataChunk ORM objects
    (or dict chunks) using the project's LLM generation client.

    Parameters
    ----------
    generation_client : LLM provider (CoHereProvider / OpenAIProvider)
    language          : "en" | "ar"
    num_questions     : how many questions to request (1-30)
    """

    def __init__(
        self,
        generation_client,
        language: str = "en",
        num_questions: int = 10,
    ) -> None:
        self._llm          = generation_client
        self._language     = language if language in ("ar", "en") else "en"
        self._num_q        = max(1, min(30, num_questions))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, chunks: list) -> List[dict]:
        """
        Generate quiz questions from *chunks*.

        *chunks* may be DataChunk ORM objects (with ``.chunk_text`` /
        ``.chunk_metadata``) or ``{text, score, metadata}`` dicts.

        Returns a list of question dicts. Returns [] on failure.
        """
        if not chunks:
            return []

        content = self._build_content(chunks)
        if not content.strip():
            return []

        # If content is very long, use only the first batch to stay within
        # the process_text limit. For quiz generation we prioritise breadth
        # of topics over completeness, so a single well-chosen window is fine.
        if len(content) > _CONTENT_BATCH_CHARS:
            content = content[:_CONTENT_BATCH_CHARS]

        raw = self._call_llm(content)
        if not raw:
            return []

        questions = self._parse_json(raw)
        logger.info("QuizAgent: generated %d questions", len(questions))
        return questions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_content(self, chunks: list) -> str:
        """Concatenate chunk texts into a single labelled content block."""
        parts: List[str] = []
        for chunk in chunks:
            if hasattr(chunk, "chunk_text"):
                text = (chunk.chunk_text or "").strip()
                meta = chunk.chunk_metadata or {}
            else:
                text = (chunk.get("text", "")).strip()
                meta = chunk.get("metadata", {})

            if not text:
                continue

            label_parts = []
            if meta.get("section"):
                label_parts.append(meta["section"])
            if meta.get("page"):
                label_parts.append(f"p.{meta['page']}")
            label = f"[{' | '.join(label_parts)}] " if label_parts else ""
            parts.append(f"{label}{text}")

        return "\n\n".join(parts)

    def _call_llm(self, content: str) -> Optional[str]:
        """Call the LLM with the quiz-generation prompt. Return raw text."""
        system_prompt = self._system_prompt()
        user_prompt   = self._user_prompt(content)

        chat_history = [
            self._llm.construct_prompt(
                prompt=system_prompt,
                role=self._llm.enums.SYSTEM.value,
            )
        ]

        try:
            result = self._llm.generate_text(
                prompt=user_prompt,
                chat_history=chat_history,
                max_output_tokens=_MAX_TOKENS,
                temperature=0.4,   # slightly creative but still focused
            )
            return result
        except Exception as exc:
            logger.error("QuizAgent: LLM call failed: %s", exc)
            return None

    def _parse_json(self, raw: str) -> List[dict]:
        """
        Extract and parse the JSON array from the LLM response.

        The LLM is instructed to return ONLY a JSON array, but it sometimes
        wraps it in markdown fences or adds a preamble — we strip those.
        """
        if not raw:
            return []

        # Strip markdown fences
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        cleaned = cleaned.strip("`").strip()

        # Find the first '[' and last ']' to isolate the array
        start = cleaned.find("[")
        end   = cleaned.rfind("]")
        if start == -1 or end == -1 or end <= start:
            logger.warning("QuizAgent: no JSON array found in response")
            return []

        json_str = cleaned[start : end + 1]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.warning("QuizAgent: JSON parse failed: %s", exc)
            # Try to salvage partial output
            data = self._salvage_partial_json(json_str)

        if not isinstance(data, list):
            return []

        questions: List[dict] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            q       = item.get("question",       "").strip()
            opts    = item.get("options",         [])
            correct = item.get("correct_answer", "").strip()
            hint    = item.get("hint",            "").strip()
            expl    = item.get("explanation",     "").strip()

            if not q or not opts or not correct:
                continue
            if not isinstance(opts, list) or len(opts) < 2:
                continue

            questions.append(_make_question(q, opts, correct, hint, expl))

        return questions

    def _salvage_partial_json(self, text: str) -> list:
        """
        Try to extract complete question objects even if the final array is
        truncated (happens with very long outputs near the token limit).
        """
        objects = []
        depth = 0
        start = None

        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    fragment = text[start : i + 1]
                    try:
                        obj = json.loads(fragment)
                        objects.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = None

        return objects

    # ------------------------------------------------------------------
    # Prompt templates
    # ------------------------------------------------------------------

    def _system_prompt(self) -> str:
        if self._language == "ar":
            return (
                "أنت مولّد أسئلة اختيار من متعدد متخصص في المحتوى التعليمي. "
                "مهمتك إنشاء أسئلة اختبار عالية الجودة من المحتوى الدراسي المُقدَّم. "
                "يجب أن تكون الأسئلة متنوعة وتغطي المفاهيم المختلفة. "
                "يجب أن تُعيد JSON صرفاً فقط — لا مقدمة ولا شرح خارج JSON."
            )
        return (
            "You are a multiple-choice quiz generator specialising in educational content. "
            "Your task is to create high-quality exam questions from the provided study material. "
            "Questions must be varied and cover different concepts and difficulty levels. "
            "You MUST return ONLY raw JSON — no preamble, no explanation outside the JSON."
        )

    def _user_prompt(self, content: str) -> str:
        if self._language == "ar":
            schema_example = json.dumps([
                {
                    "question":       "ما هو الهدف الرئيسي من تقليل الأبعاد في تعلم الآلة؟",
                    "options":        ["تحسين دقة النموذج", "تقليل التعقيد الحسابي", "زيادة عدد الميزات", "تجاهل القيم الشاذة"],
                    "correct_answer": "تقليل التعقيد الحسابي",
                    "hint":           "فكّر في الهدف الأصلي من هذه التقنية من حيث المعالجة.",
                    "explanation":    "تقليل الأبعاد يساعد على خفض التعقيد الحسابي والتخلص من الميزات غير الضرورية مع الحفاظ على المعلومات الجوهرية."
                }
            ], ensure_ascii=False, indent=2)
            return (
                f"بناءً على محتوى المحاضرة التالي، أنشئ بالضبط {self._num_q} سؤال اختيار من متعدد باللغة العربية.\n\n"
                f"يجب أن يتبع كل سؤال هذه البنية بالضبط:\n{schema_example}\n\n"
                "القواعد:\n"
                "- يجب أن يحتوي كل سؤال على 4 خيارات بالضبط (A, B, C, D)\n"
                "- يجب أن تكون الإجابة الصحيحة نصاً يطابق أحد الخيارات بالضبط\n"
                "- يجب أن يكون التلميح مفيداً دون أن يكشف الإجابة مباشرة\n"
                "- يجب أن يشرح التفسير سبب صحة الإجابة وخطأ البدائل الأخرى\n"
                "- غطِّ مواضيع مختلفة من المحتوى\n"
                "- أعد مصفوفة JSON فقط — لا نص خارجها\n\n"
                f"محتوى المحاضرة:\n{content}"
            )

        schema_example = json.dumps([
            {
                "question":       "What is the primary goal of dimensionality reduction in machine learning?",
                "options":        ["Improve model accuracy", "Reduce computational complexity", "Increase feature count", "Remove outliers"],
                "correct_answer": "Reduce computational complexity",
                "hint":           "Think about the original motivation for this technique in terms of processing.",
                "explanation":    "Dimensionality reduction lowers computational complexity and removes irrelevant features while retaining the most informative aspects of the data."
            }
        ], indent=2)
        return (
            f"Based on the following lecture content, generate exactly {self._num_q} multiple-choice questions in English.\n\n"
            f"Each question MUST follow this exact structure:\n{schema_example}\n\n"
            "Rules:\n"
            "- Each question must have exactly 4 options\n"
            "- correct_answer must be the exact text of one of the options\n"
            "- hint must be helpful but not give the answer away directly\n"
            "- explanation must explain why the answer is correct AND why the others are wrong\n"
            "- Cover different topics from the content\n"
            "- Return ONLY the JSON array — no text outside it\n\n"
            f"Lecture content:\n{content}"
        )
