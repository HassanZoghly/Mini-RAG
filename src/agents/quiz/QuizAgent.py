"""
QuizAgent - Generates an interactive multiple-choice quiz from lecture chunks.
Refactored to completely use TemplateParser, eliminating hardcoded prompts and dictionaries.
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

# Output token budget - enough for 10 detailed questions.
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
    (or dict chunks) using the project's LLM generation client and TemplateParser.
    """
    def __init__(
        self,
        generation_client,
        template_parser,
        language: str = "en",
        num_questions: int = 10,
        difficulty: str = "MEDIUM",
    ) -> None:
        self._llm             = generation_client
        self._template_parser = template_parser
        self._language        = language if language in ("ar", "en") else "en"
        self._num_q           = max(1, min(30, num_questions))
        self._difficulty      = difficulty.upper() if difficulty.upper() in ("EASY", "MEDIUM", "HARD") else "MEDIUM"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, chunks: list) -> List[dict]:
        if not chunks:
            return []

        content = self._build_content(chunks)
        if not content.strip():
            return []

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
                temperature=0.4, 
            )
            return result
        except Exception as exc:
            logger.error("QuizAgent: LLM call failed: %s", exc)
            return None

    def _parse_json(self, raw: str) -> List[dict]:
        if not raw:
            return []

        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        cleaned = cleaned.strip("`").strip()

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
            data = self._salvage_partial_json(json_str)

        if not isinstance(data, list):
            return []

        questions: List[dict] = []
        for item in data:
            if not isinstance(item, dict):
                continue

            q       = item.get("question", "").strip()
            opts    = item.get("options", [])
            correct = item.get("correct_answer", item.get("answer", "")).strip()
            hint    = item.get("hint", "").strip()
            expl    = item.get("explanation", "").strip()

            if not q or not opts or not correct:
                continue
            if not isinstance(opts, list) or len(opts) < 2:
                continue

            questions.append(_make_question(q, opts, correct, hint, expl))

        return questions

    def _salvage_partial_json(self, text: str) -> list:
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
    # Prompt templates integration
    # ------------------------------------------------------------------
    def _system_prompt(self) -> str:
        self._template_parser.set_language(self._language)
        
        base_prompt = self._template_parser.get("rag", "quiz_system_base")
        difficulty_prompt = self._template_parser.get("rag", f"quiz_system_{self._difficulty.lower()}")
        
        # Fallbacks in case templates are missing
        if not base_prompt:
            base_prompt = "You must output ONLY a valid JSON array of objects."
        if not difficulty_prompt:
            difficulty_prompt = f"DIFFICULTY: {self._difficulty}."

        return f"{base_prompt}\n\n{difficulty_prompt}"

    def _user_prompt(self, content: str) -> str:
        self._template_parser.set_language(self._language)
        
        user_prompt = self._template_parser.get("rag", "quiz_user_prompt", {
            "num_q": self._num_q,
            "content": content
        })
        
        if not user_prompt:
            user_prompt = f"Generate {self._num_q} multiple-choice questions based on this text:\n\n{content}"
            
        return user_prompt