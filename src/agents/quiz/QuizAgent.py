import json
import re
import uuid
from typing import Any, Dict, List


class QuizAgent:
    """Generate structured interactive quizzes from lecture context.

    This agent intentionally does not persist anything.  It reuses the
    already-configured generation client injected by FastAPI startup.
    """

    def __init__(self, llm_provider) -> None:
        self._llm = llm_provider

    def generate(self, context: str, num_questions: int = 5, language: str = "English") -> Dict[str, Any]:
        context = self._prepare_context(context)
        if not context:
            raise ValueError("No document context was provided for quiz generation.")

        num_questions = self._normalize_num_questions(num_questions)
        language_name = self._normalize_language(language)

        system_prompt = self._build_system_prompt(num_questions=num_questions, language=language_name)
        user_prompt = self._build_user_prompt(context=context, num_questions=num_questions, language=language_name)

        chat_history = [
            self._llm.construct_prompt(
                prompt=system_prompt,
                role=self._llm.enums.SYSTEM.value,
            )
        ]

        raw_response = self._llm.generate_text(
            prompt=user_prompt,
            chat_history=chat_history,
            max_output_tokens=min(12000, max(2500, num_questions * 700)),
            temperature=0.2,
        )

        parsed = self._parse_json_object(raw_response or "")
        questions = self._validate_questions(parsed.get("questions", []), num_questions=num_questions)

        return {
            "quiz_id": str(uuid.uuid4()),
            "questions": questions,
        }

    def _normalize_num_questions(self, num_questions: int) -> int:
        try:
            value = int(num_questions)
        except Exception:
            value = 5
        return max(1, min(value, 20))

    def _normalize_language(self, language: str) -> str:
        language = (language or "English").strip()
        if language.lower() in {"arabic", "ar", "العربية", "عربي"}:
            return "Arabic"
        return "English"

    def _prepare_context(self, context: str) -> str:
        cleaned = (context or "").replace("\x00", "").strip()
        # Keep the prompt bounded while still allowing lecture-wide coverage.
        return cleaned[:60000]

    def _build_system_prompt(self, num_questions: int, language: str) -> str:
        return "\n".join([
            "You are a careful university professor and assessment designer.",
            "Generate an interactive multiple-choice quiz from the provided lecture/document context only.",
            "Return ONLY valid JSON. Do not wrap it in Markdown fences.",
            "Never invent facts that are not supported by the context.",
            f"The quiz language MUST be {language}.",
            "",
            "Required JSON schema:",
            "{",
            '  "questions": [',
            "    {",
            '      "question": "string",',
            '      "options": ["string", "string", "string", "string"],',
            '      "correct_answer": "string that exactly equals one option",',
            '      "hint": "short helpful hint without revealing the answer",',
            '      "explanation": "why the correct answer is correct"',
            "    }",
            "  ]",
            "}",
            "",
            "Rules:",
            f"- Generate EXACTLY {num_questions} questions.",
            "- Each question must have exactly 4 options.",
            "- Exactly one option must be correct.",
            "- correct_answer MUST exactly match one of the option strings.",
            "- Each hint should help the learner think, but must not directly reveal the answer.",
            "- Each explanation should be concise and grounded in the context.",
            "- Prefer questions that test understanding, relationships, definitions, and lecture-specific details.",
        ])

    def _build_user_prompt(self, context: str, num_questions: int, language: str) -> str:
        return "\n\n".join([
            f"Create exactly {num_questions} interactive MCQ questions in {language} from this lecture context.",
            "LECTURE CONTEXT:",
            context,
            "Return valid JSON only now.",
        ])

    def _parse_json_object(self, text: str) -> Dict[str, Any]:
        cleaned = (text or "").strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("LLM did not return a JSON object for the quiz.")
            data = json.loads(cleaned[start:end + 1])

        if not isinstance(data, dict):
            raise ValueError("Quiz generation response must be a JSON object.")
        return data

    def _validate_questions(self, raw_questions: Any, num_questions: int) -> List[Dict[str, Any]]:
        if not isinstance(raw_questions, list) or not raw_questions:
            raise ValueError("Quiz generation returned no questions.")

        questions: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_questions[:num_questions], start=1):
            if not isinstance(item, dict):
                continue

            question = str(item.get("question", "")).strip()
            options = item.get("options", [])
            if not question or not isinstance(options, list):
                continue

            normalized_options = [str(opt).strip() for opt in options if str(opt).strip()]
            # Preserve exactly four unique option strings where possible.
            deduped_options: List[str] = []
            for opt in normalized_options:
                if opt not in deduped_options:
                    deduped_options.append(opt)
            normalized_options = deduped_options[:4]

            if len(normalized_options) != 4:
                continue

            correct_answer = str(item.get("correct_answer", "")).strip()
            correct_answer = self._resolve_correct_answer(correct_answer, normalized_options)
            if not correct_answer:
                continue

            questions.append({
                "question": question,
                "options": normalized_options,
                "correct_answer": correct_answer,
                "hint": str(item.get("hint", "Review the related lecture concept and eliminate unlikely options.")).strip(),
                "explanation": str(item.get("explanation", "This answer is supported by the lecture context.")).strip(),
            })

        if not questions:
            raise ValueError("Quiz generation returned invalid question objects.")

        return questions

    def _resolve_correct_answer(self, correct_answer: str, options: List[str]) -> str:
        if not correct_answer:
            return ""

        # Accept A/B/C/D if the model ignored the instruction.
        letter_map = {"A": 0, "B": 1, "C": 2, "D": 3, "أ": 0, "ب": 1, "ج": 2, "د": 3}
        cleaned = correct_answer.strip()
        if cleaned.upper() in letter_map:
            return options[letter_map[cleaned.upper()]]
        if cleaned in letter_map:
            return options[letter_map[cleaned]]

        for opt in options:
            if cleaned == opt:
                return opt
            if cleaned.lower() == opt.lower():
                return opt

        # Strip common labels such as "A)" and compare again.
        stripped = re.sub(r"^[A-Da-dأبجد][\).:\-\s]+", "", cleaned).strip()
        for opt in options:
            if stripped and stripped.lower() == opt.lower():
                return opt

        return ""
