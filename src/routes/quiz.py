"""
Quiz routes.

POST /v1/quiz/generate/{project_id}
    Generate an interactive multiple-choice quiz from the project's
    indexed lecture chunks.

    Body (JSON):
        num_questions : int   (1-30, default 10)
        asset_ids     : list  (optional — filter to specific files)
        language      : str   ("en" | "ar", default "en")

    Response:
        {
          "quiz_id":   "<uuid>",
          "questions": [
            {
              "id":             "<uuid>",
              "question":       "...",
              "options":        ["A text", "B text", "C text", "D text"],
              "correct_answer": "A text",   -- exact text of the correct option
              "hint":           "...",
              "explanation":    "..."
            },
            ...
          ]
        }

POST /v1/quiz/answer
    Stateless answer-checking endpoint.  The client sends the question
    data (from the quiz it already has) plus the user's chosen answer;
    the server validates and returns feedback without a DB lookup.

    Body (JSON):
        question_id   : str
        question      : str
        correct_answer: str
        user_answer   : str
        hint          : str
        explanation   : str

    Response:
        {
          "correct": true/false,
          "message": "...",
          "explanation": "...",
          "hint": "..."   -- only when wrong
        }
"""

from __future__ import annotations

import uuid
import logging

from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional

from agents.quiz.QuizAgent import QuizAgent
from controllers.NLPController import NLPController
from models.ChunkModel import ChunkModel
from models.ProjectModel import ProjectModel

logger = logging.getLogger("uvicorn.error")

quiz_router = APIRouter(prefix="/v1/quiz")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class QuizGenerateRequest(BaseModel):
    num_questions: Optional[int] = 10
    asset_ids:     Optional[List[str]] = Field(default_factory=list)
    language:      Optional[str] = "en"
    difficulty:    Optional[str] = "MEDIUM"


class QuizAnswerRequest(BaseModel):
    question_id:    str
    question:       str
    correct_answer: str
    user_answer:    str
    hint:           Optional[str] = ""
    explanation:    Optional[str] = ""


# ---------------------------------------------------------------------------
# Generate quiz
# ---------------------------------------------------------------------------

@quiz_router.post("/generate/{project_id}")
async def generate_quiz(
    request: Request,
    project_id: str,
    body: QuizGenerateRequest,
):
    """
    Generate a multiple-choice quiz from the project's indexed lecture chunks.
    """
    try:
        project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
        project = await project_model.get_project_or_create_one(project_id=project_id)

        chunk_model = await ChunkModel.create_instance(db_client=request.app.db_client)
        chunks = await chunk_model.get_all_chunks_ordered(
            project_id=project.project_id,
            asset_ids=body.asset_ids or [],
            max_chunks=500,   # quiz only needs a representative sample
        )

        if not chunks:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "error": "No indexed content found for this project. "
                             "Please upload and process a lecture first."
                },
            )

        agent = QuizAgent(
            generation_client=request.app.generation_client,
            template_parser=request.app.template_parser,
            language=body.language or "en",
            num_questions=body.num_questions or 10,
            difficulty=body.difficulty or "MEDIUM",
        )

        questions = agent.generate(chunks=chunks)

        if not questions:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Quiz generation failed. Please try again."},
            )

        return JSONResponse(
            content={
                "quiz_id":   str(uuid.uuid4()),
                "questions": questions,
            }
        )

    except Exception as exc:
        logger.exception("Quiz generation error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Check answer (stateless)
# ---------------------------------------------------------------------------

@quiz_router.post("/answer")
async def check_answer(body: QuizAnswerRequest):
    """
    Stateless answer validation. The client supplies the full question
    context it already has; no DB lookup is required.
    """
    is_correct = (
        body.user_answer.strip().lower() == body.correct_answer.strip().lower()
    )

    if is_correct:
        return JSONResponse(content={
            "correct":     True,
            "message":     "✅ Correct! Well done.",
            "explanation": body.explanation or "",
            "hint":        "",
        })
    else:
        return JSONResponse(content={
            "correct":     False,
            "message":     "❌ Not quite right. Here's a hint:",
            "explanation": body.explanation or "",
            "hint":        body.hint or "Review this concept in the lecture.",
        })