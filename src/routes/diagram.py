"""
Diagram routes.

POST /v1/diagram/generate/{project_id}
    Generate a Mermaid concept diagram from the project's indexed chunks.

    Body (JSON):
        asset_ids : list  (optional — filter to specific files)
        language  : str   ("en" | "ar", default "en")

    Response:
        {
          "title":        "Machine Learning Concepts",
          "diagram_type": "flowchart",
          "content":      "graph TD\\n  A[Machine Learning] --> B[Supervised]\\n  ..."
        }
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional

from agents.diagram.DiagramAgent import DiagramAgent
from models.ChunkModel import ChunkModel
from models.ProjectModel import ProjectModel

logger = logging.getLogger("uvicorn.error")

diagram_router = APIRouter(prefix="/v1/diagram")


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class DiagramRequest(BaseModel):
    asset_ids: Optional[List[str]] = Field(default_factory=list)
    language:  Optional[str] = "en"


# ---------------------------------------------------------------------------
# Generate diagram
# ---------------------------------------------------------------------------

@diagram_router.post("/generate/{project_id}")
async def generate_diagram(
    request: Request,
    project_id: str,
    body: DiagramRequest,
):
    """
    Generate a Mermaid concept diagram from the project's indexed lecture chunks.

    The endpoint runs a two-step pipeline:
    1. Concept extraction (map over ordered chunk batches).
    2. Mermaid code generation (reduce step).

    The client renders the returned Mermaid code using Mermaid JS.
    """
    try:
        project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
        project = await project_model.get_project_or_create_one(project_id=project_id)

        chunk_model = await ChunkModel.create_instance(db_client=request.app.db_client)
        chunks = await chunk_model.get_all_chunks_ordered(
            project_id=project.project_id,
            asset_ids=body.asset_ids or [],
            max_chunks=300,   # concept extraction only needs a broad sample
        )

        if not chunks:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "error": "No indexed content found for this project. "
                             "Please upload and process a lecture first."
                },
            )

        agent = DiagramAgent(
            generation_client=request.app.generation_client,
            template_parser=request.app.template_parser,
            language=body.language or "en",
        )

        result = agent.generate(chunks=chunks)

        return JSONResponse(content=result)

    except Exception as exc:
        logger.exception("Diagram generation error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))