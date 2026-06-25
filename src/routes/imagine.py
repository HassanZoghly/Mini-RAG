"""
Imagine routes.

POST /v1/imagine/generate/{project_id}
    Generate an HTML/CSS infographic poster from the project's indexed chunks.

    Body (JSON):
        asset_ids : list  (optional — filter to specific files)
        language  : str   ("en" | "ar", default "en")

    Response:
        {
          "title":        "Lecture Infographic",
          "diagram_type": "imagine",
          "content":      "<div>...HTML...</div>"
        }
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional

from agents.imagine.ImagineAgent import ImagineAgent
from models.ChunkModel import ChunkModel
from models.ProjectModel import ProjectModel

logger = logging.getLogger("uvicorn.error")

imagine_router = APIRouter(prefix="/v1/imagine")


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class ImagineRequest(BaseModel):
    asset_ids: Optional[List[str]] = Field(default_factory=list)
    language:  Optional[str] = "en"


# ---------------------------------------------------------------------------
# Generate imagine
# ---------------------------------------------------------------------------

@imagine_router.post("/generate/{project_id}")
async def generate_imagine(
    request: Request,
    project_id: str,
    body: ImagineRequest,
):
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

        agent = ImagineAgent(
            generation_client=request.app.generation_client,
            template_parser=request.app.template_parser,
            language=body.language or "en",
        )

        result = agent.generate(chunks=chunks)

        return JSONResponse(content=result)

    except Exception as exc:
        logger.exception("Imagine generation error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
