"""
NLP route request/response schemas — Phase 6.

Changes:
- ``AgentQueryRequest``:  added ``teaching_mode`` (item 7).
- ``AgentQueryResponse``: added ``citations``, ``teaching_mode`` (item 8 + 7).
- ``MultimodalQueryResponse``: added ``citations``, ``teaching_mode``.
- New ``SummaryRequest``: explicit schema for the /summarize endpoint with
  optional ``asset_ids`` and ``language`` so callers can limit the
  summary to specific lecture files and pick a language (item 1).
- All new fields are Optional with sensible defaults for full backward
  compatibility — existing callers that don't send them keep working.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class PushRequest(BaseModel):
    do_reset: Optional[int] = 0


class SearchRequest(BaseModel):
    text: str
    limit: Optional[int] = 5


class VisualizeRequest(BaseModel):
    text: str


class AgentQueryRequest(BaseModel):
    query: str
    project_id: str
    asset_ids: List[str] = Field(default_factory=list)
    session_id: str = "default"
    image_paths: List[str] = Field(default_factory=list)
    # Teaching mode selected for this query (item 7).
    # One of: "quick_review" | "full_explanation" | "exam_prep" |
    #         "step_by_step" | "" (empty = use stored preference or default)
    teaching_mode: Optional[str] = ""


class AgentQueryResponse(BaseModel):
    response: str
    agent_trace: Optional[List[str]] = None
    retrieved_chunks: Optional[List[dict]] = None
    session_id: str
    metadata: Optional[Dict[str, Any]] = None
    # Source citations derived from retrieved chunk metadata (item 8).
    citations: Optional[List[dict]] = None
    # Teaching mode that was active for this response (item 7).
    teaching_mode: Optional[str] = ""


class MultimodalQueryResponse(BaseModel):
    response: str
    agent_trace: List[str]
    retrieved_chunks: List[dict]
    sources_used: List[str]
    fusion_strategy: str
    session_id: str
    # Phase 6 additions
    citations: Optional[List[dict]] = None
    teaching_mode: Optional[str] = ""


class SummaryRequest(BaseModel):
    """
    Request schema for POST /v1/nlp/summarize/{project_id}.

    Phase 6: explicit schema replaces the implicit no-body endpoint so
    callers can scope the summary to specific asset files and choose a
    language.  All fields are optional — the endpoint falls back to
    summarising the entire project in English when omitted.
    """
    # Limit summary to these asset IDs (lecture files).  Empty = all files.
    asset_ids: Optional[List[str]] = Field(default_factory=list)
    # Language for summary output: "en" | "ar".  Default: "en".
    language: Optional[str] = "en"
