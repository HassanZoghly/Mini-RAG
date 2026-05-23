from pydantic import BaseModel
from typing import Optional, List, Dict

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
    asset_ids: List[str] = []
    session_id: str = "default"
    image_paths: List[str] = []

class AgentQueryResponse(BaseModel):
    response: str
    agent_trace: List[str]
    retrieved_chunks: List[dict]
    session_id: str
