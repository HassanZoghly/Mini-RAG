from pydantic import BaseModel
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
    asset_ids: List[str] = []
    session_id: str = "default"
    image_paths: List[str] = []

class AgentQueryResponse(BaseModel):
    response: str
    agent_trace: Optional[List[str]] = None
    retrieved_chunks: Optional[List[dict]] = None
    session_id: str
    metadata: Optional[Dict[str, Any]] = None

class MultimodalQueryResponse(BaseModel):
    response: str
    agent_trace: List[str]
    retrieved_chunks: List[dict]
    sources_used: List[str]
    fusion_strategy: str
    session_id: str
