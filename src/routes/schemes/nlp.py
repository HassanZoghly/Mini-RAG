from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class PushRequest(BaseModel):
    do_reset: Optional[int] = 0

class SearchRequest(BaseModel):
    text: str
    limit: Optional[int] = 5

class VisualizeRequest(BaseModel):
    text: str

class QuizQuestion(BaseModel):
    question: str
    options: List[str] = Field(min_length=4, max_length=4)
    correct_answer: str
    hint: str
    explanation: str

class QuizGenerateRequest(BaseModel):
    num_questions: Optional[int] = 5
    language: Optional[str] = "English"

class QuizGenerateResponse(BaseModel):
    quiz_id: str
    questions: List[QuizQuestion]

class QuizAnswerRequest(BaseModel):
    quiz_id: Optional[str] = None
    question_index: Optional[int] = None
    selected_answer: str
    correct_answer: str
    explanation: Optional[str] = None
    hint: Optional[str] = None

class QuizAnswerResponse(BaseModel):
    quiz_id: Optional[str] = None
    question_index: Optional[int] = None
    is_correct: bool
    message: str
    explanation: Optional[str] = None
    hint: Optional[str] = None

class DiagramGenerateRequest(BaseModel):
    language: Optional[str] = "English"
    diagram_type: Optional[str] = "flowchart"

class DiagramGenerateResponse(BaseModel):
    title: str
    diagram_type: str
    content: str

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
