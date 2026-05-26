from typing import List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Typed dictionary representing the shared mutable state that flows
    through every agent in the Mini-RAG agent pipeline.

    Fields
    ------
    query : str
        The original natural-language question submitted by the user.
    project_id : str
        Identifier of the active Mini-RAG project (maps to a vector-DB
        collection and asset directory).
    asset_ids : List[str]
        Subset of document/asset identifiers to restrict retrieval to.
        An empty list means *all* assets in the project are eligible.
    retrieved_chunks : List[dict]
        Text chunks returned by the vector-DB retrieval step.  Each dict
        contains at minimum ``{"text": str, "score": float}``.
    memory_context : List[dict]
        Conversation-history entries injected from an external memory
        store.  Each dict mirrors the chat-message schema used by the
        LLM provider (``{"role": str, "content": str}``).
    image_paths : List[str]
        Absolute or project-relative paths to image files attached to
        the current query (used by vision-capable agents).
    ocr_text : str
        Raw text extracted from images via OCR; populated by the OCR
        agent and consumed by subsequent reasoning agents.
    vision_description : str
        Free-form description of image contents produced by the vision
        agent (e.g. a multimodal LLM caption or analysis).
    reasoning_context : str
        Intermediate reasoning or chain-of-thought output assembled by
        the reasoning agent before final answer generation.
    final_response : str
        The finished natural-language answer returned to the user.
    agent_trace : List[str]
        Ordered log of every agent step in the format
        ``"AgentName: <action description>"``.  Append to this list
        inside each agent's ``execute`` method for full observability.
    error : Optional[str]
        Set to a non-``None`` string by any agent that encounters an
        unrecoverable error; downstream agents should check this field
        and short-circuit if it is set.
    metadata : dict
        Catch-all mapping for any extra key-value pairs that individual
        agents need to pass between themselves without polluting the
        typed schema.
    """

    query: str
    project_id: str
    asset_ids: List[str]
    retrieved_chunks: List[dict]
    memory_context: List[dict]
    image_paths: List[str]
    ocr_text: str
    vision_description: str
    reasoning_context: str
    final_response: str
    agent_trace: List[str]
    error: Optional[str]
    metadata: dict
    uploaded_files: List[dict]
    file_texts: List[dict]
    image_base64: List[dict]
    fusion_strategy: str
    sources_used: List[str]
    visualization_urls: List[str]


def create_initial_state(
    query: str,
    project_id: str,
    asset_ids: List[str],
    image_paths: List[str],
    uploaded_files: Optional[List[dict]] = None,
    visualization_urls=[],
) -> AgentState:
    """
    Factory function that returns a fully-initialised ``AgentState`` with
    sensible defaults for all optional fields.
    """
    return AgentState(
        query=query,
        project_id=project_id,
        asset_ids=asset_ids,
        retrieved_chunks=[],
        memory_context=[],
        image_paths=image_paths,
        ocr_text="",
        vision_description="",
        reasoning_context="",
        final_response="",
        agent_trace=[],
        error=None,
        metadata={},
        uploaded_files=uploaded_files or [],
        file_texts=[],
        image_base64=[],
        fusion_strategy="text_only",
        sources_used=[],
    )
