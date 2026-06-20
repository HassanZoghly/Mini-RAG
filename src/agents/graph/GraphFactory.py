from services.nlp_service import NLPService as _NLPService
from services.document_service import DocumentService as _DocumentService

from agents.graph.AgentGraph import AgentGraph
from agents.router.RouterAgent import RouterAgent
from agents.retrieval.RetrievalAgent import RetrievalAgent
from agents.retrieval.QueryRewriterAgent import QueryRewriterAgent
from agents.memory.MemoryStore import MemoryStore
from agents.memory.MemoryAgent import MemoryAgent
from agents.multimodal.OCRAgent import OCRAgent
from agents.multimodal.VisionAgent import VisionAgent
from agents.reasoning.ReasoningAgent import ReasoningAgent
from agents.response.ResponseFormatterAgent import ResponseFormatterAgent
from agents.smalltalk.SmallTalkAgent import SmallTalkAgent


class GraphFactory:
    """
    Single wiring point that reads all dependencies from the FastAPI
    ``app`` object and assembles a fully-configured ``AgentGraph``.

    All clients are attached directly to ``app`` (not ``app.state``)
    during ``startup_span`` in ``main.py``:

    * ``app.generation_client``  — LLM provider for text generation.
    * ``app.embedding_client``   — LLM provider for embedding.
    * ``app.vectordb_client``    — Vector-DB client (Qdrant / PGVector).
    * ``app.reranker_client``    — Reranker provider (Cohere, etc.).
    * ``app.template_parser``    — ``TemplateParser`` for prompt rendering.
    * ``app.db_client``          — SQLAlchemy async session factory.

    Phase 4 changes
    ---------------
    * ``QueryRewriterAgent`` is instantiated and passed to ``AgentGraph``
      so it is wired between ``memory`` and ``retrieval`` in all four
      graph builders.
    * ``RetrievalAgent`` now receives ``db_client`` so it can call
      ``ChunkModel.get_all_chunks_ordered`` for summary queries.

    Usage
    -----
    ::

        # In main.py startup_span, after all clients are created:
        app.agent_graph = GraphFactory.create(app)
    """

    @staticmethod
    def create(app) -> AgentGraph:
        """
        Instantiate every agent with the correct dependencies and return
        a wired ``AgentGraph``.
        """
        generation_client = app.generation_client
        embedding_client  = app.embedding_client
        vectordb_client   = app.vectordb_client
        template_parser   = app.template_parser
        db_client         = app.db_client
        reranker_client   = app.reranker_client

        # NLPService — shared helper for _flatten_vector and collection naming
        nlp_service = _NLPService(
            vectordb_client=vectordb_client,
            generation_client=generation_client,
            embedding_client=embedding_client,
            template_parser=template_parser,
        )

        # DocumentService — used by OCRAgent for image pre-processing
        document_service = _DocumentService(project_id="")

        # MemoryStore — semantic memory backed by pgvector
        memory_store = MemoryStore(
            nlp_controller=nlp_service,
            async_session_maker=db_client,
        )

        # -- Agents --------------------------------------------------------

        router_agent = RouterAgent(
            llm_provider=generation_client,
        )

        # Phase 4: db_client passed so summary queries can use ordered retrieval
        retrieval_agent = RetrievalAgent(
            embedding_client=embedding_client,
            vectordb_client=vectordb_client,
            nlp_controller=nlp_service,
            reranker_client=reranker_client,
            db_client=db_client,
        )

        memory_agent = MemoryAgent(
            memory_store=memory_store,
        )

        # Phase 4: QueryRewriterAgent — rewrites follow-up questions using
        # session memory before retrieval runs (item 4A)
        query_rewriter_agent = QueryRewriterAgent(
            llm_provider=generation_client,
        )

        ocr_agent = OCRAgent(
            process_controller=document_service,
        )

        vision_agent = VisionAgent(
            llm_provider=generation_client,
        )

        reasoning_agent = ReasoningAgent(
            llm_provider=generation_client,
            template_parser=template_parser,
        )

        response_formatter_agent = ResponseFormatterAgent(
            llm_provider=generation_client,
            template_parser=template_parser,
        )

        smalltalk_agent = SmallTalkAgent(
            llm_provider=generation_client,
        )

        return AgentGraph(
            router=router_agent,
            retrieval=retrieval_agent,
            memory=memory_agent,
            ocr=ocr_agent,
            vision=vision_agent,
            reasoning=reasoning_agent,
            response_formatter=response_formatter_agent,
            smalltalk=smalltalk_agent,
            query_rewriter=query_rewriter_agent,
        )
