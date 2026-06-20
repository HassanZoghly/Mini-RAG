from controllers.NLPController import NLPController
from controllers.ProcessController import ProcessController

from agents.graph.AgentGraph import AgentGraph
from agents.router.RouterAgent import RouterAgent
from agents.retrieval.RetrievalAgent import RetrievalAgent
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
    * ``app.template_parser``    — ``TemplateParser`` for prompt rendering.
    * ``app.db_client``          — SQLAlchemy async session factory.

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

        Parameters
        ----------
        app : FastAPI
            The running FastAPI application with all clients already set
            as direct attributes.

        Returns
        -------
        AgentGraph
            A fully wired graph instance whose ``run`` and ``stream``
            methods are ready to be called from route handlers.
        """
        generation_client = app.generation_client
        embedding_client  = app.embedding_client
        vectordb_client   = app.vectordb_client
        template_parser   = app.template_parser
        db_client         = app.db_client
        reranker_client   = app.reranker_client

        # NLPController — used for _flatten_vector and memory store embedding
        nlp_controller = NLPController(
            vectordb_client=vectordb_client,
            generation_client=generation_client,
            embedding_client=embedding_client,
            template_parser=template_parser,
        )

        # ProcessController — OCRAgent uses pytesseract via this controller.
        # project_id is empty here because OCRAgent calls pytesseract directly
        # on image file paths that are independent of any project directory.
        process_controller = ProcessController(project_id="")

        # Memory store — wraps nlp_controller for semantic memory operations
        memory_store = MemoryStore(
            nlp_controller=nlp_controller,
            async_session_maker=db_client  # <--- تمرير الـ Session Maker هنا
        )

        # -- Instantiate all agents -------------------------------------
        router_agent = RouterAgent(
            llm_provider=generation_client,
            template_parser=template_parser,
        )

        retrieval_agent = RetrievalAgent(
            embedding_client=embedding_client,
            vectordb_client=vectordb_client,
            nlp_controller=nlp_controller,
            reranker_client=reranker_client,
        )

        memory_agent = MemoryAgent(
            memory_store=memory_store,
        )

        ocr_agent = OCRAgent(
            process_controller=process_controller,
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
            template_parser=template_parser,
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
        )
