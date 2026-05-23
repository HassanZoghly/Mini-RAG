from typing import AsyncGenerator

from langgraph.graph import StateGraph, END

from agents.base.AgentState import AgentState
from agents.router.RouterAgent import RouterAgent
from agents.retrieval.RetrievalAgent import RetrievalAgent
from agents.memory.MemoryAgent import MemoryAgent
from agents.multimodal.OCRAgent import OCRAgent
from agents.multimodal.VisionAgent import VisionAgent
from agents.reasoning.ReasoningAgent import ReasoningAgent
from agents.response.ResponseAgent import ResponseAgent


def _route_after_router(state: AgentState) -> str:
    """
    Conditional routing function evaluated after the ``RouterAgent`` node.

    Reads the ``needs_vision`` flag written by ``RouterAgent`` into
    ``state["metadata"]`` and returns the name of the next node to execute.

    Parameters
    ----------
    state : AgentState
        Current pipeline state with ``metadata`` already populated by
        ``RouterAgent``.

    Returns
    -------
    str
        ``"ocr"`` when the query involves images that require OCR/vision
        processing, otherwise ``"retrieval"`` for the standard text path.
    """
    if state.get("metadata", {}).get("needs_vision", False):
        return "ocr"
    return "retrieval"


class AgentGraph:
    """
    LangGraph-based orchestrator that wires all 7 Mini-RAG agents into a
    directed state graph and executes them as an async pipeline.

    Graph topology
    --------------
    ::

        router ──┬─(needs_vision)──► ocr ──► vision ──► retrieval ──► memory ──► reasoning ──► response ──► END
                 └─(default)───────────────────────────► retrieval ──┘

    The ``RouterAgent`` inspects the query and image paths and sets flags
    in ``state["metadata"]``; the conditional edge reads ``needs_vision``
    to decide whether to insert the OCR + Vision leg before retrieval.

    Parameters
    ----------
    router : RouterAgent
    retrieval : RetrievalAgent
    memory : MemoryAgent
    ocr : OCRAgent
    vision : VisionAgent
    reasoning : ReasoningAgent
    response : ResponseAgent
    """

    def __init__(
        self,
        router: RouterAgent,
        retrieval: RetrievalAgent,
        memory: MemoryAgent,
        ocr: OCRAgent,
        vision: VisionAgent,
        reasoning: ReasoningAgent,
        response: ResponseAgent,
    ) -> None:
        self._router = router
        self._retrieval = retrieval
        self._memory = memory
        self._ocr = ocr
        self._vision = vision
        self._reasoning = reasoning
        self._response = response

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_graph(self):
        """
        Construct and compile the LangGraph ``StateGraph``.

        Returns
        -------
        CompiledGraph
            A compiled, runnable LangGraph graph ready for ``ainvoke``
            or ``astream``.
        """
        graph = StateGraph(AgentState)

        # -- Nodes ---------------------------------------------------------
        graph.add_node("router",    self._router.execute)
        graph.add_node("retrieval", self._retrieval.execute)
        graph.add_node("memory",    self._memory.execute)
        graph.add_node("ocr",       self._ocr.execute)
        graph.add_node("vision",    self._vision.execute)
        graph.add_node("reasoning", self._reasoning.execute)
        graph.add_node("response",  self._response.execute)

        # -- Entry point ---------------------------------------------------
        graph.set_entry_point("router")

        # -- Conditional edges from router ---------------------------------
        # needs_vision=True  →  ocr → vision → retrieval → memory → reasoning → response
        # needs_vision=False →              retrieval → memory → reasoning → response
        graph.add_conditional_edges(
            "router",
            _route_after_router,
            {
                "ocr":       "ocr",
                "retrieval": "retrieval",
            },
        )

        # -- Vision path edges ---------------------------------------------
        graph.add_edge("ocr",    "vision")
        graph.add_edge("vision", "retrieval")

        # -- Common path edges ---------------------------------------------
        graph.add_edge("retrieval", "memory")
        graph.add_edge("memory",    "reasoning")
        graph.add_edge("reasoning", "response")
        graph.add_edge("response",  END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    async def run(self, initial_state: AgentState) -> AgentState:
        """
        Execute the full agent pipeline and return the final state.

        Parameters
        ----------
        initial_state : AgentState
            Freshly created state from ``create_initial_state``.

        Returns
        -------
        AgentState
            The state after all agents have run.
        """
        compiled = self.build_graph()
        result = await compiled.ainvoke(initial_state)
        return result

    async def stream(
        self, initial_state: AgentState
    ) -> AsyncGenerator[dict, None]:
        """
        Execute the pipeline in streaming mode, yielding intermediate
        state snapshots as each node completes.

        Parameters
        ----------
        initial_state : AgentState
            Freshly created state from ``create_initial_state``.

        Yields
        ------
        dict
            A LangGraph chunk dict mapping node names to their output
            state updates (format: ``{"node_name": AgentState}``).
        """
        compiled = self.build_graph()
        async for chunk in compiled.astream(initial_state):
            yield chunk
