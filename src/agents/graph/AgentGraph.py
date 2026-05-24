from typing import AsyncGenerator

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from agents.base.AgentState import AgentState
from agents.router.RouterAgent import RouterAgent
from agents.retrieval.RetrievalAgent import RetrievalAgent
from agents.memory.MemoryAgent import MemoryAgent
from agents.multimodal.OCRAgent import OCRAgent
from agents.multimodal.VisionAgent import VisionAgent
from agents.reasoning.ReasoningAgent import ReasoningAgent
from agents.response.ResponseFormatterAgent import ResponseFormatterAgent
from agents.smalltalk.SmallTalkAgent import SmallTalkAgent


def _route_after_router(state: AgentState) -> str:
    """
    Conditional routing function evaluated after the ``RouterAgent`` node.
    """
    route = state.get("metadata", {}).get("route", "retrieval")
    needs_vision = state.get("metadata", {}).get("needs_vision", False)

    if route == "small_talk":
        return "smalltalk"
    if needs_vision:
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
        response_formatter: ResponseFormatterAgent,
        smalltalk: SmallTalkAgent,
    ) -> None:
        self._router = router
        self._retrieval = retrieval
        self._memory = memory
        self._ocr = ocr
        self._vision = vision
        self._reasoning = reasoning
        self._response_formatter = response_formatter
        self._smalltalk = smalltalk

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
        graph.add_node("response",  self._response_formatter.execute)
        graph.add_node("smalltalk", self._smalltalk.execute)

        # -- Entry point ---------------------------------------------------
        graph.set_entry_point("router")

        # -- Conditional edges from router ---------------------------------
        graph.add_conditional_edges(
            "router",
            _route_after_router,
            {
                "smalltalk": "smalltalk",
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
        graph.add_edge("smalltalk", "response")
        graph.add_edge("response",  END)

        return graph.compile()

    def build_graph_up_to_formatter(self):
        # Build the exact same graph but without the `response` node, or simply return compiled
        # Since we use `.stream` with ResponseFormatterAgent outside the graph
        # Actually LangGraph doesn't allow easily running "up to a node".
        # A simple hack: just set END after reasoning and smalltalk.
        graph = StateGraph(AgentState)
        graph.add_node("router",    self._router.execute)
        graph.add_node("retrieval", self._retrieval.execute)
        graph.add_node("memory",    self._memory.execute)
        graph.add_node("ocr",       self._ocr.execute)
        graph.add_node("vision",    self._vision.execute)
        graph.add_node("reasoning", self._reasoning.execute)
        graph.add_node("smalltalk", self._smalltalk.execute)

        graph.set_entry_point("router")
        graph.add_conditional_edges(
            "router", _route_after_router,
            {"smalltalk": "smalltalk", "ocr": "ocr", "retrieval": "retrieval"}
        )
        graph.add_edge("ocr", "vision")
        graph.add_edge("vision", "retrieval")
        graph.add_edge("retrieval", "memory")
        graph.add_edge("memory", "reasoning")
        graph.add_edge("reasoning", END)
        graph.add_edge("smalltalk", END)
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
        compiled = self.build_graph()
        async for chunk in compiled.astream(initial_state):
            yield chunk

    async def run_up_to_formatter(self, initial_state: AgentState) -> AgentState:
        compiled = self.build_graph_up_to_formatter()
        result = await compiled.ainvoke(initial_state)
        return result

    def build_multimodal_graph(self) -> CompiledStateGraph:
        graph = StateGraph(AgentState)

        graph.add_node("router",         self._router.execute)
        graph.add_node("smalltalk",      self._smalltalk.execute)
        graph.add_node("ocr_multi",      self._ocr.execute)
        graph.add_node("vision_multi",   self._vision.execute)
        graph.add_node("retrieval",      self._retrieval.execute)
        graph.add_node("memory",         self._memory.execute)
        graph.add_node("reasoning_multi",self._reasoning.execute_multimodal)
        graph.add_node("response",       self._response_formatter.execute)

        graph.set_entry_point("router")

        def route_decision(state: AgentState) -> str:
            route = state.get("metadata", {}).get("route", "retrieval")
            if route == "small_talk":
                return "smalltalk"
            if state.get("image_base64") or state.get("image_paths"):
                return "ocr_multi"
            return "retrieval"

        graph.add_conditional_edges("router", route_decision, {
            "smalltalk": "smalltalk",
            "ocr_multi": "ocr_multi",
            "retrieval": "retrieval",
        })

        graph.add_edge("smalltalk",      "response")
        graph.add_edge("ocr_multi",      "vision_multi")
        graph.add_edge("vision_multi",   "retrieval")
        graph.add_edge("retrieval",      "memory")
        graph.add_edge("memory",         "reasoning_multi")
        graph.add_edge("reasoning_multi","response")
        graph.add_edge("response",       END)

        return graph.compile()

    async def run_multimodal(self, initial_state: AgentState) -> AgentState:
        """Compiles and runs the multimodal graph."""
        compiled = self.build_multimodal_graph()
        result = await compiled.ainvoke(initial_state)
        return result

    async def stream_multimodal(self, initial_state: AgentState) -> AsyncGenerator[dict, None]:
        """Compiles and streams the multimodal graph."""
        compiled = self.build_multimodal_graph()
        async for chunk in compiled.astream(initial_state):
            yield chunk

    def build_multimodal_graph_up_to_formatter(self) -> CompiledStateGraph:
        graph = StateGraph(AgentState)

        graph.add_node("router",         self._router.execute)
        graph.add_node("smalltalk",      self._smalltalk.execute)
        graph.add_node("ocr_multi",      self._ocr.execute)
        graph.add_node("vision_multi",   self._vision.execute)
        graph.add_node("retrieval",      self._retrieval.execute)
        graph.add_node("memory",         self._memory.execute)
        graph.add_node("reasoning_multi",self._reasoning.execute_multimodal)

        graph.set_entry_point("router")

        def route_decision(state: AgentState) -> str:
            route = state.get("metadata", {}).get("route", "retrieval")
            if route == "small_talk":
                return "smalltalk"
            if state.get("image_base64") or state.get("image_paths"):
                return "ocr_multi"
            return "retrieval"

        graph.add_conditional_edges("router", route_decision, {
            "smalltalk": "smalltalk",
            "ocr_multi": "ocr_multi",
            "retrieval": "retrieval",
        })

        graph.add_edge("smalltalk",      END)
        graph.add_edge("ocr_multi",      "vision_multi")
        graph.add_edge("vision_multi",   "retrieval")
        graph.add_edge("retrieval",      "memory")
        graph.add_edge("memory",         "reasoning_multi")
        graph.add_edge("reasoning_multi",END)

        return graph.compile()

    async def run_multimodal_up_to_formatter(self, initial_state: AgentState) -> AgentState:
        compiled = self.build_multimodal_graph_up_to_formatter()
        result = await compiled.ainvoke(initial_state)
        return result
