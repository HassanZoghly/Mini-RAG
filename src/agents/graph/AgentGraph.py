"""
AgentGraph — Phase 4: QueryRewriterAgent wired into all four graph builders.

New topology (all graphs):

    router ──┬─(small_talk)──► smalltalk ──────────────────────────────────────► response ──► END
             ├─(needs_vision)──► memory ──► query_rewrite ──► ocr ──► vision ──► retrieval ──► reasoning ──► response ──► END
             └─(default)────────► memory ──► query_rewrite ──► retrieval ────────► reasoning ──► response ──► END

Key ordering change (item 4A):
  • ``memory`` now runs BEFORE ``retrieval`` (moved up from after retrieval)
    so that ``QueryRewriterAgent`` has ``state["memory_context"]`` populated
    when it rewrites the query.
  • ``query_rewrite`` is inserted between ``memory`` and ``retrieval``.

All four graph builders are updated:
  1. build_graph                          (run / stream)
  2. build_graph_up_to_formatter          (run_up_to_formatter)
  3. build_multimodal_graph               (run_multimodal / stream_multimodal)
  4. build_multimodal_graph_up_to_formatter (run_multimodal_up_to_formatter)
"""

from __future__ import annotations

from typing import AsyncGenerator

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.base.AgentState import AgentState
from agents.memory.MemoryAgent import MemoryAgent
from agents.multimodal.OCRAgent import OCRAgent
from agents.multimodal.VisionAgent import VisionAgent
from agents.reasoning.ReasoningAgent import ReasoningAgent
from agents.response.ResponseFormatterAgent import ResponseFormatterAgent
from agents.retrieval.QueryRewriterAgent import QueryRewriterAgent
from agents.retrieval.RetrievalAgent import RetrievalAgent
from agents.router.RouterAgent import RouterAgent
from agents.smalltalk.SmallTalkAgent import SmallTalkAgent
from agents.base.route_constants import ROUTE_SMALLTALK, ROUTE_RETRIEVAL


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def _route_after_router(state: AgentState) -> str:
    """
    Conditional routing evaluated after ``RouterAgent``.

    Small-talk bypasses memory/retrieval/reasoning entirely and goes
    straight to the small-talk node.  Everything else (including vision
    queries) goes to memory first so ``QueryRewriterAgent`` can use
    recent history.
    """
    route = state.get("metadata", {}).get("route", ROUTE_RETRIEVAL)
    if route == ROUTE_SMALLTALK:
        return "smalltalk"
    return "memory"


def _route_after_memory(state: AgentState) -> str:
    """
    Conditional routing evaluated after ``MemoryAgent``.

    Vision queries go to OCR; everything else goes to query rewriting.
    """
    needs_vision = state.get("metadata", {}).get("needs_vision", False)
    if needs_vision:
        return "ocr"
    return "query_rewrite"


def _route_decision_multimodal(state: AgentState) -> str:
    """Routing for the multimodal graph after ``RouterAgent``."""
    route = state.get("metadata", {}).get("route", ROUTE_RETRIEVAL)
    if route == ROUTE_SMALLTALK:
        return "smalltalk"
    return "memory"


def _route_after_memory_multimodal(state: AgentState) -> str:
    """Post-memory routing for the multimodal graph."""
    if state.get("image_base64") or state.get("image_paths"):
        return "ocr_multi"
    return "query_rewrite"


# ---------------------------------------------------------------------------
# AgentGraph
# ---------------------------------------------------------------------------

class AgentGraph:
    """
    LangGraph-based orchestrator that wires all Mini-RAG agents into a
    directed state graph.

    Graph topology
    --------------
    ::

        router ──┬─(small_talk)──────────────────────────────────────────────► smalltalk ──► response ──► END
                 └─(default)──► memory ──┬─(needs_vision)──► ocr ──► vision ──► query_rewrite ──► retrieval ──► reasoning ──► response ──► END
                                         └─(default)────────────────────────── query_rewrite ──► retrieval ──► reasoning ──► response ──► END
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
        query_rewriter: QueryRewriterAgent | None = None,
    ) -> None:
        self._router = router
        self._retrieval = retrieval
        self._memory = memory
        self._ocr = ocr
        self._vision = vision
        self._reasoning = reasoning
        self._response_formatter = response_formatter
        self._smalltalk = smalltalk
        # QueryRewriterAgent is optional so the graph degrades gracefully
        # when not supplied (identity pass-through).
        self._query_rewriter = query_rewriter

    # ------------------------------------------------------------------
    # Shared helper — query_rewrite node
    # ------------------------------------------------------------------

    async def _query_rewrite_node(self, state: AgentState) -> AgentState:
        """
        Graph node wrapper for ``QueryRewriterAgent``.

        When no rewriter is configured, sets ``query_for_retrieval`` to
        the original query so downstream agents always find it populated.
        """
        if self._query_rewriter is not None:
            return await self._query_rewriter.execute(state)

        # Passthrough — ensure the field exists
        if not state.get("query_for_retrieval"):
            state["query_for_retrieval"] = state.get("query", "")
        return state

    # ------------------------------------------------------------------
    # Graph 1: standard indexed-RAG graph
    # ------------------------------------------------------------------

    def build_graph(self) -> CompiledStateGraph:
        """
        Build the standard indexed-RAG pipeline graph.

        Topology:
            router → (small_talk | memory)
            memory → (ocr | query_rewrite)
            ocr → vision → query_rewrite
            query_rewrite → retrieval → reasoning → response → END
            smalltalk → response → END
        """
        graph = StateGraph(AgentState)

        graph.add_node("router",       self._router.execute)
        graph.add_node("smalltalk",    self._smalltalk.execute)
        graph.add_node("memory",       self._memory.execute)
        graph.add_node("ocr",          self._ocr.execute)
        graph.add_node("vision",       self._vision.execute)
        graph.add_node("query_rewrite",self._query_rewrite_node)
        graph.add_node("retrieval",    self._retrieval.execute)
        graph.add_node("reasoning",    self._reasoning.execute)
        graph.add_node("response",     self._response_formatter.execute)

        graph.set_entry_point("router")

        graph.add_conditional_edges(
            "router", _route_after_router,
            {"smalltalk": "smalltalk", "memory": "memory"},
        )

        graph.add_conditional_edges(
            "memory", _route_after_memory,
            {"ocr": "ocr", "query_rewrite": "query_rewrite"},
        )

        graph.add_edge("ocr",           "vision")
        graph.add_edge("vision",        "query_rewrite")
        graph.add_edge("query_rewrite", "retrieval")
        graph.add_edge("retrieval",     "reasoning")
        graph.add_edge("reasoning",     "response")
        graph.add_edge("smalltalk",     "response")
        graph.add_edge("response",      END)

        return graph.compile()

    def build_graph_up_to_formatter(self) -> CompiledStateGraph:
        """
        Same as ``build_graph`` but ends after ``reasoning`` (before
        ``response``). Used by streaming endpoints that call
        ``ResponseFormatterAgent.stream_execute`` directly.
        """
        graph = StateGraph(AgentState)

        graph.add_node("router",       self._router.execute)
        graph.add_node("smalltalk",    self._smalltalk.execute)
        graph.add_node("memory",       self._memory.execute)
        graph.add_node("ocr",          self._ocr.execute)
        graph.add_node("vision",       self._vision.execute)
        graph.add_node("query_rewrite",self._query_rewrite_node)
        graph.add_node("retrieval",    self._retrieval.execute)
        graph.add_node("reasoning",    self._reasoning.execute)

        graph.set_entry_point("router")

        graph.add_conditional_edges(
            "router", _route_after_router,
            {"smalltalk": "smalltalk", "memory": "memory"},
        )

        graph.add_conditional_edges(
            "memory", _route_after_memory,
            {"ocr": "ocr", "query_rewrite": "query_rewrite"},
        )

        graph.add_edge("ocr",           "vision")
        graph.add_edge("vision",        "query_rewrite")
        graph.add_edge("query_rewrite", "retrieval")
        graph.add_edge("retrieval",     "reasoning")
        graph.add_edge("reasoning",     END)
        graph.add_edge("smalltalk",     END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Graph 3 & 4: multimodal graph
    # ------------------------------------------------------------------

    def build_multimodal_graph(self) -> CompiledStateGraph:
        """
        Multimodal pipeline (images + indexed chunks).

        Topology:
            router → (small_talk | memory)
            memory → (ocr_multi | query_rewrite)
            ocr_multi → vision_multi → query_rewrite
            query_rewrite → retrieval → memory_multi → reasoning_multi → response → END
            smalltalk → response → END
        """
        graph = StateGraph(AgentState)

        graph.add_node("router",          self._router.execute)
        graph.add_node("smalltalk",       self._smalltalk.execute)
        graph.add_node("memory",          self._memory.execute)
        graph.add_node("ocr_multi",       self._ocr.execute)
        graph.add_node("vision_multi",    self._vision.execute)
        graph.add_node("query_rewrite",   self._query_rewrite_node)
        graph.add_node("retrieval",       self._retrieval.execute)
        graph.add_node("reasoning_multi", self._reasoning.execute_multimodal)
        graph.add_node("response",        self._response_formatter.execute)

        graph.set_entry_point("router")

        graph.add_conditional_edges(
            "router", _route_decision_multimodal,
            {"smalltalk": "smalltalk", "memory": "memory"},
        )

        graph.add_conditional_edges(
            "memory", _route_after_memory_multimodal,
            {"ocr_multi": "ocr_multi", "query_rewrite": "query_rewrite"},
        )

        graph.add_edge("ocr_multi",       "vision_multi")
        graph.add_edge("vision_multi",    "query_rewrite")
        graph.add_edge("query_rewrite",   "retrieval")
        graph.add_edge("retrieval",       "reasoning_multi")
        graph.add_edge("reasoning_multi", "response")
        graph.add_edge("smalltalk",       "response")
        graph.add_edge("response",        END)

        return graph.compile()

    def build_multimodal_graph_up_to_formatter(self) -> CompiledStateGraph:
        """
        Multimodal graph ending after ``reasoning_multi`` (before response).
        """
        graph = StateGraph(AgentState)

        graph.add_node("router",          self._router.execute)
        graph.add_node("smalltalk",       self._smalltalk.execute)
        graph.add_node("memory",          self._memory.execute)
        graph.add_node("ocr_multi",       self._ocr.execute)
        graph.add_node("vision_multi",    self._vision.execute)
        graph.add_node("query_rewrite",   self._query_rewrite_node)
        graph.add_node("retrieval",       self._retrieval.execute)
        graph.add_node("reasoning_multi", self._reasoning.execute_multimodal)

        graph.set_entry_point("router")

        graph.add_conditional_edges(
            "router", _route_decision_multimodal,
            {"smalltalk": "smalltalk", "memory": "memory"},
        )

        graph.add_conditional_edges(
            "memory", _route_after_memory_multimodal,
            {"ocr_multi": "ocr_multi", "query_rewrite": "query_rewrite"},
        )

        graph.add_edge("ocr_multi",       "vision_multi")
        graph.add_edge("vision_multi",    "query_rewrite")
        graph.add_edge("query_rewrite",   "retrieval")
        graph.add_edge("retrieval",       "reasoning_multi")
        graph.add_edge("reasoning_multi", END)
        graph.add_edge("smalltalk",       END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    async def run(self, initial_state: AgentState) -> AgentState:
        compiled = self.build_graph()
        return await compiled.ainvoke(initial_state)

    async def stream(self, initial_state: AgentState) -> AsyncGenerator[dict, None]:
        compiled = self.build_graph()
        async for chunk in compiled.astream(initial_state):
            yield chunk

    async def run_up_to_formatter(self, initial_state: AgentState) -> AgentState:
        compiled = self.build_graph_up_to_formatter()
        return await compiled.ainvoke(initial_state)

    async def run_multimodal(self, initial_state: AgentState) -> AgentState:
        compiled = self.build_multimodal_graph()
        return await compiled.ainvoke(initial_state)

    async def stream_multimodal(self, initial_state: AgentState) -> AsyncGenerator[dict, None]:
        compiled = self.build_multimodal_graph()
        async for chunk in compiled.astream(initial_state):
            yield chunk

    async def run_multimodal_up_to_formatter(self, initial_state: AgentState) -> AgentState:
        compiled = self.build_multimodal_graph_up_to_formatter()
        return await compiled.ainvoke(initial_state)
