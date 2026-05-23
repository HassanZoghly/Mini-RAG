from agents.base import BaseAgent, AgentState
from typing import Dict


# Keywords that suggest the user's query is about content extracted via OCR.
_OCR_KEYWORDS: frozenset = frozenset({
    "ocr", "scan", "scanned", "handwritten", "handwriting",
    "printed text", "image text", "extract text", "read text",
    "text from image", "document scan",
})


class RouterAgent(BaseAgent):
    """
    Lightweight routing agent that inspects the incoming query and attached
    assets to decide which downstream agents are required.

    The routing decision is made entirely with deterministic heuristics —
    no LLM call is performed — keeping latency and cost at zero for this
    step.

    The agent writes its decisions into ``state["metadata"]`` as boolean
    flags that downstream agents read:

    * ``needs_vision``  – ``True`` when image paths are present.
    * ``needs_ocr``     – ``True`` when the query suggests OCR extraction.
    * ``needs_memory``  – ``True`` when memory/history keywords are found.
    * ``route``         – ``"retrieval_only"`` for plain text queries with
                          no special requirements.

    Parameters
    ----------
    llm_provider : object
        Reserved for future use (e.g. LLM-assisted intent classification).
        Accepted but not used in the current heuristic implementation.
    """

    def __init__(self, llm_provider) -> None:
        self._llm_provider = llm_provider

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def agent_name(self) -> str:
        """Return the display name used in logs and agent trace entries."""
        return "RouterAgent"

    async def execute(self, state: AgentState) -> AgentState:
        """
        Analyse the query and asset context, then annotate *state* with
        routing decisions.

        Steps
        -----
        1. Validate that ``query`` is present in *state*.
        2. Call ``_detect_intent`` to determine which capabilities are needed.
        3. Merge the intent flags into ``state["metadata"]``.
        4. Set ``state["metadata"]["route"] = "retrieval_only"`` when no
           special capability is required.
        5. Append a trace entry describing the routing decision and return
           the updated state.

        Parameters
        ----------
        state : AgentState
            Current pipeline state.  Must contain at least ``query``.

        Returns
        -------
        AgentState
            Updated state with routing flags written into ``metadata``.
        """
        self.validate_state(state, ["query"])

        query: str = state["query"]
        image_paths = state.get("image_paths") or []

        intent = self._detect_intent(query)

        # image_paths in state override the query-level vision flag
        if image_paths:
            intent["needs_vision"] = True

        state["metadata"].update(intent)

        # Derive a human-readable route label for convenience
        if not any(intent.values()):
            state["metadata"]["route"] = "retrieval_only"

        route_summary = ", ".join(
            k for k, v in intent.items() if v
        ) or "retrieval_only"

        state["agent_trace"].append(
            f"{self.agent_name}: routed query → {route_summary}"
        )

        self.log_step(f"routing decision: {route_summary}")
        return state

    # ------------------------------------------------------------------
    # Intent detection
    # ------------------------------------------------------------------

    def _detect_intent(self, query: str) -> Dict[str, bool]:
        """
        Analyse *query* with lightweight heuristics and return a dict of
        boolean capability flags.

        Parameters
        ----------
        query : str
            The raw natural-language query from the user.

        Returns
        -------
        dict
            A mapping with the following keys, each with a ``bool`` value:

            * ``needs_vision``  – image-understanding required.
            * ``needs_ocr``     – OCR text extraction required.
            * ``needs_memory``  – conversation history required.
        """
        normalised = query.lower().strip()
        tokens = set(normalised.split())

        needs_ocr = bool(_OCR_KEYWORDS & tokens) or any(
            phrase in normalised for phrase in _OCR_KEYWORDS if " " in phrase
        )

        needs_memory = any(
            kw in normalised
            for kw in (
                "previous", "earlier", "last time", "you said",
                "remember", "recall", "history", "before",
                "as i mentioned", "conversation",
            )
        )

        # Vision flag is set to False here; execute() will override it
        # when state["image_paths"] is non-empty.
        return {
            "needs_vision": False,
            "needs_ocr": needs_ocr,
            "needs_memory": needs_memory,
        }
