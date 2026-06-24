from agents.base import BaseAgent, AgentState
from typing import Dict, Any
import json


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

    def __init__(self, llm_provider, template_parser) -> None:
        self._llm_provider = llm_provider
        self._template_parser = template_parser

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

        route_summary = intent.get("route", "retrieval")
        confidence = intent.get("confidence", 0.0)

        state["agent_trace"].append(
            f"{self.agent_name}: routed query → {route_summary} (confidence: {confidence:.2f})"
        )

        self.log_step(f"routing decision: {route_summary} (confidence: {confidence:.2f})")
        return state

    # ------------------------------------------------------------------
    # Intent detection
    # ------------------------------------------------------------------

    def _detect_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyse *query* with the LLM to classify educational vs conversational intent,
        returning a JSON object with confidence scoring.

        Returns
        -------
        dict
            A mapping with boolean capability flags, a 'route' string, and 'confidence'.
        """
        normalised = query.lower().strip()
        tokens = set(normalised.split())

        # Check for vision/ocr via heuristics
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

        system_prompt = self._template_parser.get("rag", "router_system_prompt")
        user_prompt = self._template_parser.get("rag", "router_user_prompt", {"query": query})

        try:
            llm_response = self._llm_provider.generate_text(
                prompt=user_prompt,
                chat_history=[
                    self._llm_provider.construct_prompt(prompt=system_prompt, role=self._llm_provider.enums.SYSTEM.value)
                ]
            )

            # Clean possible markdown wrapping from LLM response
            clean_json = (llm_response or "").strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:]
            if clean_json.startswith("```"):
                clean_json = clean_json[3:]
            if clean_json.endswith("```"):
                clean_json = clean_json[:-3]
            clean_json = clean_json.strip()

            parsed = json.loads(clean_json)
            detected_category = parsed.get("intent", "retrieval").strip().lower()
            confidence = float(parsed.get("confidence", 0.0))

            self.log_step(f"LLM intent output: {parsed}")
        except Exception as exc:
            self.log_step(f"JSON intent detection failed or parsing error: {exc}. Defaulting to retrieval.")
            detected_category = "retrieval"
            confidence = 0.0

        # Enforce Minimum Confidence Threshold
        MIN_CONFIDENCE = 0.70
        if confidence < MIN_CONFIDENCE:
            self.log_step(f"Confidence {confidence:.2f} < {MIN_CONFIDENCE}. Fallback to retrieval.")
            detected_category = "retrieval"

        # Map to valid routes
        valid_routes = {"small_talk", "retrieval", "memory", "multimodal", "reasoning"}
        if detected_category not in valid_routes:
            detected_category = "retrieval"

        return {
            "route": detected_category,
            "confidence": confidence,
            "needs_vision": False,
            "needs_ocr": needs_ocr or detected_category == "multimodal",
            "needs_memory": needs_memory or detected_category == "memory",
        }