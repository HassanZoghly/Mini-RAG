from agents.base import BaseAgent, AgentState
from typing import Dict, Any
import json


class RouterAgent(BaseAgent):
    """
    Lightweight routing agent that inspects the incoming query and attached
    assets to decide which downstream agents are required.

    The routing decision is made entirely via LLM semantic understanding,
    removing keyword heuristics.

    The agent writes its decisions into ``state["metadata"]``:

    * ``route``         – 'smalltalk', 'retrieval', 'direct', 'quiz'
    * ``mode``          – 'summary', 'explain', 'qa'
    * ``needs_retrieval`` – True/False
    * ``needs_vision``  – True when image paths are present.
    """

    def __init__(self, llm_provider, template_parser) -> None:
        self._llm_provider = llm_provider
        self._template_parser = template_parser

    @property
    def agent_name(self) -> str:
        return "RouterAgent"

    async def execute(self, state: AgentState) -> AgentState:
        self.validate_state(state, ["query"])

        query: str = state["query"]
        image_paths = state.get("image_paths") or []

        # 1. اكتشاف اللغة وتوجيه الـ Parser أوتوماتيكياً
        query_lower = query.lower().strip()
        is_arabic = any('\u0600' <= char <= '\u06FF' for char in query) or "arabic" in query_lower or "عربي" in query_lower
        self._template_parser.set_language("ar" if is_arabic else "en")

        intent = self._detect_intent(query)

        if image_paths:
            intent["needs_vision"] = True

        state["metadata"].update(intent)

        route_summary = intent.get("route", "retrieval")
        mode = intent.get("mode", "qa")
        needs_retrieval = intent.get("needs_retrieval", True)

        state["agent_trace"].append(
            f"{self.agent_name}: routed query → route: {route_summary}, mode: {mode}, needs_retrieval: {needs_retrieval}"
        )

        self.log_step(f"routing decision: route={route_summary}, mode={mode}, needs_retrieval={needs_retrieval}")
        return state

    def _detect_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyse *query* with the LLM to classify intent, returning a JSON object.
        """
        system_prompt = self._template_parser.get("rag", "router_system_prompt")

        # Default fallback values
        detected_route = "retrieval"
        detected_mode = "qa"
        needs_retrieval = True

        try:
            llm_response = self._llm_provider.generate_text(
                prompt=f"Query: {query}\n\nRespond with strictly valid JSON only.",
                chat_history=[
                    self._llm_provider.construct_prompt(prompt=system_prompt, role=self._llm_provider.enums.SYSTEM.value)
                ]
            )

            import re
            json_match = re.search(r'\{.*\}', llm_response or "", re.DOTALL)
            if json_match:
                clean_json = json_match.group(0)
            else:
                clean_json = (llm_response or "").strip()

            parsed = json.loads(clean_json)
            detected_route = parsed.get("route", "retrieval").strip().lower()
            detected_mode = parsed.get("mode", "qa").strip().lower()
            
            nr = parsed.get("needs_retrieval", True)
            if isinstance(nr, str):
                needs_retrieval = nr.lower() == "true"
            else:
                needs_retrieval = bool(nr)

            self.log_step(f"LLM intent output: {parsed}")
        except Exception as exc:
            self.log_step(f"JSON intent detection failed or parsing error: {exc}. Defaulting to retrieval/qa/true.")
            detected_route = "retrieval"
            detected_mode = "qa"
            needs_retrieval = True

        valid_routes = {"smalltalk", "retrieval", "direct"}
        if detected_route == "small_talk":
            detected_route = "smalltalk"
        if detected_route not in valid_routes:
            detected_route = "retrieval"

        valid_modes = {"summary", "explain", "qa"}
        if detected_mode not in valid_modes:
            detected_mode = "qa"

        return {
            "route": detected_route,
            "mode": detected_mode,
            "needs_retrieval": needs_retrieval,
        }

