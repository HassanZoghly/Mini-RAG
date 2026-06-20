from agents.base import BaseAgent, AgentState

class SmallTalkAgent(BaseAgent):
    def __init__(self, llm_provider, template_parser) -> None:
        self._llm = llm_provider
        self._template_parser = template_parser

    @property
    def agent_name(self) -> str:
        return "SmallTalkAgent"

    async def execute(self, state: AgentState) -> AgentState:
        self.validate_state(state, ["query"])
        query: str = state["query"]
        self.log_step(f"handling casual chat for query: '{query[:60]}'")

        prompt = self._template_parser.get("rag", "smalltalk_prompt", {"query": query})

        try:
            answer = self._llm.generate_text(prompt=prompt, chat_history=[])
            state["reasoning_context"] = answer or "Hello! How can I help you today?"
        except Exception as exc:
            self.log_step(f"SmallTalk generation failed: {exc}")
            state["reasoning_context"] = "Hello! How can I help you today?"

        state["agent_trace"].append(f"{self.agent_name}: handled casual chat directly")
        return state