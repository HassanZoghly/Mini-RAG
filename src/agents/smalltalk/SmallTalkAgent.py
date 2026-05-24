from agents.base import BaseAgent, AgentState

class SmallTalkAgent(BaseAgent):
    """
    Agent responsible for handling small talk, greetings, and casual questions.
    It writes directly to state["reasoning_context"] or state["final_response"] 
    and bypasses the complex retrieval and reasoning layers.
    """

    def __init__(self, llm_provider) -> None:
        self._llm = llm_provider

    @property
    def agent_name(self) -> str:
        return "SmallTalkAgent"

    async def execute(self, state: AgentState) -> AgentState:
        self.validate_state(state, ["query"])

        query: str = state["query"]
        self.log_step(f"handling casual chat for query: '{query[:60]}'")

        prompt = (
            "You are a friendly, helpful AI assistant. The user just said something casual or a greeting. "
            "Respond naturally, briefly, and politely.\n\n"
            f"User: {query}\nResponse:"
        )

        try:
            answer = self._llm.generate_text(
                prompt=prompt,
                chat_history=[],
            )
            # We put the answer directly in reasoning_context so the ResponseFormatterAgent
            # can pick it up. Or we can put it in final_response if ResponseFormatterAgent handles it.
            state["reasoning_context"] = answer or "Hello! How can I help you today?"
        except Exception as exc:
            self.log_step(f"SmallTalk generation failed: {exc}")
            state["reasoning_context"] = "Hello! How can I help you today?"

        state["agent_trace"].append(f"{self.agent_name}: handled casual chat directly")
        return state
