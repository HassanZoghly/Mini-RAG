from typing import AsyncGenerator

from agents.base import BaseAgent, AgentState


class ResponseAgent(BaseAgent):
    """
    Agent that calls the LLM to produce the final natural-language answer.

    It follows the same prompt-construction pattern as
    ``NLPController.answer_rag_question``:

    1. Fetch the system prompt from ``template_parser``.
    2. Build a document-prompt block from ``state["reasoning_context"]``
       (treated as a single pre-assembled document).
    3. Build a footer prompt that embeds the user's query.
    4. Combine into ``full_prompt`` and call the LLM.

    Two execution modes are provided:

    * ``execute`` — blocking call; writes the complete answer to
      ``state["final_response"]`` and returns the state.
    * ``stream_execute`` — async generator that yields tokens as they
      arrive from the LLM, mirroring the pattern in
      ``NLPController.answer_rag_question_stream``.

    Parameters
    ----------
    llm_provider : object
        LLM provider conforming to ``LLMInterface``.  Must implement
        ``generate_text``, ``generate_stream``, ``construct_prompt``,
        and expose an ``enums`` attribute with a ``SYSTEM`` value.
    template_parser : object
        ``TemplateParser`` instance used to render prompts from the
        ``rag`` template group.
    """

    def __init__(self, llm_provider, template_parser) -> None:
        self._llm = llm_provider
        self._template_parser = template_parser

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def agent_name(self) -> str:
        """Return the display name used in logs and agent trace entries."""
        return "ResponseAgent"

    async def execute(self, state: AgentState) -> AgentState:
        """
        Generate the final answer and write it to ``state["final_response"]``.

        Steps
        -----
        1. Validate that ``query`` and ``reasoning_context`` are present.
        2. Build the prompt following the NLPController RAG pattern:
           system prompt → document block → footer with the query.
        3. Call ``llm_provider.generate_text`` to get the full response.
        4. Write to ``state["final_response"]``.
        5. Append trace and return the updated state.

        Parameters
        ----------
        state : AgentState
            Current pipeline state.  Must contain ``query`` and
            ``reasoning_context``.

        Returns
        -------
        AgentState
            Updated state with ``final_response`` populated.
        """
        self.validate_state(state, ["query", "reasoning_context"])

        full_prompt, chat_history = self._build_prompt(state)

        self.log_step(
            f"calling LLM for query='{state['query'][:60]}' "
            f"({len(full_prompt)} prompt chars)"
        )

        try:
            answer = self._llm.generate_text(
                prompt=full_prompt,
                chat_history=chat_history,
            )
            state["final_response"] = answer or ""
        except Exception as exc:
            self.log_step(f"LLM generation failed: {exc}")
            state["final_response"] = ""
            state["error"] = str(exc)

        state["agent_trace"].append(
            f"{self.agent_name}: generated final answer"
        )
        self.log_step(
            f"final_response length={len(state['final_response'])} chars"
        )
        return state

    async def stream_execute(
        self, state: AgentState
    ) -> AsyncGenerator[str, None]:
        """
        Streaming variant: yields answer tokens as they arrive from the LLM.

        Uses the same prompt-building logic as ``execute`` but calls
        ``llm_provider.generate_stream`` instead of ``generate_text``,
        matching the pattern in ``NLPController.answer_rag_question_stream``.

        When the context is empty (no retrieved chunks, no memory, etc.)
        a single error message is yielded and the generator returns early.

        Parameters
        ----------
        state : AgentState
            Current pipeline state.  Must contain ``query`` and
            ``reasoning_context``.

        Yields
        ------
        str
            Successive text tokens from the LLM.
        """
        self.validate_state(state, ["query", "reasoning_context"])

        reasoning_context: str = state.get("reasoning_context", "").strip()
        if not reasoning_context:
            yield "I could not find enough context to answer your question."
            return

        full_prompt, chat_history = self._build_prompt(state)

        self.log_step(
            f"streaming LLM response for query='{state['query'][:60]}'"
        )

        try:
            for chunk in self._llm.generate_stream(
                prompt=full_prompt,
                chat_history=chat_history,
            ):
                yield chunk
        except Exception as exc:
            self.log_step(f"LLM streaming failed: {exc}")
            yield "An error occurred while generating the response."

        state["agent_trace"].append(
            f"{self.agent_name}: generated final answer (streaming)"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, state: AgentState):
        """
        Construct the system prompt, document block, and footer following
        the NLPController RAG pattern.

        The ``reasoning_context`` assembled by ``ReasoningAgent`` is treated
        as a single pre-formatted document chunk.  This avoids duplicating
        chunk-iteration logic that is already handled upstream.

        Parameters
        ----------
        state : AgentState
            Pipeline state providing ``query`` and ``reasoning_context``.

        Returns
        -------
        tuple[str, list]
            ``(full_prompt, chat_history)`` ready to pass to the LLM provider.
        """
        query: str = state["query"]
        reasoning_context: str = state.get("reasoning_context", "").strip()

        # -- System prompt -------------------------------------------------
        system_prompt = self._template_parser.get("rag", "system_prompt")
        chat_history = [
            self._llm.construct_prompt(
                prompt=system_prompt,
                role=self._llm.enums.SYSTEM.value,
            )
        ]

        # -- Document block ------------------------------------------------
        if reasoning_context:
            document_block = self._template_parser.get(
                "rag",
                "document_prompt",
                {"doc_num": 1, "chunk_text": reasoning_context},
            )
        else:
            document_block = "(No relevant context was found.)"

        # -- Footer --------------------------------------------------------
        footer = self._template_parser.get(
            "rag", "footer_prompt", {"query": query}
        )

        full_prompt = "\n\n".join([document_block, footer])
        return full_prompt, chat_history
