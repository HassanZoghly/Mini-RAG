from abc import ABC, abstractmethod
from typing import List
import logging


class BaseAgent(ABC):
    """
    Abstract base class that every agent in the Mini-RAG agent layer must inherit.

    Subclasses are required to implement:
      - ``agent_name``  (property) – a human-readable identifier for the agent.
      - ``execute``     (async method) – the agent's core processing logic.

    Concrete helpers available to all subclasses:
      - ``log_step``       – emits a structured log line prefixed with the agent name.
      - ``validate_state`` – asserts that a set of required keys are present in the
                             shared pipeline state dict, raising ``ValueError`` otherwise.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """
        A short, human-readable name that uniquely identifies this agent.

        Used as a prefix in log messages and appended to ``agent_trace``
        entries inside the pipeline state.

        Returns
        -------
        str
            The agent's display name (e.g. ``"RetrieverAgent"``).
        """

    @abstractmethod
    async def execute(self, state: dict) -> dict:
        """
        Execute the agent's core logic against the current pipeline state.

        Each agent receives the shared mutable ``state`` dict (an
        ``AgentState``-shaped mapping), performs its work, updates the
        relevant keys, appends a trace entry to ``state["agent_trace"]``,
        and returns the updated state.

        Parameters
        ----------
        state : dict
            The current pipeline state produced by previous agents or
            initialised via ``create_initial_state``.

        Returns
        -------
        dict
            The updated pipeline state with this agent's contributions.
        """

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def log_step(self, message: str) -> None:
        """
        Emit an INFO-level log line prefixed with the agent's name.

        The log record is written to the logger named after the fully
        qualified module of the concrete subclass, consistent with the
        rest of the Mini-RAG codebase.

        Parameters
        ----------
        message : str
            Descriptive message to include after the agent-name prefix.
        """
        logger = logging.getLogger(type(self).__module__)
        logger.info("[%s] %s", self.agent_name, message)

    def validate_state(self, state: dict, required_keys: List[str]) -> bool:
        """
        Assert that every key in *required_keys* is present in *state*.

        Parameters
        ----------
        state : dict
            The pipeline state dict to inspect.
        required_keys : List[str]
            Keys that must exist in *state* before the agent can proceed.

        Returns
        -------
        bool
            ``True`` when all required keys are present.

        Raises
        ------
        ValueError
            If one or more required keys are missing from *state*, with a
            message that lists every missing key.
        """
        missing = [key for key in required_keys if key not in state]

        if missing:
            raise ValueError(
                f"[{self.agent_name}] Missing required state keys: {missing}"
            )

        return True
