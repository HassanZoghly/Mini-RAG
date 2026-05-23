import base64
from pathlib import Path

from agents.base import BaseAgent, AgentState


class VisionAgent(BaseAgent):
    """
    Agent that generates a natural-language description of images in the
    context of the user's query.

    If the injected LLM provider exposes a multimodal / vision capability
    (detected via ``_supports_vision``), the agent sends each image plus a
    context-aware prompt to the model and writes the response to
    ``state["vision_description"]``.

    When the provider does not support vision, the agent gracefully falls
    back to whatever OCR text was already extracted by ``OCRAgent`` (i.e.
    ``state["ocr_text"]``), ensuring downstream agents always have
    *something* in ``vision_description`` to work with.

    Parameters
    ----------
    llm_provider : object
        An LLM provider instance conforming to ``LLMInterface``.  The
        provider is used for generation if it supports vision; otherwise
        the agent uses the OCR fallback.
    """

    def __init__(self, llm_provider) -> None:
        self._llm = llm_provider

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def agent_name(self) -> str:
        """Return the display name used in logs and agent trace entries."""
        return "VisionAgent"

    async def execute(self, state: AgentState) -> AgentState:
        """
        Produce a visual description of the images attached to the query.

        Steps
        -----
        1. Validate that ``image_paths`` and ``query`` are present in *state*.
        2. If no image paths are present, short-circuit with an empty
           ``vision_description``.
        3. If the LLM provider supports vision:
           a. Build the prompt: ``"Describe this image in context of: {query}"``.
           b. Encode each image as Base64 and call the provider's vision
              interface.
           c. Concatenate all descriptions and write to
              ``state["vision_description"]``.
        4. Otherwise, fall back to ``state["ocr_text"]`` so the field is
           never empty.
        5. Append trace and return the updated state.

        Parameters
        ----------
        state : AgentState
            Current pipeline state.  Must contain ``image_paths`` and ``query``.

        Returns
        -------
        AgentState
            Updated state with ``vision_description`` populated.
        """
        self.validate_state(state, ["image_paths", "query"])

        image_paths = state["image_paths"]
        query: str = state["query"]

        if not image_paths:
            state["vision_description"] = ""
            state["agent_trace"].append(
                f"{self.agent_name}: no images to describe"
            )
            self.log_step("no image paths — skipping vision description")
            return state

        if self._supports_vision(self._llm):
            descriptions = []
            prompt = f"Describe this image in context of: {query}"

            for image_path in image_paths:
                try:
                    description = self._describe_image(image_path, prompt)
                    if description:
                        descriptions.append(description.strip())
                except Exception as exc:
                    self.log_step(
                        f"WARNING: vision description failed for '{image_path}': {exc}"
                    )

            state["vision_description"] = "\n\n".join(descriptions)
            self.log_step(
                f"generated vision description for {len(descriptions)} image(s)"
            )
        else:
            # Graceful fallback: use whatever OCR text was already extracted.
            fallback = state.get("ocr_text", "")
            state["vision_description"] = fallback
            self.log_step(
                "provider does not support vision — using ocr_text as fallback"
            )

        state["agent_trace"].append(
            f"{self.agent_name}: generated visual description"
        )
        return state

    # ------------------------------------------------------------------
    # Vision capability detection
    # ------------------------------------------------------------------

    def _supports_vision(self, provider) -> bool:
        """
        Determine whether *provider* has a multimodal / vision capability.

        Detection strategy (in priority order):

        1. Check for an explicit ``supports_vision`` boolean attribute.
        2. Check for a ``describe_image`` method, the conventional name
           used by vision-capable LLM wrappers.
        3. Check for a ``vision`` attribute that evaluates to ``True``.

        Parameters
        ----------
        provider : object
            LLM provider instance to inspect.

        Returns
        -------
        bool
            ``True`` when the provider is considered vision-capable.
        """
        if hasattr(provider, "supports_vision"):
            return bool(provider.supports_vision)
        if hasattr(provider, "describe_image") and callable(
            getattr(provider, "describe_image")
        ):
            return True
        if hasattr(provider, "vision") and provider.vision:
            return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _describe_image(self, image_path: str, prompt: str) -> str:
        """
        Ask the LLM provider to describe a single image.

        The method tries two calling conventions, in order, so it works
        with different provider implementations:

        1. ``provider.describe_image(image_path, prompt)`` — preferred
           when the provider accepts a filesystem path directly.
        2. ``provider.generate_text(prompt, images=[base64_str])`` — for
           providers that accept Base64-encoded image data alongside the
           text prompt.

        Parameters
        ----------
        image_path : str
            Absolute path to the image file.
        prompt : str
            Context-aware instruction for the vision model.

        Returns
        -------
        str
            The provider's textual description of the image, or an empty
            string if both calling conventions fail.
        """
        if hasattr(self._llm, "describe_image") and callable(
            getattr(self._llm, "describe_image")
        ):
            return self._llm.describe_image(image_path, prompt) or ""

        # Fallback: encode image as Base64 and pass via generate_text
        image_data = Path(image_path).read_bytes()
        b64 = base64.b64encode(image_data).decode("utf-8")
        return (
            self._llm.generate_text(prompt=prompt, images=[b64]) or ""
        )
