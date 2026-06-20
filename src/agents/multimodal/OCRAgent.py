import pytesseract
from PIL import Image

from agents.base import BaseAgent, AgentState
from services.document_service import DocumentService


class OCRAgent(BaseAgent):
    """
    Agent that extracts plain text from image files using Tesseract OCR.

    The agent delegates to the same ``pytesseract`` integration that
    ``ProcessController._extract_page_via_ocr`` uses, keeping the OCR
    configuration (language packs, Tesseract binary path, etc.) consistent
    across the codebase.  The ``ProcessController`` instance is injected
    rather than constructed internally so the caller controls its
    ``project_id`` context.

    Parameters
    ----------
    process_controller : ProcessController
        Pre-configured controller instance.  Its ``pytesseract`` import is
        reused here; no second import of the library is needed.
    """

    def __init__(self, process_controller: DocumentService) -> None:
        self._process_controller = process_controller

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def agent_name(self) -> str:
        """Return the display name used in logs and agent trace entries."""
        return "OCRAgent"

    async def execute(self, state: AgentState) -> AgentState:
        """
        Run Tesseract OCR on every image in ``state["image_paths"]`` and
        concatenate the results into ``state["ocr_text"]``.

        Steps
        -----
        1. Validate that ``image_paths`` is present in *state*.
        2. For each path in ``state["image_paths"]``:
           a. Open the image with ``PIL.Image``.
           b. Call ``pytesseract.image_to_string`` with the ``eng+ara``
              language pack (matching ``ProcessController``'s OCR setup).
           c. Append non-empty text to the running buffer.
        3. Join all extracted segments with a double newline and write to
           ``state["ocr_text"]``.
        4. On any exception, log a warning and set ``state["ocr_text"] = ""``.
        5. Append a trace entry and return the updated state.

        Parameters
        ----------
        state : AgentState
            Current pipeline state.  Must contain ``image_paths``.

        Returns
        -------
        AgentState
            Updated state with ``ocr_text`` populated.
        """
        self.validate_state(state, ["image_paths"])

        image_paths = state["image_paths"]

        if not image_paths:
            state["ocr_text"] = ""
            state["agent_trace"].append(
                f"{self.agent_name}: no images to process"
            )
            self.log_step("no image paths provided — skipping OCR")
            return state

        self.log_step(f"running OCR on {len(image_paths)} image(s)")

        extracted_parts = []

        for image_path in image_paths:
            try:
                img = Image.open(image_path)
                # Use the same language pack as ProcessController._extract_page_via_ocr
                text = pytesseract.image_to_string(img, lang="eng+ara")
                if text.strip():
                    extracted_parts.append(text.strip())
            except Exception as exc:
                self.log_step(
                    f"WARNING: OCR failed for '{image_path}': {exc}"
                )

        state["ocr_text"] = "\n\n".join(extracted_parts)

        state["agent_trace"].append(
            f"{self.agent_name}: extracted text from {len(image_paths)} image(s)"
        )
        self.log_step(
            f"extracted {len(extracted_parts)} non-empty OCR result(s) "
            f"from {len(image_paths)} image(s)"
        )
        return state
