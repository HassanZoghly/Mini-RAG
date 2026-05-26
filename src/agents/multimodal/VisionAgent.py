import base64
from pathlib import Path
from agents.base import BaseAgent, AgentState

class VisionAgent(BaseAgent):
    """
    Advanced Vision Agent that acts as an Educational Image Analyzer.
    Processes both direct image uploads and images extracted from scanned PDFs.
    """

    def __init__(self, llm_provider) -> None:
        self._llm = llm_provider

    @property
    def agent_name(self) -> str:
        return "VisionAgent"

    async def execute(self, state: AgentState) -> AgentState:
        # 🔥 قراءة الصور سواء كانت مرفوعة مباشرة أو مستخرجة من PDF
        image_base64_list = state.get("image_base64", [])
        query: str = state.get("query", "")

        if not image_base64_list:
            state["vision_description"] = state.get("ocr_text", "")
            state["agent_trace"].append(f"{self.agent_name}: no images to describe")
            self.log_step("no images — skipping vision description")
            return state

        if self._supports_vision(self._llm):
            descriptions = []

            # 🔥 Prompt أكاديمي لاستخراج النصوص وفهم المخططات (بدلاً من مجرد الوصف)
            prompt = (
                "You are an expert academic AI assistant analyzing a lecture slide or document page.\n"
                "1. Extract ALL visible text with high accuracy.\n"
                "2. If there are diagrams, flowcharts, tables, or mathematical equations, explain them in detailed steps.\n"
                "3. Ensure the output is highly structured and useful for summarizing the lecture material.\n"
                f"User context/query: {query}"
            )

            for img_dict in image_base64_list:
                try:
                    b64 = img_dict["b64"]
                    # نرسل الصورة كـ Base64 مباشرة للموديل
                    description = self._llm.generate_text(prompt=prompt, images=[b64])
                    if description:
                        descriptions.append(f"### Content from: {img_dict['file_name']}\n{description.strip()}")
                except Exception as exc:
                    self.log_step(f"WARNING: vision description failed for '{img_dict['file_name']}': {exc}")

            state["vision_description"] = "\n\n".join(descriptions)
            self.log_step(f"generated vision description for {len(descriptions)} image(s)")
        else:
            state["vision_description"] = state.get("ocr_text", "")
            self.log_step("provider does not support vision — using ocr_text as fallback")

        state["agent_trace"].append(f"{self.agent_name}: generated visual description")
        return state

    def _supports_vision(self, provider) -> bool:
        if hasattr(provider, "supports_vision"):
            return bool(provider.supports_vision)
        if hasattr(provider, "describe_image") and callable(getattr(provider, "describe_image")):
            return True
        if hasattr(provider, "vision") and provider.vision:
            return True
        return False
