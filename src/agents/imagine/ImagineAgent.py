"""
ImagineAgent — generates a NotebookLM-style infographic poster from lecture chunks.

Flow
----
1. Receive ordered chunks for the project.
2. Run a concept-extraction pass over batches (map step) to extract JSON.
3. Send the extracted concepts to the LLM to generate HTML/CSS (reduce step).
4. Return a dict: {title, diagram_type, content (HTML code)}.

The agent is stateless and synchronous.
"""

from __future__ import annotations

import logging
import json
import re
from typing import Optional

logger = logging.getLogger(__name__)

_CONCEPT_BATCH_CHARS = 10_000
_CONCEPT_MAX_TOKENS  = 1500
_HTML_MAX_TOKENS     = 4000


class ImagineAgent:
    def __init__(self, generation_client, template_parser, language: str = "en") -> None:
        self._llm             = generation_client
        self._template_parser = template_parser
        self._language        = language if language in ("ar", "en") else "en"

    def generate(self, chunks: list) -> dict:
        if not chunks:
            return self._error_response("No lecture content provided.")

        # Step 1 — concept extraction
        content_blocks = self._build_content_blocks(chunks)
        json_data_list = self._extract_concepts(content_blocks)

        if not json_data_list:
            return self._error_response("Could not extract concepts from the lecture.")

        # Combine JSON lists into a unified string for step 2
        combined_json_str = self._combine_json_concepts(json_data_list)

        # Step 2 — HTML generation
        html_code = self._generate_html(combined_json_str)
        if not html_code:
            return self._error_response("Could not generate infographic HTML.")

        html_code = self._clean_html(html_code)
        
        # Add a download button outside the canvas container
        html_code = self._inject_download_button(html_code)
        
        title = "Lecture Infographic"
        try:
            parsed = json.loads(json_data_list[0])
            title = parsed.get("poster_title", title)
        except:
            pass

        return {
            "title":        title,
            "diagram_type": "imagine",
            "content":      html_code,
        }

    def _build_content_blocks(self, chunks: list) -> list[str]:
        batches: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for chunk in chunks:
            if hasattr(chunk, "chunk_text"):
                text = (chunk.chunk_text or "").strip()
                meta = chunk.chunk_metadata or {}
            else:
                text = (chunk.get("text", "")).strip()
                meta = chunk.get("metadata", {})

            if not text:
                continue

            section = meta.get("section", "")
            block = f"[{section}]\n{text}" if section else text
            block_len = len(block)

            if current_parts and current_len + block_len > _CONCEPT_BATCH_CHARS:
                batches.append("\n\n".join(current_parts))
                current_parts = [block]
                current_len = block_len
            else:
                current_parts.append(block)
                current_len += block_len

        if current_parts:
            batches.append("\n\n".join(current_parts))

        return batches

    def _extract_concepts(self, batches: list[str]) -> list[str]:
        system = self._template_parser.get("rag", "imagine_concept_system")
        chat_history = [self._llm.construct_prompt(prompt=system, role=self._llm.enums.SYSTEM.value)]

        results = []
        for i, batch in enumerate(batches, 1):
            try:
                user_prompt = self._template_parser.get("rag", "imagine_concept_user", {"content": batch})
                result = self._llm.generate_text(
                    prompt=user_prompt, 
                    chat_history=chat_history,
                    max_output_tokens=_CONCEPT_MAX_TOKENS
                )
                if result:
                    # Clean up markdown JSON fences
                    cleaned = re.sub(r"```(?:json)?", "", result).strip()
                    results.append(cleaned)
            except Exception as exc:
                logger.warning("ImagineAgent: concept extraction batch %d failed: %s", i, exc)

        return results

    def _combine_json_concepts(self, json_list: list[str]) -> str:
        # For simplicity, we stringify the array of JSON strings or objects
        combined = []
        for j in json_list:
            try:
                combined.append(json.loads(j))
            except:
                combined.append(j)
        return json.dumps(combined, indent=2)

    def _generate_html(self, json_data: str) -> Optional[str]:
        system = self._template_parser.get("rag", "imagine_html_system")
        user_prompt = self._template_parser.get("rag", "imagine_html_user", {"json_data": json_data})

        chat_history = [
            self._llm.construct_prompt(
                prompt=system, role=self._llm.enums.SYSTEM.value
            )
        ]

        try:
            return self._llm.generate_text(
                prompt=user_prompt,
                chat_history=chat_history,
                max_output_tokens=_HTML_MAX_TOKENS,
            )
        except Exception as exc:
            logger.error("ImagineAgent: HTML generation failed: %s", exc)
            return None

    def _clean_html(self, raw: str) -> str:
        cleaned = re.sub(r"```(?:html)?", "", raw)
        return cleaned.strip()

    def _inject_download_button(self, html_code: str) -> str:
        wrapper = f"""
        <div>
            <div style="text-align: right; padding: 10px; background: #f8f9fa; border-bottom: 1px solid #ddd; position: sticky; top: 0; z-index: 9999;">
                <button id="download-imagine-btn" style="padding: 8px 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; font-family: sans-serif;">
                    ⬇️ Download PNG
                </button>
            </div>
            {html_code}
            
            <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
            <script>
                document.getElementById('download-imagine-btn').addEventListener('click', function() {{
                    const canvasContainer = document.getElementById('imagine-canvas');
                    if (canvasContainer) {{
                        html2canvas(canvasContainer, {{ scale: 2 }}).then(canvas => {{
                            const link = document.createElement('a');
                            link.download = 'infographic.png';
                            link.href = canvas.toDataURL('image/png');
                            link.click();
                        }});
                    }} else {{
                        alert("Could not find the infographic canvas (<div id='imagine-canvas'>) to download.");
                    }}
                }});
            </script>
        </div>
        """
        return wrapper

    def _error_response(self, message: str) -> dict:
        return {
            "title":        "Error",
            "diagram_type": "imagine",
            "content":      f"<div id='imagine-canvas' style='padding: 20px; color: red;'><h3>Error Generating Infographic</h3><p>{message}</p></div>",
        }