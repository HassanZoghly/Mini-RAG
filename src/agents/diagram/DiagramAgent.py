import json
import re
from typing import Any, Dict


class DiagramAgent:
    """Generate lecture-wide Mermaid diagrams from document context."""

    def __init__(self, llm_provider) -> None:
        self._llm = llm_provider

    def generate(self, context: str, language: str = "English", diagram_type: str = "flowchart") -> Dict[str, str]:
        context = self._prepare_context(context)
        if not context:
            raise ValueError("No document context was provided for diagram generation.")

        language_name = self._normalize_language(language)
        diagram_type = (diagram_type or "flowchart").strip().lower()

        system_prompt = self._build_system_prompt(language=language_name, diagram_type=diagram_type)
        user_prompt = self._build_user_prompt(context=context, language=language_name, diagram_type=diagram_type)

        chat_history = [
            self._llm.construct_prompt(
                prompt=system_prompt,
                role=self._llm.enums.SYSTEM.value,
            )
        ]

        raw_response = self._llm.generate_text(
            prompt=user_prompt,
            chat_history=chat_history,
            max_output_tokens=4096,
            temperature=0.15,
        )

        parsed = self._parse_json_object(raw_response or "")
        title = str(parsed.get("title", "Lecture Diagram")).strip() or "Lecture Diagram"
        content = self._clean_mermaid(str(parsed.get("content", "")).strip())
        returned_type = str(parsed.get("diagram_type", diagram_type or "flowchart")).strip() or "flowchart"

        if not content:
            raise ValueError("Diagram generation returned empty Mermaid content.")

        return {
            "title": title,
            "diagram_type": returned_type,
            "content": content,
        }

    def _normalize_language(self, language: str) -> str:
        language = (language or "English").strip()
        if language.lower() in {"arabic", "ar", "العربية", "عربي"}:
            return "Arabic"
        return "English"

    def _prepare_context(self, context: str) -> str:
        cleaned = (context or "").replace("\x00", "").strip()
        return cleaned[:60000]

    def _build_system_prompt(self, language: str, diagram_type: str) -> str:
        return "\n".join([
            "You are an expert lecture visualizer.",
            "Your task is to convert lecture/document context into a concise Mermaid diagram.",
            "Return ONLY valid JSON. Do not wrap the JSON or Mermaid in Markdown fences.",
            "Use only concepts supported by the provided context.",
            f"Diagram labels should be in {language}.",
            "",
            "Required JSON schema:",
            "{",
            '  "title": "string",',
            '  "diagram_type": "flowchart",',
            '  "content": "Mermaid code string"',
            "}",
            "",
            "Mermaid rules:",
            "- Prefer `graph TD` for lecture topic hierarchies and concept flows.",
            "- CRITICAL: Node IDs must be alphanumeric and contain NO spaces (e.g., node1, topicA).",
            "- CRITICAL: Node labels MUST be wrapped in double quotes. Example: A[\"Node Label\"] --> B[\"Another Label (Extra)\"]",
            "- CRITICAL: Do NOT use double quotes inside the label text itself.",
            "- CRITICAL: The diagram MUST be highly detailed and informative. Do not just list high-level terms.",
            "- Include specific definitions, conditions, formulas, or examples inside the node labels (e.g., instead of just 'Core Point', use 'Core Point: Has >= MinPts within Eps').",
            "- Use descriptive text on arrows to explain HOW or WHY nodes are connected (e.g., A -->|\"If condition met\"| B).",
            "- Do not include Markdown code fences.",
            "- Avoid unsupported Mermaid syntax.",
        ])

    def _build_user_prompt(self, context: str, language: str, diagram_type: str) -> str:
        return "\n\n".join([
            f"Generate a highly detailed and explanatory {diagram_type} Mermaid diagram in {language} for the whole lecture.",
            "Make sure the diagram acts as a comprehensive study guide. Include exact conditions, definitions, and step-by-step logic from the text, rather than just a table of contents.",
            "LECTURE CONTEXT:",
            context,
            "Return valid JSON only now.",
        ])

    def _parse_json_object(self, text: str) -> Dict[str, Any]:
        cleaned = (text or "").strip()
        cleaned_no_fences = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned_no_fences = re.sub(r"\s*```$", "", cleaned_no_fences)

        try:
            data = json.loads(cleaned_no_fences, strict=False)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(cleaned[start:end + 1], strict=False)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
                
        # Fallback 1: Extract mermaid block if JSON fails or isn't present
        mermaid_match = re.search(r'```(?:mermaid)?\n(.*?)\n```', cleaned, re.DOTALL | re.IGNORECASE)
        if mermaid_match:
            return {
                "title": "Lecture Diagram",
                "diagram_type": "flowchart",
                "content": mermaid_match.group(1).strip()
            }
            
        # Fallback 2: Maybe the whole text is just a mermaid graph?
        if "graph " in cleaned or "flowchart " in cleaned or "stateDiagram" in cleaned:
            return {
                "title": "Lecture Diagram",
                "diagram_type": "flowchart",
                "content": cleaned_no_fences
            }

        raise ValueError(f"LLM did not return a JSON object or Mermaid diagram. Snippet: {cleaned[:250]}")

    def _clean_mermaid(self, content: str) -> str:
        content = (content or "").strip()
        
        content = re.sub(r"```mermaid\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"```\s*", "", content)
        content = content.replace("\\n", "\n").strip()

        lines = content.split('\n')
        start_idx = -1
        for i, line in enumerate(lines):
            if re.match(r"^\s*(graph|flowchart|stateDiagram|mindmap|pie|sequenceDiagram|classDiagram|erDiagram|gantt|journey|gitGraph)", line, re.IGNORECASE):
                start_idx = i
                break
        
        if start_idx != -1:
            lines = lines[start_idx:]
        else:
            lines = ["graph TD"] + lines

        shapes = [
            (r'\[\(', r'\)\]'), (r'\(\[', r'\]\)'), (r'\[\[', r'\]\]'), (r'\(\(', r'\)\)'),
            (r'\{\{', r'\}\}'), (r'\[/', r'/\]'), (r'\[\\', r'\\\]'), (r'\[/', r'\\\]'), (r'\[\\', r'/\]'),
            (r'\[', r'\]'), (r'\(', r'\)'), (r'\{', r'\}'), (r'>', r'\]')
        ]

        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('%%') or re.match(r"^\s*(graph|flowchart|stateDiagram|mindmap|pie|sequenceDiagram|classDiagram|subgraph|end|style|classDef|class|click|linkStyle)", line, re.IGNORECASE):
                cleaned_lines.append(line)
                continue
                
            parts = re.split(r'(\s*(?:-[-\.]*->|-[-\.]*-|=[=]*=>|=[=]*=)\s*)', line)
            new_parts = []
            
            for i, p in enumerate(parts):
                if i % 2 == 1:
                    new_parts.append(p)
                    continue
                
                p = p.strip()
                arrow_label = ""
                if p.startswith('|'):
                    end_pipe = p.find('|', 1)
                    if end_pipe != -1:
                        arrow_label = p[:end_pipe+1]
                        p = p[end_pipe+1:].strip()
                
                if not p:
                    new_parts.append(arrow_label)
                    continue
                
                m = re.match(r'^([^\[\(\{\>]+)(.*)$', p)
                if m:
                    raw_id = m.group(1).strip()
                    rest = m.group(2).strip()
                    
                    safe_id = re.sub(r'[^a-zA-Z0-9_]', '_', raw_id)
                    if safe_id and safe_id[0].isdigit():
                        safe_id = "n_" + safe_id
                    if not safe_id:
                        safe_id = "node"
                        
                    if rest:
                        b_open, inner, b_close = None, None, None
                        for open_pat, close_pat in shapes:
                            m_shape = re.match(f'^({open_pat})(.*)({close_pat})$', rest)
                            if m_shape:
                                b_open = rest[:len(open_pat.replace('\\', ''))]
                                inner = m_shape.group(2).strip()
                                b_close = rest[-len(close_pat.replace('\\', '')):]
                                break
                                
                        if b_open:
                            if inner.startswith('"') and inner.endswith('"'):
                                inner_quoted = inner
                            else:
                                inner_clean = inner.replace('"', "'")
                                inner_quoted = f'"{inner_clean}"'
                            new_node = f'{safe_id}{b_open}{inner_quoted}{b_close}'
                        else:
                            new_node = f'{safe_id}{rest}'
                    else:
                        if raw_id == safe_id:
                            new_node = safe_id
                        else:
                            safe_label = raw_id.replace('"', "'")
                            new_node = f'{safe_id}["{safe_label}"]'
                            
                    new_parts.append(arrow_label + new_node)
                else:
                    new_parts.append(arrow_label + p)
                    
            cleaned_lines.append("".join(new_parts))
            
        return '\n'.join(cleaned_lines)
