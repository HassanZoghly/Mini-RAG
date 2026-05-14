from dataclasses import dataclass
import os
import re
import statistics
from typing import Dict, List, Tuple

import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import TextLoader

from models import ProcessingEnum

from .BaseController import BaseController
from .ProjectController import ProjectController


@dataclass
class Document:
    page_content: str
    metadata: dict


class ProcessController(BaseController):

    def __init__(self, project_id: str):
        super().__init__()

        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)

    def get_file_extension(self, file_id: str):
        return os.path.splitext(file_id)[-1]

    def get_file_loader(self, file_id: str):

        file_ext = self.get_file_extension(file_id=file_id)
        file_path = os.path.join(self.project_path, file_id)

        if not os.path.exists(file_path):
            return None

        if file_ext == ProcessingEnum.TXT.value:
            return TextLoader(file_path, encoding="utf-8")

        if file_ext == ProcessingEnum.PDF.value:
            return PyMuPDFLoader(file_path)

        return None

    def get_file_content(self, file_id: str):

        file_ext = self.get_file_extension(file_id=file_id)
        if file_ext == ProcessingEnum.PDF.value:
            file_path = os.path.join(self.project_path, file_id)
            return self.extract_pdf_as_markdown(file_path=file_path)

        loader = self.get_file_loader(file_id=file_id)
        if loader:
            return loader.load()

        return None

    def process_file_content(self, file_content: list, file_id: str,
                            chunk_size: int = 100, overlap_size: int = 20):

        file_content_texts = [
            rec.page_content
            for rec in file_content
        ]

        file_content_metadata = [
            rec.metadata
            for rec in file_content
        ]

        chunks = self.process_markdown_splitter(
            texts=file_content_texts,
            metadatas=file_content_metadata,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
        )

        return chunks

    def process_markdown_splitter(self, texts: List[str], metadatas: List[dict],
                                  chunk_size: int, overlap_size: int):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            separators=[
                "\n```",
                "\n\n|",
                "\n\n## ",
                "\n\n### ",
                "\n\n#### ",
                "\n\n",
                "\n",
                " ",
                "",
            ],
            keep_separator=True,
        )

        chunks = []
        for idx, text in enumerate(texts):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            normalized_text = self.clean_markdown_text(text)

            if not normalized_text.strip():
                continue

            if len(normalized_text) <= chunk_size:
                chunks.append(Document(
                    page_content=normalized_text,
                    metadata=metadata
                ))
                continue

            blocks = self.extract_markdown_blocks(normalized_text)
            page_chunks = self.build_chunks_from_blocks(
                blocks=blocks,
                splitter=splitter,
                chunk_size=chunk_size,
                overlap_size=overlap_size,
            )

            for chunk_text in page_chunks:
                if chunk_text.strip():
                    chunks.append(Document(
                        page_content=chunk_text.strip(),
                        metadata=metadata
                    ))

        return chunks

    def extract_pdf_as_markdown(self, file_path: str) -> List[Document]:
        doc = fitz.open(file_path)
        pages = []

        try:
            for page_index, page in enumerate(doc):
                page_markdown = self.extract_page_markdown(page=page)
                cleaned_markdown = self.clean_markdown_text(page_markdown)
                if not cleaned_markdown:
                    continue

                pages.append(Document(
                    page_content=cleaned_markdown,
                    metadata={
                        "page": page_index + 1,
                        "source": os.path.basename(file_path),
                    }
                ))
        finally:
            doc.close()

        return pages

    def extract_page_markdown(self, page) -> str:
        page_dict = page.get_text("dict", sort=True)
        blocks = page_dict.get("blocks", [])
        body_font_size = self.estimate_body_font_size(blocks)

        table_blocks = self.extract_tables_from_page(page)

        markdown_lines = []
        emitted_table_ids = set()

        for block in blocks:
            block_bbox = tuple(block.get("bbox", (0, 0, 0, 0)))

            table_for_block = self.get_table_for_bbox(block_bbox, table_blocks)
            if table_for_block:
                table_id, table_text = table_for_block
                if table_id not in emitted_table_ids:
                    markdown_lines.append("")
                    markdown_lines.append(table_text.strip())
                    markdown_lines.append("")
                    emitted_table_ids.add(table_id)
                continue

            line_texts = []
            for line in block.get("lines", []):
                text, max_font_size, font_names = self.extract_line_properties(line)
                if not text:
                    continue

                normalized = self.normalize_line(
                    line_text=text,
                    max_font_size=max_font_size,
                    font_names=font_names,
                    body_font_size=body_font_size,
                )
                line_texts.append(normalized)

            if line_texts:
                markdown_lines.append("\n".join(line_texts))

        for table_id, table_text in table_blocks:
            if table_id not in emitted_table_ids:
                markdown_lines.append("")
                markdown_lines.append(table_text.strip())
                markdown_lines.append("")

        return "\n".join(markdown_lines)

    def extract_tables_from_page(self, page) -> List[Tuple[Tuple[float, float, float, float], str]]:
        tables = []

        try:
            table_finder = page.find_tables()
            raw_tables = getattr(table_finder, "tables", table_finder)

            for table in raw_tables:
                markdown = table.to_markdown().strip()
                if not markdown:
                    continue
                tables.append((tuple(table.bbox), markdown))
        except Exception:
            return []

        return tables

    def get_table_for_bbox(self, bbox: Tuple[float, float, float, float],
                           tables: List[Tuple[Tuple[float, float, float, float], str]]):
        for table_bbox, table_text in tables:
            if self.boxes_overlap(bbox, table_bbox):
                return table_bbox, table_text
        return None

    def boxes_overlap(self, first_box: Tuple[float, float, float, float],
                      second_box: Tuple[float, float, float, float]) -> bool:
        x0, y0, x1, y1 = first_box
        a0, b0, a1, b1 = second_box

        overlap_x = max(0, min(x1, a1) - max(x0, a0))
        overlap_y = max(0, min(y1, b1) - max(y0, b0))

        return overlap_x > 0 and overlap_y > 0

    def estimate_body_font_size(self, blocks: List[Dict]) -> float:
        font_sizes = []
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        font_sizes.append(float(span.get("size", 0)))

        if not font_sizes:
            return 11.0

        return statistics.median(font_sizes)

    def extract_line_properties(self, line: Dict) -> Tuple[str, float, List[str]]:
        spans = line.get("spans", [])
        text_parts = []
        max_size = 0.0
        font_names = []

        for span in spans:
            span_text = span.get("text", "")
            if not span_text:
                continue
            text_parts.append(span_text)
            max_size = max(max_size, float(span.get("size", 0)))
            font_names.append((span.get("font", "") or "").lower())

        text = re.sub(r"\s+", " ", "".join(text_parts)).strip()
        return text, max_size, font_names

    def normalize_line(self, line_text: str, max_font_size: float,
                       font_names: List[str], body_font_size: float) -> str:
        text = line_text.strip()
        if not text:
            return ""

        if self.is_list_item(text):
            return self.normalize_list_item(text)

        if self.is_heading(text, max_font_size, body_font_size):
            clean_text = text.lstrip("#").strip()
            return f"## {clean_text}"

        if self.is_code_line(text, font_names):
            return f"`{text}`"

        return text

    def is_heading(self, text: str, max_font_size: float, body_font_size: float) -> bool:
        if text.startswith("#"):
            return True

        if len(text) > 120:
            return False

        looks_like_title_case = bool(re.match(r"^([A-Z][^\n.]{2,})$", text))
        looks_numbered_header = bool(re.match(r"^\d+(\.\d+)*\s+[A-Za-z].*", text))

        return (
            max_font_size >= body_font_size + 1.25
            or looks_like_title_case
            or looks_numbered_header
        )

    def is_list_item(self, text: str) -> bool:
        pattern = r"^([-*+\u2022\u25E6\u25AA\u2023]|\d+[.)])\s+"
        return bool(re.match(pattern, text))

    def normalize_list_item(self, text: str) -> str:
        pattern = r"^([-*+\u2022\u25E6\u25AA\u2023]|\d+[.)])\s+"
        return re.sub(pattern, "- ", text)

    def is_code_line(self, text: str, font_names: List[str]) -> bool:
        if not font_names:
            return False

        is_monospace = any("courier" in name or "mono" in name for name in font_names)
        if not is_monospace:
            return False

        return bool(re.search(r"[{}();=<>]|^\s{2,}", text))

    def clean_markdown_text(self, text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")

        code_block_pattern = re.compile(r"```[\s\S]*?```")
        code_blocks = []

        def protect_code(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

        protected = code_block_pattern.sub(protect_code, normalized)

        protected = re.sub(r"[ \t]+", " ", protected)
        protected = re.sub(r"\n{3,}", "\n\n", protected)

        cleaned_lines = [line.rstrip() for line in protected.split("\n")]
        cleaned = "\n".join(cleaned_lines).strip()

        for idx, block in enumerate(code_blocks):
            cleaned = cleaned.replace(f"__CODE_BLOCK_{idx}__", block)

        return cleaned

    def extract_markdown_blocks(self, text: str) -> List[Dict[str, str]]:
        lines = text.split("\n")
        blocks = []
        current_lines = []
        in_code_block = False

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("```"):
                if current_lines:
                    blocks.append(self.make_block(current_lines))
                    current_lines = []

                current_lines.append(line)
                in_code_block = not in_code_block
                if not in_code_block:
                    blocks.append({"type": "code", "text": "\n".join(current_lines).strip()})
                    current_lines = []
                continue

            if in_code_block:
                current_lines.append(line)
                continue

            if stripped == "":
                if current_lines:
                    blocks.append(self.make_block(current_lines))
                    current_lines = []
                continue

            if self.is_markdown_table_line(stripped):
                if current_lines and not self.is_markdown_table_line(current_lines[-1].strip()):
                    blocks.append(self.make_block(current_lines))
                    current_lines = []
                current_lines.append(line)
                continue

            if current_lines and self.is_markdown_table_line(current_lines[-1].strip()):
                blocks.append({"type": "table", "text": "\n".join(current_lines).strip()})
                current_lines = []

            current_lines.append(line)

        if current_lines:
            if self.is_markdown_table_line(current_lines[0].strip()):
                blocks.append({"type": "table", "text": "\n".join(current_lines).strip()})
            else:
                blocks.append(self.make_block(current_lines))

        return blocks

    def is_markdown_table_line(self, line: str) -> bool:
        if "|" not in line:
            return False

        separator_pattern = r"^\|\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?$"
        if re.match(separator_pattern, line):
            return True

        return line.count("|") >= 2

    def make_block(self, lines: List[str]) -> Dict[str, str]:
        text = "\n".join(lines).strip()
        if not text:
            return {"type": "paragraph", "text": ""}

        first_line = lines[0].strip()
        if first_line.startswith("#"):
            return {"type": "heading", "text": text}

        if self.is_list_item(first_line):
            return {"type": "list", "text": text}

        return {"type": "paragraph", "text": text}

    def build_chunks_from_blocks(self, blocks: List[Dict[str, str]], splitter: RecursiveCharacterTextSplitter,
                                 chunk_size: int, overlap_size: int) -> List[str]:
        chunks = []
        current_blocks = []

        def build_text(items: List[str]) -> str:
            return "\n\n".join(items).strip()

        def append_current_chunk():
            nonlocal current_blocks
            if not current_blocks:
                return

            chunk_text = build_text(current_blocks)
            if chunk_text:
                chunks.append(chunk_text)

            if overlap_size <= 0:
                current_blocks = []
                return

            overlap_blocks = []
            overlap_length = 0
            for previous_block in reversed(current_blocks):
                overlap_blocks.insert(0, previous_block)
                overlap_length += len(previous_block) + 2
                if overlap_length >= overlap_size:
                    break

            current_blocks = overlap_blocks

        for block in blocks:
            block_text = block.get("text", "").strip()
            if not block_text:
                continue

            block_type = block.get("type", "paragraph")
            protected_block = block_type in {"code", "table"}

            if len(block_text) > chunk_size and not protected_block:
                append_current_chunk()

                split_texts = splitter.split_text(block_text)
                for segment in split_texts:
                    stripped_segment = segment.strip()
                    if stripped_segment:
                        chunks.append(stripped_segment)
                continue

            candidate_blocks = current_blocks + [block_text]
            candidate_text = build_text(candidate_blocks)
            if current_blocks and len(candidate_text) > chunk_size:
                append_current_chunk()
                current_blocks.append(block_text)
            else:
                current_blocks = candidate_blocks

        append_current_chunk()

        return chunks

    def process_and_chunk_text(self, raw_text: str, chunk_size: int, overlap_size: int):
        # 1. تنظيف مبدئي للنص من المسافات الغريبة والرموز الميتة
        clean_text = raw_text.replace('\x00', '')
        clean_text = re.sub(r'\n{3,}', '\n\n', clean_text) # منع المسافات الفاضية الطويلة جداً

        # 2. التقطيع الذكي (Smart Chunking)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            length_function=len,
            # الفواصل دي بتجبره يقطع عند نهاية البراجراف، لو مقدرش يقطع عند نهاية الجملة، وهكذا
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        )

        chunks = text_splitter.split_text(clean_text)

        # 3. تجهيز الداتا عشان تتخزن في الداتا بيز
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                "chunk_text": chunk.strip(),
                "chunk_order": i + 1,
                "chunk_metadata": {} # شلنا الميتاداتا زي ما اتفقنا
            })

        return processed_chunks
