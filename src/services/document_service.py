from dataclasses import dataclass
import os
import re
import base64
import mimetypes
import statistics
from typing import Dict, List, Tuple

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import TextLoader

from models import ProcessingEnum
from services.project_service import ProjectService
from utils.tokenizer import count_tokens
from utils.tokenizer import count_tokens

import pytesseract
from pdf2image import convert_from_path
from PIL import Image


@dataclass
class Document:
    page_content: str
    metadata: dict


class DocumentService:

    # ── file-type registries ──────────────────────────────────
    IMAGE_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"}
    TEXT_EXTENSIONS   = {".txt", ".md", ".csv", ".json", ".html", ".xml", ".yaml", ".yml"}
    PDF_EXTENSION     = ".pdf"

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.project_path = ProjectService().get_project_path(project_id=project_id)

    # ─────────────────────────────────────────────────────────
    # existing helpers (unchanged)
    # ─────────────────────────────────────────────────────────

    def get_file_extension(self, file_id: str):
        return os.path.splitext(file_id)[-1].lower()

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
        file_path = os.path.join(self.project_path, file_id)

        if not os.path.exists(file_path):
            return None

        if file_ext == ProcessingEnum.PDF.value:
            return self.extract_pdf_as_markdown(file_path=file_path)

        if file_ext in self.TEXT_EXTENSIONS:
            text = self.extract_text_file(file_path=file_path)
            if text.strip():
                return [
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_id,
                            "file_type": "text",
                        },
                    )
                ]
            return None

        if file_ext in self.IMAGE_EXTENSIONS:
            text = self.extract_image_text(file_path=file_path)
            if text.strip():
                return [
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_id,
                            "file_type": "image",
                            "ocr_used": True,
                        },
                    )
                ]
            return None

        loader = self.get_file_loader(file_id=file_id)
        if loader:
            return loader.load()

        return None

    def process_file_content(self, file_content: list, file_id: str,
                            chunk_size: int = 800, overlap_size: int = 125):
        """
        Split a file's extracted pages/records into chunks.

        ``chunk_size``/``overlap_size`` are expressed in **tokens** (not
        characters) — see ``utils.tokenizer.count_tokens``. The
        recommended ranges are 700-1000 tokens per chunk with 100-150
        tokens of overlap, which keeps enough context for retrieval while
        still allowing accurate, focused answers.

        Every produced chunk is stamped with:
        - ``source``      → ``file_id`` (the lecture/file name)
        - ``chunk_index`` → 1-based position of the chunk within this file
        - ``section``     → nearest preceding markdown heading, if any
        - ``page``        → preserved from the original extraction metadata
                            (PDF pages already set this)
        """

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

        # Normalise/augment metadata for every chunk so downstream features
        # (citations, ordered full-lecture retrieval, dynamic filtering by
        # lecture/section) have consistent fields regardless of file type.
        for idx, chunk in enumerate(chunks):
            chunk.metadata = dict(chunk.metadata or {})
            chunk.metadata["source"] = file_id
            chunk.metadata["chunk_index"] = idx + 1

        return chunks

    def process_markdown_splitter(self, texts: List[str], metadatas: List[dict],
                                  chunk_size: int, overlap_size: int):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            length_function=count_tokens,
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

            if count_tokens(normalized_text) <= chunk_size:
                chunk_metadata = dict(metadata)
                section = self._first_heading(normalized_text)
                if section:
                    chunk_metadata["section"] = section

                chunks.append(Document(
                    page_content=normalized_text,
                    metadata=chunk_metadata
                ))
                continue

            blocks = self.extract_markdown_blocks(normalized_text)
            page_chunks = self.build_chunks_from_blocks(
                blocks=blocks,
                splitter=splitter,
                chunk_size=chunk_size,
                overlap_size=overlap_size,
            )

            for chunk_text, section in page_chunks:
                if chunk_text.strip():
                    chunk_metadata = dict(metadata)
                    if section:
                        chunk_metadata["section"] = section

                    chunks.append(Document(
                        page_content=chunk_text.strip(),
                        metadata=chunk_metadata
                    ))

        return chunks

    def _first_heading(self, text: str) -> str:
        """Return the first markdown heading found in *text*, if any."""
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
        return ""

    def extract_pdf_as_markdown(self, file_path: str) -> List[Document]:
        doc = fitz.open(file_path)
        pages = []

        try:
            for page_index, page in enumerate(doc):
                page_markdown = self.extract_page_markdown(page=page)
                cleaned_markdown = self.clean_markdown_text(page_markdown)

                if len(cleaned_markdown.strip()) < 50:
                    print(
                        f"[ProcessController] Page {page_index + 1} has < 50 chars "
                        f"after fitz extraction — running OCR fallback..."
                    )
                    cleaned_markdown = self._extract_page_via_ocr(
                        file_path=file_path,
                        page_number=page_index + 1,
                    )

                if not cleaned_markdown.strip():
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

    def _extract_page_via_ocr(self, file_path: str, page_number: int) -> str:
        try:
            images = convert_from_path(
                file_path,
                first_page=page_number,
                last_page=page_number,
            )

            ocr_parts = []
            for img in images:
                text = pytesseract.image_to_string(img, lang="eng+ara")
                if text.strip():
                    ocr_parts.append(text)

            return "\n\n".join(ocr_parts)

        except Exception as e:
            print(f"[ProcessController] OCR failed for page {page_number}: {e}")
            return ""

    # ─────────────────────────────────────────────────────────
    # NEW: multimodal file extraction methods
    # All methods below work on an absolute file_path directly.
    # They do NOT depend on self.project_path.
    # ─────────────────────────────────────────────────────────

    def classify_file_type(self, file_path: str) -> str:
        """
        Classify a file by extension.
        Returns: "pdf" | "image" | "text" | "unsupported"
        Never reads the file — extension only.
        """
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == self.PDF_EXTENSION:
            return "pdf"
        if ext in self.IMAGE_EXTENSIONS:
            return "image"
        if ext in self.TEXT_EXTENSIONS:
            return "text"
        return "unsupported"

    def extract_image_text(self, file_path: str) -> str:
        """
        Run pytesseract OCR on an image file (jpg, png, webp, etc.).
        Returns extracted text string. Returns "" on failure — never raises.
        """
        try:
            img = Image.open(file_path)
            # convert to RGB so tesseract handles all modes (RGBA, P, L, etc.)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            text = pytesseract.image_to_string(img, lang="eng+ara")
            return text.strip()
        except Exception as e:
            print(f"[ProcessController] image OCR failed for {file_path}: {e}")
            return ""

    def extract_image_base64(self, file_path: str) -> dict:
        """
        Encode an image file to base64 for vision-model consumption.
        Returns {"file_name": str, "b64": str, "mime": str}
        mime is guessed from extension; defaults to "image/jpeg".
        Never raises — returns empty dict on failure.
        """
        try:
            mime, _ = mimetypes.guess_type(file_path)
            if not mime or not mime.startswith("image/"):
                mime = "image/jpeg"
            with open(file_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return {
                "file_name": os.path.basename(file_path),
                "b64": b64,
                "mime": mime,
            }
        except Exception as e:
            print(f"[ProcessController] base64 encoding failed for {file_path}: {e}")
            return {}

    def extract_text_file(self, file_path: str) -> str:
        """
        Read a plain text file (.txt, .md, .csv, .json, .html, etc.).
        Tries UTF-8 first, falls back to latin-1.
        Returns text string. Returns "" on failure — never raises.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    return f.read()
            except Exception as e:
                print(f"[ProcessController] text read failed for {file_path}: {e}")
                return ""
        except Exception as e:
            print(f"[ProcessController] text read failed for {file_path}: {e}")
            return ""

    def extract_pdf_text_and_images(self, file_path: str) -> tuple:
        """
        Extract text AND embedded images from a PDF.

        Text extraction:
        - Uses existing extract_pdf_as_markdown() which already has
          per-page OCR fallback when fitz returns < 50 chars.

        Image extraction:
        - Iterates pages with fitz, extracts embedded raster images.
        - Each image encoded to base64 via extract_image_base64().

        Returns:
            (
                full_text: str,                      # all pages joined
                images: list[dict]                   # [{"file_name", "b64", "mime"}]
            )
        Never raises — returns ("", []) on total failure.
        """
        full_text = ""
        images = []

        try:
            # --- text ---
            pages = self.extract_pdf_as_markdown(file_path)
            full_text = "\n\n".join(p.page_content for p in pages if p.page_content.strip())

            # --- embedded images ---
            doc = fitz.open(file_path)
            try:
                for page_index, page in enumerate(doc):
                    for img_index, img_info in enumerate(page.get_images(full=True)):
                        xref = img_info[0]
                        try:
                            base_image = doc.extract_image(xref)
                            img_bytes  = base_image["image"]
                            img_ext    = base_image.get("ext", "png")
                            mime       = f"image/{img_ext}" if img_ext != "jpg" else "image/jpeg"
                            b64        = base64.b64encode(img_bytes).decode("utf-8")
                            images.append({
                                "file_name": f"{os.path.basename(file_path)}_p{page_index+1}_img{img_index+1}.{img_ext}",
                                "b64": b64,
                                "mime": mime,
                            })
                        except Exception as img_err:
                            print(f"[ProcessController] image extraction failed xref={xref}: {img_err}")
            finally:
                doc.close()

        except Exception as e:
            print(f"[ProcessController] PDF extraction failed for {file_path}: {e}")

        return full_text, images

    def extract_any_file(self, file_path: str) -> dict:
        """
        Universal dispatcher. Classifies the file and returns a unified result dict:
        {
            "file_name": str,
            "file_type": "pdf" | "image" | "text" | "unsupported",
            "text": str,          # extracted text (empty string if none)
            "images": list[dict], # [{"file_name", "b64", "mime"}] — populated for images and PDFs with embedded images
            "ocr_used": bool,     # True when pytesseract was the source of text
        }
        Never raises — returns a result with empty fields on failure.
        """
        file_name = os.path.basename(file_path)
        file_type = self.classify_file_type(file_path)

        result = {
            "file_name": file_name,
            "file_type": file_type,
            "text": "",
            "images": [],
            "ocr_used": False,
        }

        if file_type == "pdf":
            text, images = self.extract_pdf_text_and_images(file_path)
            result["text"] = text
            result["images"] = images
            # OCR was used if any page had < 50 chars from fitz (handled internally)
            # We flag it when text came back non-empty but fitz alone wouldn't have done it.
            # Simple heuristic: if images exist in a scanned PDF, ocr_used is likely True.
            result["ocr_used"] = True  # conservative — always flag PDF as potentially OCR'd

        elif file_type == "image":
            text = self.extract_image_text(file_path)
            b64_data = self.extract_image_base64(file_path)
            result["text"] = text
            result["ocr_used"] = True
            if b64_data:
                result["images"] = [b64_data]

        elif file_type == "text":
            result["text"] = self.extract_text_file(file_path)
            result["ocr_used"] = False

        else:
            print(f"[ProcessController] unsupported file type for: {file_path}")

        return result

    # ─────────────────────────────────────────────────────────
    # existing page-level parsing (unchanged below this line)
    # ─────────────────────────────────────────────────────────

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
                                 chunk_size: int, overlap_size: int) -> List[Tuple[str, str]]:
        """
        Build chunks from markdown blocks.

        Returns a list of ``(chunk_text, section)`` tuples where
        ``section`` is the nearest preceding heading text (or ``""`` if
        none was seen yet). Sizing decisions use ``count_tokens`` so
        ``chunk_size``/``overlap_size`` are interpreted as token counts.
        """
        chunks: List[Tuple[str, str]] = []
        current_blocks = []
        current_section = ""
        chunk_section = ""

        def build_text(items: List[str]) -> str:
            return "\n\n".join(items).strip()

        def append_current_chunk():
            nonlocal current_blocks, chunk_section
            if not current_blocks:
                return

            chunk_text = build_text(current_blocks)
            if chunk_text:
                chunks.append((chunk_text, chunk_section))

            if overlap_size <= 0:
                current_blocks = []
                return

            overlap_blocks = []
            overlap_tokens = 0
            for previous_block in reversed(current_blocks):
                overlap_blocks.insert(0, previous_block)
                overlap_tokens += count_tokens(previous_block)
                if overlap_tokens >= overlap_size:
                    break

            current_blocks = overlap_blocks

        for block in blocks:
            block_text = block.get("text", "").strip()
            if not block_text:
                continue

            block_type = block.get("type", "paragraph")
            protected_block = block_type in {"code", "table"}

            if block_type == "heading":
                current_section = block_text.lstrip("#").strip()

            # The section a chunk is tagged with is whatever section was
            # active when its first block was added.
            if not current_blocks:
                chunk_section = current_section

            if count_tokens(block_text) > chunk_size and not protected_block:
                append_current_chunk()

                split_texts = splitter.split_text(block_text)
                for segment in split_texts:
                    stripped_segment = segment.strip()
                    if stripped_segment:
                        chunks.append((stripped_segment, current_section))
                continue

            candidate_blocks = current_blocks + [block_text]
            candidate_text = build_text(candidate_blocks)
            if current_blocks and count_tokens(candidate_text) > chunk_size:
                append_current_chunk()
                current_blocks.append(block_text)
                if len(current_blocks) == 1:
                    chunk_section = current_section
            else:
                current_blocks = candidate_blocks

        append_current_chunk()

        return chunks

    def process_and_chunk_text(self, raw_text: str, chunk_size: int, overlap_size: int):
        clean_text = raw_text.replace('\x00', '')
        clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            length_function=len,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        )

        chunks = text_splitter.split_text(clean_text)

        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                "chunk_text": chunk.strip(),
                "chunk_order": i + 1,
                "chunk_metadata": {}
            })

        return processed_chunks
