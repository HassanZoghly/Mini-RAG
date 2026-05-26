import os
import base64
import mimetypes
import logging
import asyncio
from pathlib import Path

# Try importing fitz and pdf2image/pytesseract for PDF parsing
try:
    import fitz
except ImportError:
    fitz = None

try:
    from pdf2image import convert_from_path
    import pytesseract
except ImportError:
    convert_from_path = None
    pytesseract = None


logger = logging.getLogger(__name__)

class MultiFileProcessor:
    """
    Pre-processes a list of uploaded files before the agent graph runs.
    Classifies each file, extracts text, encodes images.
    One bad file never blocks the others — errors are caught per file.
    """

    SUPPORTED_TEXT_EXT  = {".txt", ".md", ".csv", ".json", ".html"}
    SUPPORTED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    SUPPORTED_PDF_EXT   = {".pdf"}

    def __init__(self):
        pass

    def classify_file(self, file_path: str) -> str:
        """Returns: "pdf" | "image" | "text" | "unsupported" — based on extension only."""
        ext = Path(file_path).suffix.lower()
        if ext in self.SUPPORTED_PDF_EXT:   return "pdf"
        if ext in self.SUPPORTED_IMAGE_EXT: return "image"
        if ext in self.SUPPORTED_TEXT_EXT:  return "text"
        return "unsupported"

    def _read_text_file(self, file_path: str) -> str:
        """UTF-8 read with latin-1 fallback."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()

    def _image_to_base64(self, file_path: str) -> dict:
        """
        Returns {"file_name": str, "b64": str, "mime": str}.
        Use mimetypes.guess_type() for mime. Default mime: "image/jpeg".
        """
        file_name = Path(file_path).name
        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = mime_type or "image/jpeg"
        with open(file_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        return {"file_name": file_name, "b64": b64_data, "mime": mime_type}

    def _pdf_to_text_and_images(self, file_path: str) -> tuple[str, list[dict]]:
        """
        1. Try PyMuPDF (fitz) for text extraction.
        2. Also extract embedded images from each page using fitz.
        3. If extracted text is empty or < 50 chars → OCR fallback:
           use pdf2image to convert pages to PIL images, then pytesseract.
        Returns: (full_text: str, images: list[dict])
        """
        full_text = ""
        images = []
        file_name = Path(file_path).name

        if fitz:
            try:
                doc = fitz.open(file_path)
                text_parts = []
                for page_idx in range(len(doc)):
                    page = doc[page_idx]
                    text_parts.append(page.get_text())

                    # Extract embedded images
                    image_list = page.get_images(full=True)
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        b64_data = base64.b64encode(image_bytes).decode("utf-8")
                        mime_type = f"image/{image_ext}"
                        images.append({
                            "file_name": f"{file_name}_p{page_idx}_img{img_index}.{image_ext}",
                            "b64": b64_data,
                            "mime": mime_type
                        })
                full_text = "\n".join(text_parts).strip()
            except Exception as e:
                logger.warning(f"PyMuPDF failed on {file_path}: {e}")

        # OCR fallback if text is too short or fitz failed
        if len(full_text) < 50 and convert_from_path and pytesseract:
            logger.info(f"Using OCR fallback and converting pages to images for {file_path}")
            try:
                pages = convert_from_path(file_path)
                ocr_parts = []
                import io # مكتبة ضرورية للتعامل مع الصور في الذاكرة

                for idx, page in enumerate(pages):
                    # 1. استخراج نص مبدئي (اختياري)
                    text = pytesseract.image_to_string(page, lang="eng+ara")
                    ocr_parts.append(text)

                    # 2. 🔥 التعديل الأهم: حفظ الصفحة كصورة وتمريرها للـ Vision Agent
                    img_byte_arr = io.BytesIO()
                    page.save(img_byte_arr, format='PNG')
                    b64_data = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                    images.append({
                        "file_name": f"{file_name}_scanned_page_{idx+1}.png",
                        "b64": b64_data,
                        "mime": "image/png"
                    })

                full_text = "\n".join(ocr_parts).strip()
            except Exception as e:
                logger.warning(f"OCR fallback failed on {file_path}: {e}")

        return full_text, images

    async def process_files(self, file_paths: list[str]) -> dict:
        """
        Processes all files. Returns a dict to merge into AgentState.
        """
        state_update = {
            "uploaded_files": [],
            "file_texts": [],
            "image_base64": [],
            "image_paths": [],
            "fusion_strategy": "text_only",
            "ocr_text": "",
        }

        loop = asyncio.get_event_loop()
        ocr_texts = []

        for path in file_paths:
            try:
                file_type = self.classify_file(path)
                file_name = Path(path).name
                state_update["uploaded_files"].append({
                    "path": path,
                    "type": file_type,
                    "name": file_name
                })

                if file_type == "text":
                    text = await loop.run_in_executor(None, self._read_text_file, path)
                    if text.strip():
                        state_update["file_texts"].append({
                            "file_name": file_name,
                            "text": text.strip(),
                            "source": "direct"
                        })
                elif file_type == "image":
                    state_update["image_paths"].append(path)
                    b_dict = await loop.run_in_executor(None, self._image_to_base64, path)
                    state_update["image_base64"].append(b_dict)
                elif file_type == "pdf":
                    text, images = await loop.run_in_executor(None, self._pdf_to_text_and_images, path)
                    if text.strip():
                        state_update["file_texts"].append({
                            "file_name": file_name,
                            "text": text.strip(),
                            "source": "pdf_parse"
                        })
                    state_update["image_base64"].extend(images)
            except Exception as e:
                logger.warning(f"Failed to process {path}: {e}")

        if ocr_texts:
            state_update["ocr_text"] = "\n\n".join(ocr_texts)

        has_images = len(state_update["image_base64"]) > 0 or len(state_update["image_paths"]) > 0
        has_text = len(state_update["file_texts"]) > 0

        if has_images and has_text:
            state_update["fusion_strategy"] = "full_multimodal"
        elif has_images:
            state_update["fusion_strategy"] = "ocr_only"
        else:
            state_update["fusion_strategy"] = "text_only"

        return state_update
