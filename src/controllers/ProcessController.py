from .BaseController import BaseController
from .ProjectController import ProjectController
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from models import ProcessingEnum
import pytesseract
from PIL import Image
import fitz

class ImageLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        try:
            image = Image.open(self.file_path)
            text = pytesseract.image_to_string(image)
            return [Document(page_content=text, metadata={"source": self.file_path, "page": 1})]
        except Exception as e:
            print(f"Error extracting text from image {self.file_path}: {e}")
            return []

class EnhancedPyMuPDFLoader(PyMuPDFLoader):
    def load(self):
        docs = super().load()
        for doc in docs:
            # If the extracted text is too short, it might be a scanned page. Let's try OCR.
            if len(doc.page_content.strip()) < 50:
                try:
                    pdf_doc = fitz.open(self.file_path)
                    page_num = doc.metadata.get("page", 0)
                    page = pdf_doc.load_page(page_num)
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = pytesseract.image_to_string(img)
                    if len(text.strip()) > len(doc.page_content.strip()):
                        doc.page_content = text
                except Exception as e:
                    print(f"Error performing OCR on PDF page {self.file_path}: {e}")
        return docs

class ProcessController(BaseController):

    def __init__(self, project_id: str):
        super().__init__()

        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)

    def get_file_extension(self, file_id: str):
        return os.path.splitext(file_id)[-1]

    def get_file_loader(self, file_id: str):

        file_ext = self.get_file_extension(file_id=file_id)
        file_path = os.path.join(
            self.project_path,
            file_id
        )

        if not os.path.exists(file_path):
            return None

        if file_ext == ProcessingEnum.TXT.value:
            return TextLoader(file_path, encoding="utf-8")

        if file_ext == ProcessingEnum.PDF.value:
            return EnhancedPyMuPDFLoader(file_path)

        if file_ext in [ProcessingEnum.PNG.value, ProcessingEnum.JPG.value, ProcessingEnum.JPEG.value]:
            return ImageLoader(file_path)

        return None

    def get_file_content(self, file_id: str):

        loader = self.get_file_loader(file_id=file_id)
        if loader:
            return loader.load()

        return None

    def process_file_content(self, file_content: list, file_id: str,
                            chunk_size: int=100, overlap_size: int=20):

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            length_function=len,
        )

        file_content_texts = [
            rec.page_content
            for rec in file_content
        ]

        file_content_metadata = [
            rec.metadata
            for rec in file_content
        ]

        chunks = text_splitter.create_documents(
            file_content_texts,
            metadatas=file_content_metadata
        )

        return chunks
