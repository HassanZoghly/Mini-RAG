from fastapi import UploadFile
from models import ResponseSignal
import re
import os
import json
import random
import string
from core.settings import get_settings
from services.project_service import ProjectService

class DataService:

    ALLOWED_EXTENSIONS = {
        ".pdf",
        ".txt", ".md", ".csv", ".json", ".html", ".xml", ".yaml", ".yml",
        ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif",
    }

    def __init__(self):
        self.app_settings = get_settings()
        self.size_scale = 1048576 # Convert MB to bytes

    def generate_random_string(self, length=12):
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

    def validate_uploaded_file(self, file: UploadFile):
        allowed_types = self.app_settings.FILE_ALLOWED_TYPES
        if isinstance(allowed_types, str):
            try:
                allowed_types = json.loads(allowed_types)
            except json.JSONDecodeError:
                allowed_types = [
                    item.strip()
                    for item in allowed_types.split(",")
                    if item.strip()
                ]

        file_ext = os.path.splitext(file.filename or "")[-1].lower()
        content_type = file.content_type or ""

        if content_type not in allowed_types and file_ext not in self.ALLOWED_EXTENSIONS:
            return False, ResponseSignal.FILE_TYPE_NOT_SUPPORTED

        if file.size is not None and file.size > self.app_settings.FILE_MAX_SIZE * self.size_scale:
            return False, ResponseSignal.FILE_SIZE_EXCEEDED

        return True, ResponseSignal.FILE_UPLOAD_SUCCESS

    def generate_unique_filepath(self, orig_file_name: str, project_id: str):
        random_key = self.generate_random_string()
        project_path = ProjectService().get_project_path(project_id=project_id)

        cleaned_filename = self.get_clean_filename(orig_file_name=orig_file_name)
        new_file_path = os.path.join(
            project_path,
            random_key + "_" + cleaned_filename
        )

        while os.path.exists(new_file_path):
            random_key = self.generate_random_string()
            new_file_path = os.path.join(
                project_path,
                random_key + "_" + cleaned_filename
            )

        return new_file_path, random_key + "_" + cleaned_filename

    def get_clean_filename(self, orig_file_name: str):
        Cleaned_filename = re.sub(r'[^\w.]', '', (orig_file_name or "").strip())
        Cleaned_filename = Cleaned_filename.replace(" ", "_")
        if not Cleaned_filename:
            Cleaned_filename = "upload"

        return Cleaned_filename
