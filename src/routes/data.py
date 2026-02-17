from fastapi import APIRouter, FastAPI, Depends, UploadFile, status
from fastapi.responses import JSONResponse
import os
from helpers.config import get_settings, Settings
from controllers import DataController, ProjectController
from models import ResponseSignal
import aiofiles
import logging

logger = logging.getLogger("uvicorn.error") # to show the problems of something happend

data_router = APIRouter(
    prefix="/v1/data"
)

@data_router.post("/upload/{project_id}")

async def upload_data(project_id: str, file: UploadFile,
                    app_settings: Settings = Depends(get_settings)):

    # Validate the file properties
    is_valid, result_signal = DataController().validate_uploaded_file(file=file)

    if not is_valid:
        return {
            "Signal": result_signal
        }

    project_dir_path = ProjectController().get_project_path(project_id=project_id)
    file_path, file_id = DataController().generate_unique_filepath(
        orig_file_name=file.filename,
        project_id=project_id
    )

    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK):
                await f.write(chunk)
    except Exception as e:

        logger.error(f"Error While Uploading File: {e}")

        return {
        "Signal": ResponseSignal.FILE_UPLOAD_FAILED
    }

    return JSONResponse(
            content= {
        "Signal": ResponseSignal.FILE_UPLOAD_SUCCESS.value,
        "File_id": file_id
    })
