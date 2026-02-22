from fastapi import APIRouter, FastAPI, Depends, UploadFile, status, Request
from fastapi.responses import JSONResponse
import os
from helpers.config import get_settings, Settings
from controllers import DataController, ProjectController, ProcessController
from models import ResponseSignal
import aiofiles
import logging
from .schemes.data import ProcessRequest
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from models.db_schemes import DataChunk

logger = logging.getLogger("uvicorn.error") # to show the problems of something happend

data_router = APIRouter(
    prefix="/v1/data"
)

@data_router.post("/upload/{project_id}")

async def upload_data(request: Request, project_id: str, file: UploadFile,
                    app_settings: Settings = Depends(get_settings)):

    project_model = ProjectModel(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

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


@data_router.post("/process/{project_id}")
async def process_endpoint(request: Request, project_id: str, process_request: ProcessRequest):

    file_id = process_request.file_id
    chunk_size = process_request.chunk_size
    overlap_size = process_request.overlap_size
    do_reset = process_request.do_reset

    project_model = ProjectModel(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    process_controller = ProcessController(project_id=project_id)

    file_content = process_controller.get_file_content(file_id=file_id)

    file_chunks = process_controller.process_file_content(
        file_content=file_content,
        file_id=file_id,
        chunk_size=chunk_size,
        overlap_size=overlap_size
    )

    if file_chunks is None or len(file_chunks) == 0:
        return JSONResponse(
            content={
                "Signal": ResponseSignal.PROCESSING_FAILED
            }
        )

    file_chunks_records = [
        DataChunk(
            chunk_text=chunk.page_content,
            chunk_metadata=chunk.metadata,
            chunk_order=i+1,
            chunk_project_id=project.id
        )
        for i, chunk in enumerate(file_chunks)
    ]

    chunk_model = ChunkModel(
        db_client=request.app.db_client
    )

    if do_reset == 1:
        _ = await chunk_model.delete_chunks_by_project_id(
            project_id=project.id
        )

    no_records = await chunk_model.insert_many_chunks(chunks=file_chunks_records)

    return JSONResponse(
        content={
                "Signal": ResponseSignal.PROCESSING_SUCCESS.value,
                "Inserted_chunks": no_records
        }
    )
