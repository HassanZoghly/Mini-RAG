"""
Data routes — upload, async process/index pipeline, status polling, assets.

Key changes (Phase 1):

1. POST /v1/data/process/{project_id}
   Now schedules a BackgroundTask and returns immediately with status=PROCESSING.
   The background job walks through EXTRACTING → CHUNKING → EMBEDDING → INDEXING → READY
   (or FAILED), writing each transition to ``processing_status_store`` so the
   frontend can poll without blocking on a long HTTP request.

2. GET /v1/data/status/{project_id}
   New endpoint. Returns the current processing status for a project so
   Streamlit (and any other client) can poll until status == READY.

3. GET /v1/data/assets/{project_id}
   New endpoint. Returns the list of uploaded assets for a project so the
   frontend can let the user pick which lecture(s) to chat about.

4. The original synchronous process logic is unchanged for any callers
   that still need it — they call the helper ``_run_process_and_index``
   directly.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, Request, UploadFile, status
from fastapi.responses import JSONResponse

from controllers import DataController, NLPController, ProcessController, ProjectController
from helpers.config import Settings, get_settings
from models import ResponseSignal
from models.AssetModel import AssetModel
from models.ChunkModel import ChunkModel
from models.ProjectModel import ProjectModel
from models.db_schemes import Asset, DataChunk
from models.enums.AssetTypeEnum import AssetTypeEnum
from utils.processing_status import ProcessingStatus, processing_status_store

from .schemes.data import ProcessRequest

logger = logging.getLogger("uvicorn.error")

data_router = APIRouter(prefix="/v1/data")


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@data_router.post("/upload/{project_id}")
async def upload_data(
    request: Request,
    project_id: str,
    file: UploadFile,
    app_settings: Settings = Depends(get_settings),
):
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    data_controller = DataController()
    is_valid, result_signal = data_controller.validate_uploaded_file(file=file)

    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": result_signal},
        )

    ProjectController().get_project_path(project_id=project_id)
    file_path, file_id = data_controller.generate_unique_filepath(
        orig_file_name=file.filename,
        project_id=project_id,
    )

    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK):
                await f.write(chunk)
    except Exception as exc:
        logger.error(f"Error while uploading file: {exc}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.FILE_UPLOAD_FAILED.value},
        )

    asset_model = await AssetModel.create_instance(db_client=request.app.db_client)
    asset_resource = Asset(
        asset_project_id=project.project_id,
        asset_type=AssetTypeEnum.FILE.value,
        asset_name=file_id,
        asset_size=os.path.getsize(file_path),
    )
    asset_record = await asset_model.create_asset(asset=asset_resource)

    # Mark the project as "UPLOADED" — not yet processed.
    await processing_status_store.set_status(
        project_id=str(project_id),
        status=ProcessingStatus.UPLOADED,
        detail=f"File '{file.filename}' uploaded, awaiting processing.",
    )

    return JSONResponse(
        content={
            "signal": ResponseSignal.FILE_UPLOAD_SUCCESS.value,
            "file_id": str(asset_record.asset_name),
        }
    )


# ---------------------------------------------------------------------------
# Background processing job
# ---------------------------------------------------------------------------

async def _run_process_and_index(
    project_id: str,
    process_request: ProcessRequest,
    db_client,
    vectordb_client,
    generation_client,
    embedding_client,
    template_parser,
):
    """
    Full extract → chunk → embed → index pipeline, run as a background task.

    Status transitions written to ``processing_status_store``:
        PROCESSING → EXTRACTING → CHUNKING → EMBEDDING → INDEXING → READY
        (FAILED on any unhandled exception)
    """
    pid = str(project_id)

    try:
        await processing_status_store.set_status(pid, ProcessingStatus.PROCESSING,
                                                  detail="Starting processing pipeline…")

        # ── Project / assets ────────────────────────────────────────────
        project_model = await ProjectModel.create_instance(db_client=db_client)
        project = await project_model.get_project_or_create_one(project_id=project_id)

        asset_model = await AssetModel.create_instance(db_client=db_client)
        chunk_model = await ChunkModel.create_instance(db_client=db_client)

        nlp_controller = NLPController(
            vectordb_client=vectordb_client,
            generation_client=generation_client,
            embedding_client=embedding_client,
            template_parser=template_parser,
        )

        # ── Resolve file list ────────────────────────────────────────────
        project_files_ids: dict = {}
        if process_request.file_id:
            asset_record = await asset_model.get_asset_record(
                asset_project_id=project.project_id,
                asset_name=process_request.file_id,
            )
            if asset_record is None:
                await processing_status_store.set_status(
                    pid, ProcessingStatus.FAILED,
                    detail=ResponseSignal.FILE_ID_ERROR.value,
                )
                return
            project_files_ids = {asset_record.asset_id: asset_record.asset_name}
        else:
            project_files = await asset_model.get_all_project_assets(
                asset_project_id=project.project_id,
                asset_type=AssetTypeEnum.FILE.value,
            )
            project_files_ids = {r.asset_id: r.asset_name for r in project_files}

        if not project_files_ids:
            await processing_status_store.set_status(
                pid, ProcessingStatus.FAILED,
                detail=ResponseSignal.NO_FILES_ERROR.value,
            )
            return

        # ── Optional reset ────────────────────────────────────────────────
        if process_request.do_reset == 1:
            collection_name = nlp_controller.create_collection_name(project_id=project.project_id)
            await vectordb_client.delete_collection(collection_name=collection_name)
            await chunk_model.delete_chunks_by_project_id(project_id=project.project_id)

        process_controller = ProcessController(project_id=project_id)
        total_files = len(project_files_ids)
        no_records = 0
        no_files = 0

        for file_idx, (asset_id, file_id) in enumerate(project_files_ids.items(), start=1):
            file_label = f"file {file_idx}/{total_files} ({file_id})"

            # ── EXTRACTING ──────────────────────────────────────────────
            await processing_status_store.set_status(
                pid, ProcessingStatus.EXTRACTING,
                detail=f"Extracting text from {file_label}…",
                files_done=file_idx - 1, files_total=total_files,
            )
            # Offload blocking I/O to the thread pool so the event loop stays free.
            file_content = await asyncio.get_event_loop().run_in_executor(
                None, process_controller.get_file_content, file_id
            )

            if file_content is None:
                logger.error(f"Could not extract content from {file_id}")
                continue

            # ── CHUNKING ────────────────────────────────────────────────
            await processing_status_store.set_status(
                pid, ProcessingStatus.CHUNKING,
                detail=f"Chunking {file_label}…",
            )
            file_chunks = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda fc=file_content, fi=file_id: process_controller.process_file_content(
                    file_content=fc,
                    file_id=fi,
                    chunk_size=process_request.chunk_size,
                    overlap_size=process_request.overlap_size,
                ),
            )

            if not file_chunks:
                await processing_status_store.set_status(
                    pid, ProcessingStatus.FAILED,
                    detail=f"Chunking produced no output for {file_label}.",
                )
                return

            file_chunks_records = [
                DataChunk(
                    chunk_text=chunk.page_content,
                    chunk_metadata=chunk.metadata,
                    chunk_order=i + 1,
                    chunk_project_id=project.project_id,
                    chunk_asset_id=asset_id,
                )
                for i, chunk in enumerate(file_chunks)
            ]

            inserted = await chunk_model.insert_many_chunks(chunks=file_chunks_records)

            # Fetch back the inserted records to get their DB-assigned IDs.
            from sqlalchemy.future import select as sa_select
            from models.db_schemes import DataChunk as DataChunkScheme
            async with db_client() as session:
                stmt = sa_select(DataChunkScheme).where(
                    DataChunkScheme.chunk_asset_id == asset_id,
                    DataChunkScheme.chunk_project_id == project.project_id,
                ).order_by(DataChunkScheme.chunk_order)
                result = await session.execute(stmt)
                saved_chunks = result.scalars().all()

            chunks_ids = [c.chunk_id for c in saved_chunks]
            saved_chunk_list = list(saved_chunks)

            # ── EMBEDDING ───────────────────────────────────────────────
            await processing_status_store.set_status(
                pid, ProcessingStatus.EMBEDDING,
                detail=f"Generating embeddings for {file_label} ({len(saved_chunk_list)} chunks)…",
            )
            vectors = await asyncio.get_event_loop().run_in_executor(
                None, nlp_controller.embed_chunks, saved_chunk_list
            )

            # ── INDEXING ────────────────────────────────────────────────
            await processing_status_store.set_status(
                pid, ProcessingStatus.INDEXING,
                detail=f"Inserting vectors for {file_label} into vector DB…",
            )
            await nlp_controller.insert_chunks_into_vector_db(
                project=project,
                chunks=saved_chunk_list,
                chunks_ids=chunks_ids,
                vectors=vectors,
                do_reset=(process_request.do_reset == 1 and no_files == 0),
            )

            no_records += inserted
            no_files += 1

        if no_files == 0:
            await processing_status_store.set_status(
                pid,
                ProcessingStatus.FAILED,
                detail="No files could be extracted or indexed.",
                files_done=0,
                files_total=total_files,
                chunks_indexed=0,
            )
            return

        # ── READY ────────────────────────────────────────────────────────
        await processing_status_store.set_status(
            pid, ProcessingStatus.READY,
            detail=f"Done. {no_files} file(s) processed, {no_records} chunks indexed.",
            files_done=no_files, files_total=total_files, chunks_indexed=no_records,
        )
        logger.info(f"[process/{project_id}] READY — {no_files} files, {no_records} chunks")

    except Exception as exc:
        logger.exception(f"[process/{project_id}] Background job failed: {exc}")
        await processing_status_store.set_status(
            pid, ProcessingStatus.FAILED,
            detail=f"Unexpected error: {exc}",
        )


# ---------------------------------------------------------------------------
# Process endpoint (now async — returns immediately, job runs in background)
# ---------------------------------------------------------------------------

@data_router.post("/process/{project_id}")
async def process_endpoint(
    request: Request,
    project_id: str,
    process_request: ProcessRequest,
    background_tasks: BackgroundTasks,
):
    """
    Schedule the extract → chunk → embed → index pipeline as a background
    task and return immediately.  Poll ``GET /v1/data/status/{project_id}``
    to track progress.
    """
    # Immediately stamp PROCESSING so the frontend doesn't see the old UPLOADED state.
    await processing_status_store.set_status(
        str(project_id),
        ProcessingStatus.PROCESSING,
        detail="Job queued, starting shortly…",
    )

    background_tasks.add_task(
        _run_process_and_index,
        project_id=project_id,
        process_request=process_request,
        db_client=request.app.db_client,
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "signal": "processing_started",
            "status": ProcessingStatus.PROCESSING.value,
            "project_id": project_id,
            "message": f"Processing started in background. Poll /v1/data/status/{project_id} for updates.",
        },
    )


# ---------------------------------------------------------------------------
# Status endpoint (new — item 3)
# ---------------------------------------------------------------------------

@data_router.get("/status/{project_id}")
async def get_processing_status(project_id: str):
    """
    Return the current processing status for *project_id*.

    Response fields:
        status      — one of UPLOADED | PROCESSING | EXTRACTING | CHUNKING |
                      EMBEDDING | INDEXING | READY | FAILED  (null if never started)
        detail      — human-readable detail message
        updated_at  — Unix timestamp of the last status update
        progress    — dict with extra counters (files_done, files_total, chunks_indexed)
        is_ready    — bool convenience flag: True when status == READY
        is_failed   — bool convenience flag: True when status == FAILED
    """
    entry = await processing_status_store.get_status(str(project_id))
    current_status = entry.get("status")
    return JSONResponse(
        content={
            "project_id": project_id,
            "status": current_status,
            "detail": entry.get("detail", ""),
            "updated_at": entry.get("updated_at"),
            "progress": entry.get("progress", {}),
            "is_ready": current_status == ProcessingStatus.READY.value,
            "is_failed": current_status == ProcessingStatus.FAILED.value,
        }
    )


# ---------------------------------------------------------------------------
# Assets endpoint (new — item 3, used by frontend lecture selector)
# ---------------------------------------------------------------------------

@data_router.get("/assets/{project_id}")
async def get_project_assets(request: Request, project_id: str):
    """
    Return the list of uploaded assets (lecture files) for *project_id*.
    Used by the Streamlit frontend to populate the lecture selector and
    to pass ``asset_ids`` to the agent query endpoints.
    """
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    asset_model = await AssetModel.create_instance(db_client=request.app.db_client)
    assets = await asset_model.get_all_project_assets(
        asset_project_id=project.project_id,
        asset_type=AssetTypeEnum.FILE.value,
    )

    return JSONResponse(
        content={
            "project_id": project_id,
            "assets": [
                {
                    "asset_id": str(a.asset_id),
                    "asset_name": a.asset_name,
                    "asset_size": a.asset_size,
                    "asset_type": a.asset_type,
                }
                for a in assets
            ],
        }
    )