from fastapi import FastAPI, APIRouter, status, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from .schemes.nlp import (
    PushRequest, SearchRequest, VisualizeRequest,
    AgentQueryRequest, AgentQueryResponse, MultimodalQueryResponse,
    SummaryRequest,
)
from agents.base import create_initial_state
from core.settings import get_settings
import json
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from services.nlp_service import NLPService
from models import ResponseSignal
from tqdm.auto import tqdm
import os
import httpx
import base64
import asyncio
import shutil
import tempfile
from agents.multimodal.MultiFileProcessor import MultiFileProcessor

import logging

logger = logging.getLogger("uvicorn.error")

nlp_router = APIRouter(
    prefix="/v1/nlp"
)

@nlp_router.post("/index/push/{project_id}")
async def index_project(request: Request, project_id: str, push_request: PushRequest):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    chunk_model = await ChunkModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    if not project:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value
            }
        )

    nlp_service = NLPService(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    has_records = True
    page_no = 1
    inserted_items_count = 0
    idx = 0

    # create collection if not exists
    collection_name = nlp_service.create_collection_name(project_id=project.project_id)

    _ = await request.app.vectordb_client.create_collection(
        collection_name=collection_name,
        embedding_size=request.app.embedding_client.embedding_size,
        do_reset=push_request.do_reset,
    )

    # setup batching
    total_chunks_count = await chunk_model.get_total_chunks_count(project_id=project.project_id)
    pbar = tqdm(total=total_chunks_count, desc="Vector Indexing", position=0)

    while has_records:
        page_chunks = await chunk_model.get_poject_chunks(project_id=project.project_id, page_no=page_no)
        if len(page_chunks):
            page_no += 1

        if not page_chunks or len(page_chunks) == 0:
            has_records = False
            break

        chunks_ids =  [ c.chunk_id for c in page_chunks ]
        idx += len(page_chunks)

        is_inserted = await nlp_service.index_into_vector_db(
            project=project,
            chunks=page_chunks,
            chunks_ids=chunks_ids
        )

        if not is_inserted:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.INSERT_INTO_VECTORDB_ERROR.value
                }
            )

        pbar.update(len(page_chunks))
        inserted_items_count += len(page_chunks)

        await asyncio.sleep(10)

    return JSONResponse(
        content={
            "signal": ResponseSignal.INSERT_INTO_VECTORDB_SUCCESS.value,
            "inserted_items_count": inserted_items_count
        }
    )

@nlp_router.get("/index/info/{project_id}")
async def get_project_index_info(request: Request, project_id: str):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_service = NLPService(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    collection_info = await nlp_service.get_vector_db_collection_info(project=project)

    return JSONResponse(
        content={
            "signal": ResponseSignal.VECTORDB_COLLECTION_RETRIEVED.value,
            "collection_info": collection_info
        }
    )

@nlp_router.post("/index/search/{project_id}")
async def search_index(request: Request, project_id: str, search_request: SearchRequest):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_service = NLPService(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    results = await nlp_service.search_vector_db_collection(
        project=project, text=search_request.text, limit=search_request.limit
    )

    if not results:
        return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.VECTORDB_SEARCH_ERROR.value
                }
            )

    return JSONResponse(
        content={
            "signal": ResponseSignal.VECTORDB_SEARCH_SUCCESS.value,
            "results": [ result.dict()  for result in results ]
        }
    )

@nlp_router.post("/index/answer/{project_id}")
async def answer_rag(request: Request, project_id: str, search_request: SearchRequest):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_service = NLPService(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    answer, full_prompt, chat_history = await nlp_service.answer_rag_question(
        project=project,
        query=search_request.text,
        limit=search_request.limit,
    )

    if not answer:
        return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.RAG_ANSWER_ERROR.value
                }
        )

    return JSONResponse(
        content={
            "signal": ResponseSignal.RAG_ANSWER_SUCCESS.value,
            "answer": answer,
            "full_prompt": full_prompt,
            "chat_history": chat_history
        }
    )


@nlp_router.post("/visualize")
async def generate_visualization(request: VisualizeRequest):
    napkin_api_key = os.getenv("NAPKIN_API_KEY")
    if not napkin_api_key:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.NAPKIN_CONFIG_ERROR.value,
                "error": "Napkin API Key is missing"
            }
        )

    headers = {
        "Authorization": f"Bearer {napkin_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "content": request.text,
        "format": "png"
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            create_res = await client.post("https://api.napkin.ai/v1/visual", json=payload, headers=headers)
            if create_res.status_code not in [200, 201]:
                return JSONResponse(
                    status_code=create_res.status_code,
                    content={
                        "signal": ResponseSignal.NAPKIN_API_ERROR.value,
                        "error": create_res.text
                    }
                )

            req_id = create_res.json().get("id")

            status_state = "pending"
            status_data = {}
            for _ in range(20):
                await asyncio.sleep(3)
                status_res = await client.get(f"https://api.napkin.ai/v1/visual/{req_id}/status", headers=headers)
                status_data = status_res.json()
                status_state = status_data.get("status")
                if status_state != "pending":
                    break

            if status_state != "completed":
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "signal": ResponseSignal.NAPKIN_TIMEOUT_ERROR.value,
                        "error": "Generation failed or timed out"
                    }
                )

            # التعديل هنا لجلب كل الرسومات مش رسمة واحدة بس
            generated_files = status_data.get("generated_files", [])

            if not generated_files or len(generated_files) == 0:
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={
                        "signal": ResponseSignal.NAPKIN_DOWNLOAD_ERROR.value,
                        "error": "No files found in Napkin AI response."
                    }
                )

            images_base64 = []

            # حلقة تكرارية لتحميل كل الصور اللي Napkin ولدها
            for f in generated_files:
                file_url = f.get("url")
                if file_url:
                    file_res = await client.get(file_url, headers=headers)
                    if file_res.status_code == 200:
                        img_base64 = base64.b64encode(file_res.content).decode("utf-8")
                        images_base64.append(img_base64)

            if images_base64:
                return JSONResponse(
                    content={
                        "signal": ResponseSignal.NAPKIN_SUCCESS.value,
                        "images_base64": images_base64  # إرجاع مصفوفة من الصور
                    }
                )
            else:
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={
                        "signal": ResponseSignal.NAPKIN_DOWNLOAD_ERROR.value,
                        "error": "Failed to download generated images."
                    }
                )

    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "signal": ResponseSignal.NAPKIN_INTERNAL_ERROR.value,
                "error": str(e)
            }
        )


@nlp_router.post("/quiz/{project_id}")
async def get_quiz(request: Request, project_id: str):
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    nlp_service = NLPService(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    quiz = await nlp_service.generate_quiz(project=project)

    if not quiz:
        return JSONResponse(status_code=400, content={"signal": "quiz_error"})

    return JSONResponse(content={"signal": "quiz_success", "quiz": quiz})

@nlp_router.post("/summarize/{project_id}")
async def get_summary(request: Request, project_id: str, summary_request: SummaryRequest = None):
    """
    Generate a full structured lecture summary using map-reduce.

    Phase 6: now accepts an optional ``SummaryRequest`` body with
    ``asset_ids`` (limit to specific files) and ``language`` ("en"/"ar").

    Streams the summary token-by-token as ``text/event-stream`` using
    ``NLPController.generate_summary_stream`` with the new
    ``chunk_model`` parameter that triggers the map-reduce
    ``SummaryGenerator`` path over all ordered chunks.
    """
    if summary_request is None:
        summary_request = SummaryRequest()

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    chunk_model = await ChunkModel.create_instance(db_client=request.app.db_client)

    nlp_service = NLPService(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    return StreamingResponse(
        nlp_service.generate_summary_stream(
            project=project,
            chunk_model=chunk_model,
            asset_ids=summary_request.asset_ids or [],
            language=summary_request.language or "en",
        ),
        media_type="text/event-stream",
    )


@nlp_router.post("/index/answer_stream/{project_id}")
async def answer_rag_stream(request: Request, project_id: str, search_request: SearchRequest):
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    nlp_service = NLPService(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    return StreamingResponse(
        nlp_service.answer_rag_question_stream(project=project, query=search_request.text, limit=search_request.limit),
        media_type="text/event-stream"
    )

@nlp_router.post("/agent-query", response_model=AgentQueryResponse)
async def agent_query(request: Request, query_request: AgentQueryRequest):
    """
    Process a multi-agent RAG query (non-streaming).

    Phase 6: ``teaching_mode`` from the request is passed into the initial
    state so agents can adapt depth/style. ``citations`` are returned in the
    response when available (item 8).
    """
    try:
        initial_state = create_initial_state(
            query=query_request.query,
            project_id=query_request.project_id,
            asset_ids=query_request.asset_ids,
            image_paths=query_request.image_paths,
            teaching_mode=query_request.teaching_mode or "",
        )
        initial_state["metadata"]["session_id"] = query_request.session_id

        result = await request.app.agent_graph.run(initial_state)

        try:
            memory_agent = request.app.agent_graph._memory
            await memory_agent.save_interaction(result)
        except Exception as e:
            logger.warning(f"Failed to save interaction: {e}")

        settings = get_settings()

        response_kwargs = {
            "response":       result.get("final_response", ""),
            "session_id":     query_request.session_id,
            "teaching_mode":  result.get("teaching_mode", query_request.teaching_mode or ""),
            "citations":      result.get("citations", []),
        }

        if settings.DEBUG_MODE:
            response_kwargs["agent_trace"]      = result.get("agent_trace", [])
            response_kwargs["retrieved_chunks"] = result.get("retrieved_chunks", [])
            response_kwargs["metadata"]         = result.get("metadata", {})

        return AgentQueryResponse(**response_kwargs)
    except Exception as e:
        logger.error(f"Agent query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@nlp_router.post("/agent-query/stream")
async def agent_query_stream(request: Request, query_request: AgentQueryRequest):
    """
    Process a multi-agent RAG query and stream the LLM answer as SSE.

    Phase 6: ``teaching_mode`` is passed through to the initial state.
    """
    try:
        initial_state = create_initial_state(
            query=query_request.query,
            project_id=query_request.project_id,
            asset_ids=query_request.asset_ids,
            image_paths=query_request.image_paths,
            teaching_mode=query_request.teaching_mode or "",
        )
        initial_state["metadata"]["session_id"] = query_request.session_id
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    async def token_stream_generator():
        try:
            pipeline_state = await request.app.agent_graph.run_up_to_formatter(initial_state)

            response_formatter = request.app.agent_graph._response_formatter
            async for token in response_formatter.stream_execute(pipeline_state):
                if token:
                    yield f"data: {token}\n\n"

            try:
                memory_agent = request.app.agent_graph._memory
                await memory_agent.save_interaction(pipeline_state)
            except Exception as mem_exc:
                logger.warning(f"Stream: memory save failed (non-fatal): {mem_exc}")

        except Exception as e:
            logger.error(f"Agent streaming error: {e}")
            yield f"data: [Error: {str(e)}]\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        token_stream_generator(),
        media_type="text/event-stream"
    )

@nlp_router.post("/multimodal-query")
async def multimodal_query(
    request: Request,
    query: str = Form(...),
    project_id: str = Form(...),
    session_id: str = Form(default="default"),
    teaching_mode: str = Form(default=""),
    asset_ids: str = Form(default=""),
    files: list[UploadFile] = File(default=[])
):
    """
    Phase 6: added ``teaching_mode`` and ``asset_ids`` form fields.
    ``asset_ids`` is a comma-separated string of asset IDs (e.g. "1,2,3").
    ``citations`` are included in the response when indexed chunks were used.
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        saved_paths = []
        for f in files:
            if not f.filename:
                continue
            dest = os.path.join(tmp_dir, f.filename)
            with open(dest, "wb") as out:
                out.write(await f.read())
            saved_paths.append(dest)

        # Parse comma-separated asset_ids string into a list
        parsed_asset_ids = [a.strip() for a in asset_ids.split(",") if a.strip()] if asset_ids else []

        multi_processor = MultiFileProcessor()
        file_state = await multi_processor.process_files(saved_paths)

        initial_state = create_initial_state(
            query=query,
            project_id=project_id,
            asset_ids=parsed_asset_ids,
            image_paths=file_state.get("image_paths", []),
            uploaded_files=file_state.get("uploaded_files", []),
            teaching_mode=teaching_mode or "",
        )

        initial_state.update({k: v for k, v in file_state.items()
                               if k not in ("image_paths", "uploaded_files")})
        initial_state["metadata"]["session_id"] = session_id

        result = await request.app.agent_graph.run_multimodal(initial_state)

        try:
            memory_agent = request.app.agent_graph._memory
            await memory_agent.save_interaction(result)
        except Exception as e:
            logger.error(f"Failed to save multimodal memory: {e}")

        DEBUG_MODE = get_settings().DEBUG_MODE
        final_trace  = result.get("agent_trace", []) if DEBUG_MODE else []
        final_chunks = result.get("retrieved_chunks", []) if DEBUG_MODE else []

        return MultimodalQueryResponse(
            response=result["final_response"],
            agent_trace=final_trace,
            retrieved_chunks=final_chunks,
            sources_used=result.get("sources_used", []),
            fusion_strategy=result.get("fusion_strategy", "text_only"),
            session_id=session_id,
            citations=result.get("citations", []),
            teaching_mode=result.get("teaching_mode", teaching_mode or ""),
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@nlp_router.post("/multimodal-query/stream")
async def multimodal_query_stream(
    request: Request,
    query: str = Form(...),
    project_id: str = Form(...),
    session_id: str = Form(default="default"),
    teaching_mode: str = Form(default=""),
    asset_ids: str = Form(default=""),
    visualize: bool = Form(default=False),
    files: list[UploadFile] = File(default=[])
):
    """
    Phase 6: added ``teaching_mode`` and ``asset_ids`` form fields.
    ``asset_ids`` is a comma-separated string of asset IDs.
    """
    tmp_dir = tempfile.mkdtemp()

    saved_paths = []
    for f in files:
        if not f.filename:
            continue
        dest = os.path.join(tmp_dir, f.filename)
        with open(dest, "wb") as out:
            out.write(await f.read())
        saved_paths.append(dest)

    # Parse comma-separated asset_ids string into a list
    parsed_asset_ids = [a.strip() for a in asset_ids.split(",") if a.strip()] if asset_ids else []

    multi_processor = MultiFileProcessor()
    file_state = await multi_processor.process_files(saved_paths)

    initial_state = create_initial_state(
        query=query,
        project_id=project_id,
        asset_ids=parsed_asset_ids,
        image_paths=file_state.get("image_paths", []),
        uploaded_files=file_state.get("uploaded_files", []),
        teaching_mode=teaching_mode or "",
    )
    initial_state.update({k: v for k, v in file_state.items()
                           if k not in ("image_paths", "uploaded_files")})
    initial_state["metadata"]["session_id"] = session_id
    initial_state["metadata"]["visualize"] = visualize

    async def event_generator():
        try:
            state = await request.app.agent_graph.run_multimodal_up_to_formatter(initial_state)

            formatter = request.app.agent_graph._response_formatter
            full_text = ""
            async for token in formatter.stream_execute(state):
                full_text += token.replace("\\n", "\n")
                yield f"data: {token}\n\n"

            if state.get("metadata", {}).get("visualize"):
                yield f"data: \\n\\n⏳ *جاري إنشاء رسوم توضيحية للملخص (Napkin AI)...*\\n\\n"

                state["final_response"] = full_text
                from agents.visualization.VisualizationAgent import VisualizationAgent
                vision_agent = VisualizationAgent()
                state = await vision_agent.execute(state)

                vis_urls = state.get("visualization_urls", [])
                if vis_urls:
                    yield f"data: \\n\\n### 🎨 رسوم ومخططات توضيحية:\\n\\n"
                    for idx, v_url in enumerate(vis_urls):
                        md_img = f"![Visualization {idx+1}]({v_url})\\n\\n"
                        yield f"data: {md_img}\n\n"

            yield "data: [DONE]\n\n"
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

@nlp_router.get("/agent-trace/{session_id}")
async def get_agent_trace(request: Request, session_id: str):
    """
    Returns all memory records for a session.
    """
    try:
        memory_store = request.app.agent_graph._memory._memory_store
        records = await memory_store.retrieve_memories(query="*", session_id=session_id, k=50)
        return JSONResponse(content=[dict(r) for r in records])
    except Exception as e:
        logger.error(f"Agent trace error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
