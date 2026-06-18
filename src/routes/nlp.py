from fastapi import FastAPI, APIRouter, status, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from routes.schemes.nlp import (
    PushRequest, SearchRequest, VisualizeRequest,
    QuizAnswerRequest, QuizAnswerResponse, QuizGenerateResponse, DiagramGenerateResponse,
    AgentQueryRequest, AgentQueryResponse, MultimodalQueryResponse,
)
from agents.base import create_initial_state
from helpers.config import get_settings
import json
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from controllers import NLPController, ProcessController
from models import ResponseSignal
from tqdm.auto import tqdm
import os
import httpx
import base64
import asyncio
import shutil
import tempfile
from agents.multimodal.MultiFileProcessor import MultiFileProcessor
from agents.quiz.QuizAgent import QuizAgent
from agents.diagram.DiagramAgent import DiagramAgent
from stores.llm.LLMEnums import DocumentTypeEnum

import logging

logger = logging.getLogger("uvicorn.error")


def _normalize_answer_text(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


async def _read_generation_payload(request: Request):
    """Read JSON, form-url-encoded, or multipart payloads for tool endpoints.

    The new quiz/diagram endpoints must support both:
    - Streamlit multipart requests with uploaded files
    - JSON/form requests that rely on already indexed project chunks
    """
    content_type = request.headers.get("content-type", "").lower()
    data = {}
    files = []

    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        for key, value in form.multi_items():
            if hasattr(value, "filename") and hasattr(value, "read"):
                if getattr(value, "filename", None):
                    files.append(value)
            else:
                data[key] = value
        return data, files

    if "application/json" in content_type:
        try:
            json_data = await request.json()
            if isinstance(json_data, dict):
                data.update(json_data)
        except Exception:
            pass

    return data, files


async def _context_from_uploaded_files(uploaded_files) -> str:
    """Extract text/OCR context from uploaded files without persisting them."""
    if not uploaded_files:
        return ""

    tmp_dir = tempfile.mkdtemp()
    try:
        saved_paths = []
        for uploaded in uploaded_files:
            filename = os.path.basename(uploaded.filename or "uploaded_file")
            dest = os.path.join(tmp_dir, filename)
            with open(dest, "wb") as out:
                out.write(await uploaded.read())
            saved_paths.append(dest)

        process_controller = ProcessController(project_id="")
        loop = asyncio.get_event_loop()
        context_parts = []

        for path in saved_paths:
            result = await loop.run_in_executor(None, process_controller.extract_any_file, path)
            text = (result or {}).get("text", "").strip()
            if text:
                context_parts.append(
                    f"--- START OF FILE: {(result or {}).get('file_name', os.path.basename(path))} ---\n"
                    f"{text}\n"
                    f"--- END OF FILE ---"
                )

        return "\n\n".join(context_parts).strip()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


async def _context_from_project_chunks(request: Request, project_id: str, max_chunks: int = 120) -> str:
    """Load persisted chunks from PostgreSQL when project_id is an integer DB id."""
    if not str(project_id).isdigit():
        return ""

    try:
        project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
        project = await project_model.get_project_or_create_one(project_id=int(project_id))
        chunk_model = await ChunkModel.create_instance(db_client=request.app.db_client)
        chunks = await chunk_model.get_poject_chunks(
            project_id=project.project_id,
            page_no=1,
            page_size=max_chunks,
        )
    except Exception as exc:
        logger.warning(f"Could not load DB chunks for project_id={project_id}: {exc}")
        return ""

    context_parts = []
    for idx, chunk in enumerate(chunks or [], start=1):
        text = (chunk.chunk_text or "").strip()
        if text:
            context_parts.append(f"--- CHUNK {idx} ---\n{text}")

    return "\n\n".join(context_parts).strip()


async def _context_from_vector_search(request: Request, project_id: str, query_text: str, limit: int = 40) -> str:
    """Fallback context retrieval from the vector DB collection namespace."""
    try:
        nlp_controller = NLPController(
            vectordb_client=request.app.vectordb_client,
            generation_client=request.app.generation_client,
            embedding_client=request.app.embedding_client,
            template_parser=request.app.template_parser,
        )
        raw_vec = request.app.embedding_client.embed_text(
            text=query_text,
            document_type=DocumentTypeEnum.QUERY.value,
        )
        query_vector = nlp_controller._flatten_vector(raw_vec)
        if not any(query_vector):
            return ""

        collection_name = nlp_controller.create_collection_name(project_id=project_id)
        results = await request.app.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=query_vector,
            limit=limit,
        )
    except Exception as exc:
        logger.warning(f"Could not retrieve vector context for project_id={project_id}: {exc}")
        return ""

    context_parts = []
    for idx, doc in enumerate(results or [], start=1):
        text = getattr(doc, "text", "").strip()
        if text:
            context_parts.append(f"--- RETRIEVED CHUNK {idx} ---\n{text}")

    return "\n\n".join(context_parts).strip()


async def _build_generation_context(request: Request, project_id: str, uploaded_files, task: str) -> str:
    """Use uploaded files first; otherwise fall back to DB chunks/vector retrieval."""
    if uploaded_files:
        return await _context_from_uploaded_files(uploaded_files)

    context = await _context_from_project_chunks(request=request, project_id=project_id)
    if context:
        return context

    if task == "diagram":
        query_text = "lecture overview main topics subtopics relationships concept map summary"
    else:
        query_text = "lecture key concepts definitions important details examples assessment questions"

    return await _context_from_vector_search(
        request=request,
        project_id=project_id,
        query_text=query_text,
    )


nlp_router = APIRouter(
    prefix="/v1/nlp"
)

@nlp_router.post("/index/push/{project_id}")
async def index_project(request: Request, project_id: int, push_request: PushRequest):

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

    nlp_controller = NLPController(
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
    collection_name = nlp_controller.create_collection_name(project_id=project.project_id)

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

        is_inserted = await nlp_controller.index_into_vector_db(
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
async def get_project_index_info(request: Request, project_id: int):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    collection_info = await nlp_controller.get_vector_db_collection_info(project=project)

    return JSONResponse(
        content={
            "signal": ResponseSignal.VECTORDB_COLLECTION_RETRIEVED.value,
            "collection_info": collection_info
        }
    )

@nlp_router.post("/index/search/{project_id}")
async def search_index(request: Request, project_id: int, search_request: SearchRequest):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    results = await nlp_controller.search_vector_db_collection(
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
async def answer_rag(request: Request, project_id: int, search_request: SearchRequest):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    answer, full_prompt, chat_history = await nlp_controller.answer_rag_question(
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



@nlp_router.post("/quiz/generate/{project_id}", response_model=QuizGenerateResponse)
async def generate_interactive_quiz(request: Request, project_id: str):
    """Generate a structured, interactive MCQ quiz from uploaded files or indexed chunks."""
    try:
        payload, uploaded_files = await _read_generation_payload(request)
        num_questions = int(payload.get("num_questions", payload.get("count", 5)) or 5)
        language = payload.get("language", "English")

        context = await _build_generation_context(
            request=request,
            project_id=project_id,
            uploaded_files=uploaded_files,
            task="quiz",
        )

        if not context:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": "quiz_context_error",
                    "error": "No document context found. Upload files or process/index project documents first.",
                },
            )

        quiz_agent = QuizAgent(llm_provider=request.app.generation_client)
        quiz = quiz_agent.generate(
            context=context,
            num_questions=num_questions,
            language=language,
        )

        return JSONResponse(content=quiz)
    except ValueError as exc:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"signal": "quiz_validation_error", "error": str(exc)},
        )
    except Exception as exc:
        logger.error(f"Interactive quiz generation error: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": "quiz_generation_error", "error": str(exc)},
        )


@nlp_router.post("/quiz/answer", response_model=QuizAnswerResponse)
async def check_quiz_answer(answer_request: QuizAnswerRequest):
    """Stateless answer checker for non-persisted frontend quizzes."""
    selected = _normalize_answer_text(answer_request.selected_answer)
    correct = _normalize_answer_text(answer_request.correct_answer)
    is_correct = bool(selected and correct and selected == correct)

    if is_correct:
        message = "Correct answer."
    else:
        message = "Wrong answer. Review the hint and try again."

    return QuizAnswerResponse(
        quiz_id=answer_request.quiz_id,
        question_index=answer_request.question_index,
        is_correct=is_correct,
        message=message,
        explanation=answer_request.explanation if is_correct else None,
        hint=None if is_correct else answer_request.hint,
    )


@nlp_router.post("/diagram/generate/{project_id}", response_model=DiagramGenerateResponse)
async def generate_lecture_diagram(request: Request, project_id: str):
    """Generate a lecture-wide Mermaid diagram from uploaded files or indexed chunks."""
    try:
        payload, uploaded_files = await _read_generation_payload(request)
        language = payload.get("language", "English")
        diagram_type = payload.get("diagram_type", "flowchart")

        context = await _build_generation_context(
            request=request,
            project_id=project_id,
            uploaded_files=uploaded_files,
            task="diagram",
        )

        if not context:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": "diagram_context_error",
                    "error": "No document context found. Upload files or process/index project documents first.",
                },
            )

        diagram_agent = DiagramAgent(llm_provider=request.app.generation_client)
        diagram = diagram_agent.generate(
            context=context,
            language=language,
            diagram_type=diagram_type,
        )

        return JSONResponse(content=diagram)
    except ValueError as exc:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"signal": "diagram_validation_error", "error": str(exc)},
        )
    except Exception as exc:
        logger.error(f"Lecture diagram generation error: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"signal": "diagram_generation_error", "error": str(exc)},
        )

@nlp_router.post("/quiz/{project_id}")
async def get_quiz(request: Request, project_id: int):
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    quiz = await nlp_controller.generate_quiz(project=project)

    if not quiz:
        return JSONResponse(status_code=400, content={"signal": "quiz_error"})

    return JSONResponse(content={"signal": "quiz_success", "quiz": quiz})

@nlp_router.post("/summarize/{project_id}")
async def get_summary(request: Request, project_id: int):
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    # Streaming — prevents read timeout on large/OCR-scanned lectures.
    # The client receives text chunks word-by-word instead of waiting for the
    # full response, exactly like answer_stream does for chat.
    return StreamingResponse(
        nlp_controller.generate_summary_stream(project=project, limit=15),
        media_type="text/event-stream",
    )


@nlp_router.post("/index/answer_stream/{project_id}")
async def answer_rag_stream(request: Request, project_id: int, search_request: SearchRequest):
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    return StreamingResponse(
        nlp_controller.answer_rag_question_stream(project=project, query=search_request.text, limit=search_request.limit),
        media_type="text/event-stream"
    )

@nlp_router.post("/agent-query", response_model=AgentQueryResponse)
async def agent_query(request: Request, query_request: AgentQueryRequest):
    """
    Process a multi-agent RAG query.
    """
    try:
        initial_state = create_initial_state(
            query=query_request.query,
            project_id=query_request.project_id,
            asset_ids=query_request.asset_ids,
            image_paths=query_request.image_paths
        )
        initial_state["metadata"]["session_id"] = query_request.session_id

        result = await request.app.agent_graph.run(initial_state)

        # Save interaction via memory agent
        try:
            memory_agent = request.app.agent_graph._memory
            await memory_agent.save_interaction(result)
        except Exception as e:
            logger.warning(f"Failed to save interaction: {e}")

        settings = get_settings()

        response_kwargs = {
            "response": result.get("final_response", ""),
            "session_id": query_request.session_id
        }

        if settings.DEBUG_MODE:
            response_kwargs["agent_trace"] = result.get("agent_trace", [])
            response_kwargs["retrieved_chunks"] = result.get("retrieved_chunks", [])
            response_kwargs["metadata"] = result.get("metadata", {})

        return AgentQueryResponse(**response_kwargs)
    except Exception as e:
        logger.error(f"Agent query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@nlp_router.post("/agent-query/stream")
async def agent_query_stream(request: Request, query_request: AgentQueryRequest):
    """
    Process a multi-agent RAG query and stream the LLM answer as Server-Sent Events.

    The pipeline runs all agents up to and including ``ReasoningAgent`` to
    build the full context, then uses ``ResponseAgent.stream_execute`` to
    stream LLM answer tokens in real-time.  Each token is emitted as an
    SSE ``data:`` line.  A final ``data: [DONE]`` sentinel closes the stream.

    Request body
    ------------
    Same as ``POST /agent-query``.

    Returns
    -------
    StreamingResponse
        ``text/event-stream`` with one ``data: <token>`` line per LLM token.
    """
    try:
        initial_state = create_initial_state(
            query=query_request.query,
            project_id=query_request.project_id,
            asset_ids=query_request.asset_ids,
            image_paths=query_request.image_paths
        )
        initial_state["metadata"]["session_id"] = query_request.session_id
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    async def token_stream_generator():
        try:
            # Step 1: Run the full pipeline
            # The pipeline now ends at ResponseFormatterAgent, which sets final_response.
            # But wait, we want to stream the final_response!
            # ResponseFormatterAgent has stream_execute. We run up to the node before it,
            # or just run the whole pipeline without the final node if possible.
            # However, for simplicity, we will update the stream endpoint to just call
            # the formatter's stream_execute.
            pipeline_state = await request.app.agent_graph.run_up_to_formatter(initial_state)

            # Step 2: Stream LLM tokens via ResponseFormatterAgent
            response_formatter = request.app.agent_graph._response_formatter
            async for token in response_formatter.stream_execute(pipeline_state):
                if token:
                    yield f"data: {token}\n\n"

            # Step 3: Save interaction in memory
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
    files: list[UploadFile] = File(default=[])
):
    tmp_dir = tempfile.mkdtemp()
    try:
        saved_paths = []
        for f in files:
            # Check if file has an empty filename (happens when no files uploaded but form field present)
            if not f.filename:
                continue
            dest = os.path.join(tmp_dir, f.filename)
            with open(dest, "wb") as out:
                out.write(await f.read())
            saved_paths.append(dest)

        multi_processor = MultiFileProcessor()
        file_state = await multi_processor.process_files(saved_paths)

        initial_state = create_initial_state(
            query=query,
            project_id=project_id,
            asset_ids=[],
            image_paths=file_state.get("image_paths", []),
            uploaded_files=file_state.get("uploaded_files", []),
        )

        # Merge remaining file_state fields into initial_state
        initial_state.update({k: v for k, v in file_state.items()
                               if k not in ("image_paths", "uploaded_files")})
        initial_state["metadata"]["session_id"] = session_id

        result = await request.app.agent_graph.run_multimodal(initial_state)

        # Get memory agent safely to save interaction
        try:
            # Just extract response and query, mock save or rely on ReasoningAgent
            memory_agent = request.app.agent_graph._memory  # Accessing private memory agent for simplicity
            await memory_agent.save_interaction(result)
        except Exception as e:
            logger.error(f"Failed to save multimodal memory: {e}")

        # Hide internal states if debug disabled
        DEBUG_MODE = get_settings().DEBUG_MODE
        final_trace = result.get("agent_trace", []) if DEBUG_MODE else []
        final_chunks = result.get("retrieved_chunks", []) if DEBUG_MODE else []

        return MultimodalQueryResponse(
            response=result["final_response"],
            agent_trace=final_trace,
            retrieved_chunks=final_chunks,
            sources_used=result.get("sources_used", []),
            fusion_strategy=result.get("fusion_strategy", "text_only"),
            session_id=session_id,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@nlp_router.post("/multimodal-query/stream")
async def multimodal_query_stream(
    request: Request,
    query: str = Form(...),
    project_id: str = Form(...),
    session_id: str = Form(default="default"),
    visualize: bool = Form(default=False), # <-- تم إضافة متغير الرسم هنا
    files: list[UploadFile] = File(default=[])
):
    tmp_dir = tempfile.mkdtemp()

    saved_paths = []
    for f in files:
        if not f.filename:
            continue
        dest = os.path.join(tmp_dir, f.filename)
        with open(dest, "wb") as out:
            out.write(await f.read())
        saved_paths.append(dest)

    multi_processor = MultiFileProcessor()
    file_state = await multi_processor.process_files(saved_paths)

    initial_state = create_initial_state(
        query=query,
        project_id=project_id,
        asset_ids=[],
        image_paths=file_state.get("image_paths", []),
        uploaded_files=file_state.get("uploaded_files", []),
    )
    initial_state.update({k: v for k, v in file_state.items()
                           if k not in ("image_paths", "uploaded_files")})
    initial_state["metadata"]["session_id"] = session_id
    initial_state["metadata"]["visualize"] = visualize # حفظ اختيار المستخدم

    async def event_generator():
        try:
            state = await request.app.agent_graph.run_multimodal_up_to_formatter(initial_state)

            formatter = request.app.agent_graph._response_formatter
            full_text = ""
            async for token in formatter.stream_execute(state):
                full_text += token.replace("\\n", "\n")
                yield f"data: {token}\n\n"

            # 🔥 الجزء الجديد الخاص بتوليد الرسمة بعد انتهاء الكتابة
            if state.get("metadata", {}).get("visualize"):
                yield f"data: \\n\\n⏳ *جاري إنشاء رسوم توضيحية للملخص (Napkin AI)...*\\n\\n"

                state["final_response"] = full_text
                from agents.visualization.VisualizationAgent import VisualizationAgent
                vision_agent = VisualizationAgent()
                state = await vision_agent.execute(state)

                vis_urls = state.get("visualization_urls", [])
                if vis_urls:
                    yield f"data: \\n\\n### 🎨 رسوم ومخططات توضيحية:\\n\\n"
                    # عرض كل الصور تحت بعضها
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
