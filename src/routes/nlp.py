from fastapi import FastAPI, APIRouter, status, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from routes.schemes.nlp import PushRequest, SearchRequest, VisualizeRequest, AgentQueryRequest, AgentQueryResponse
from agents.base import create_initial_state
import json
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from controllers import NLPController
from models import ResponseSignal
from tqdm.auto import tqdm
import os
import httpx
import base64
import asyncio

import logging

logger = logging.getLogger("uvicorn.error")

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

            # التعديل الصحيح لاستخراج الرابط المباشر
            generated_files = status_data.get("generated_files", [])

            if not generated_files or len(generated_files) == 0:
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={
                        "signal": ResponseSignal.NAPKIN_DOWNLOAD_ERROR.value,
                        "error": "No files found in Napkin AI response."
                    }
                )

            file_url = generated_files[0].get("url")

            file_res = await client.get(file_url, headers=headers)

            if file_res.status_code == 200:
                img_base64 = base64.b64encode(file_res.content).decode("utf-8")
                return JSONResponse(
                    content={
                        "signal": ResponseSignal.NAPKIN_SUCCESS.value,
                        "image_base64": img_base64
                    }
                )
            else:
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={
                        "signal": ResponseSignal.NAPKIN_DOWNLOAD_ERROR.value,
                        "error": f"Failed to download image. HTTP {file_res.status_code}: {file_res.text}"
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
        memory_agent = request.app.agent_graph._memory
        await memory_agent.save_interaction(result)
        
        return AgentQueryResponse(
            response=result.get("final_response", ""),
            agent_trace=result.get("agent_trace", []),
            retrieved_chunks=result.get("retrieved_chunks", []),
            session_id=query_request.session_id
        )
    except Exception as e:
        logger.error(f"Agent query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@nlp_router.post("/agent-query/stream")
async def agent_query_stream(request: Request, query_request: AgentQueryRequest):
    """
    Process a multi-agent RAG query with a streaming response.
    """
    initial_state = create_initial_state(
        query=query_request.query,
        project_id=query_request.project_id,
        asset_ids=query_request.asset_ids,
        image_paths=query_request.image_paths
    )
    initial_state["metadata"]["session_id"] = query_request.session_id

    async def stream_generator():
        try:
            async for chunk in request.app.agent_graph.stream(initial_state):
                yield f"data: {json.dumps(chunk, default=str)}\n\n"
        except Exception as e:
            logger.error(f"Agent streaming error: {e}")
            yield f"data: Error: {str(e)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_generator(),
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
