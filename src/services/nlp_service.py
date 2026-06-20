from models.db_schemes import Project, DataChunk
from stores.llm.LLMEnums import DocumentTypeEnum
from typing import List
import json

class NLPService:

    def __init__(self, vectordb_client, generation_client,
                 embedding_client, template_parser):
        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser

    def create_collection_name(self, project_id: str):
        return f"collection_{self.vectordb_client.default_vector_size}_{project_id}".strip()

    async def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return await self.vectordb_client.delete_collection(collection_name=collection_name)

    async def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = await self.vectordb_client.get_collection_info(collection_name=collection_name)
        return json.loads(json.dumps(collection_info, default=lambda x: x.__dict__))

    def flatten_vector(self, raw_vec):
        flat_floats = []
        def recurse(item):
            if isinstance(item, (list, tuple)):
                for i in item: recurse(i)
            elif hasattr(item, "tolist"):
                recurse(item.tolist())
            elif isinstance(item, str):
                try:
                    parsed = json.loads(item)
                    if isinstance(parsed, (list, tuple)): recurse(parsed)
                except: pass
            else:
                try: flat_floats.append(float(item))
                except: pass

        recurse(raw_vec)

        target_size = self.embedding_client.embedding_size
        if len(flat_floats) > target_size:
            flat_floats = flat_floats[:target_size]
        elif len(flat_floats) < target_size:
            flat_floats.extend([0.0] * (target_size - len(flat_floats)))

        return flat_floats

    async def index_into_vector_db(self, project: Project, chunks: List[DataChunk],
                                   chunks_ids: List[int], do_reset: bool = False):
        """
        Embed *chunks* and insert them into the project's vector DB
        collection in one call (kept for backward compatibility with
        ``/v1/nlp/index/push``).

        For the async processing pipeline (``/v1/data/process``), prefer
        calling ``embed_chunks`` and ``insert_chunks_into_vector_db``
        separately so the caller can report ``EMBEDDING`` / ``INDEXING``
        status transitions between the two steps.
        """
        vectors = self.embed_chunks(chunks=chunks)

        return await self.insert_chunks_into_vector_db(
            project=project,
            chunks=chunks,
            chunks_ids=chunks_ids,
            vectors=vectors,
            do_reset=do_reset,
        )

    def embed_chunks(self, chunks: List[DataChunk]) -> List[List[float]]:
        """Compute embedding vectors for *chunks* (the "EMBEDDING" step)."""
        texts = [c.chunk_text.replace('\x00', '').strip() for c in chunks]

        vectors = []
        for text in texts:
            raw_vec = self.embedding_client.embed_text(text=text, document_type=DocumentTypeEnum.DOCUMENT.value)
            flat_vec = self.flatten_vector(raw_vec)
            vectors.append(flat_vec)

        return vectors

    async def insert_chunks_into_vector_db(self, project: Project, chunks: List[DataChunk],
                                            chunks_ids: List[int], vectors: List[List[float]],
                                            do_reset: bool = False):
        """Create (if needed) the project's collection and insert pre-computed
        *vectors* for *chunks* (the "INDEXING" step)."""
        collection_name = self.create_collection_name(project_id=project.project_id)
        texts = [c.chunk_text.replace('\x00', '').strip() for c in chunks]
        metadata = []
        for chunk, chunk_id in zip(chunks, chunks_ids):
            chunk_metadata = dict(chunk.chunk_metadata or {})
            chunk_metadata["asset_id"] = chunk.chunk_asset_id
            chunk_metadata["chunk_id"] = chunk_id
            metadata.append(chunk_metadata)

        _ = await self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset,
        )

        _ = await self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            metadata=metadata,
            vectors=vectors,
            record_ids=chunks_ids,
        )
        return True

    async def search_vector_db_collection(self, project: Project, text: str, limit: int = 10):
        collection_name = self.create_collection_name(project_id=project.project_id)
        raw_vec = self.embedding_client.embed_text(text=text, document_type=DocumentTypeEnum.QUERY.value)

        query_vector = self.flatten_vector(raw_vec)

        if not any(query_vector):
            return False

        results = await self.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=query_vector,
            limit=limit
        )
        # search_by_vector returns None when collection is empty — normalise to []
        return results if results is not None else []

    def _rerank_documents(self, query: str, retrieved_documents: list, top_n: int):
        if not retrieved_documents:
            return []

        if not hasattr(self.generation_client, "rerank"):
            return retrieved_documents[:top_n]

        docs_texts = [getattr(doc, "text", "") for doc in retrieved_documents]

        try:
            reranked_texts = self.generation_client.rerank(query=query, documents=docs_texts, top_n=top_n)

            class DummyDoc:
                def __init__(self, text):
                    self.text = text

            return [DummyDoc(t) for t in reranked_texts]
        except Exception as e:
            return retrieved_documents[:top_n]

    async def answer_rag_question(self, project: Project, query: str, limit: int = 10):
        fetch_limit = limit * 3
        retrieved_documents = await self.search_vector_db_collection(
            project=project, text=query, limit=fetch_limit,
        )

        retrieved_documents = self._rerank_documents(query=query, retrieved_documents=retrieved_documents, top_n=limit)

        if not retrieved_documents or len(retrieved_documents) == 0:
            return None, None, None

        system_prompt = self.template_parser.get("rag", "system_prompt")

        documents_prompts = "\n".join([
            self.template_parser.get("rag", "document_prompt", {
                "doc_num": idx + 1,
                "chunk_text": self.generation_client.process_text(getattr(doc, "text", "")),
            })
            for idx, doc in enumerate(retrieved_documents)
        ])

        footer_prompt = self.template_parser.get("rag", "footer_prompt", {"query": query})

        chat_history = [
            self.generation_client.construct_prompt(prompt=system_prompt, role=self.generation_client.enums.SYSTEM.value)
        ]

        full_prompt = "\n\n".join([documents_prompts, footer_prompt])

        answer = self.generation_client.generate_text(prompt=full_prompt, chat_history=chat_history)

        return answer, full_prompt, chat_history

    async def answer_rag_question_stream(self, project: Project, query: str, limit: int = 10):
        fetch_limit = limit * 3
        retrieved_documents = await self.search_vector_db_collection(
            project=project, text=query, limit=fetch_limit,
        )

        retrieved_documents = self._rerank_documents(query=query, retrieved_documents=retrieved_documents, top_n=limit)

        if not retrieved_documents or len(retrieved_documents) == 0:
            yield "Sorry, I could not find an answer in this document."
            return

        system_prompt = self.template_parser.get("rag", "system_prompt")

        documents_prompts = "\n".join([
            self.template_parser.get("rag", "document_prompt", {
                "doc_num": idx + 1,
                "chunk_text": self.generation_client.process_text(getattr(doc, "text", "")),
            })
            for idx, doc in enumerate(retrieved_documents)
        ])

        full_prompt = "\n\n".join([documents_prompts, self.template_parser.get("rag", "footer_prompt", {"query": query})])
        chat_history = [self.generation_client.construct_prompt(prompt=system_prompt, role=self.generation_client.enums.SYSTEM.value)]

        async for chunk in self.generation_client.generate_stream(prompt=full_prompt, chat_history=chat_history):
            yield chunk

    async def generate_quiz(self, project: Project, limit: int = 5):
        query_text = "key concepts, main definitions, summary, important details"
        fetch_limit = limit * 3
        retrieved_documents = await self.search_vector_db_collection(
            project=project,
            text=query_text,
            limit=fetch_limit,
        )

        retrieved_documents = self._rerank_documents(query=query_text, retrieved_documents=retrieved_documents, top_n=limit)

        if not retrieved_documents:
            return None

        documents_prompts = "\n".join([
            self.template_parser.get("rag", "document_prompt", {
                "doc_num": idx + 1,
                "chunk_text": getattr(doc, "text", ""),
            })
            for idx, doc in enumerate(retrieved_documents)
        ])

        system_prompt = self.template_parser.get("rag", "quiz_system_prompt", {"num_questions": 3})
        footer_prompt = self.template_parser.get("rag", "quiz_footer_prompt", {"num_questions": 3})

        chat_history = [
            self.generation_client.construct_prompt(prompt=system_prompt, role=self.generation_client.enums.SYSTEM.value)
        ]

        full_prompt = "\n\n".join([documents_prompts, footer_prompt])

        quiz = self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=chat_history,
            max_output_tokens=2000
        )
        return quiz

    async def generate_summary(self, project: Project, limit: int = 15,
                                chunk_model=None, asset_ids: list = None,
                                language: str = "en"):
        """
        Generate a full structured lecture summary.

        When *chunk_model* is supplied the method uses map-reduce over ALL
        ordered chunks for the project (items 1 + 4C).  This is the path
        taken by the agent pipeline and the /v1/nlp/summarize route after
        Phase 1.

        Falls back to the old similarity-search approach when *chunk_model*
        is None so that any existing callers that don't pass it keep working.
        """
        # ── Full-lecture map-reduce path (preferred) ─────────────────────
        if chunk_model is not None:
            from agents.response.SummaryGenerator import SummaryGenerator

            chunks = await chunk_model.get_all_chunks_ordered(
                project_id=project.project_id,
                asset_ids=asset_ids or [],
                max_chunks=2000,
            )

            if not chunks:
                return None

            generator = SummaryGenerator(
                generation_client=self.generation_client,
                template_parser=self.template_parser,
                language=language,
            )
            return generator.generate(chunks=chunks)

        # ── Legacy similarity-search path (fallback) ─────────────────────
        query_text = "overview, main concepts, summary, introduction, conclusion, important details"
        fetch_limit = limit * 2
        retrieved_documents = await self.search_vector_db_collection(
            project=project,
            text=query_text,
            limit=fetch_limit,
        )

        retrieved_documents = self._rerank_documents(query=query_text, retrieved_documents=retrieved_documents, top_n=limit)

        if not retrieved_documents:
            return None

        documents_prompts = "\n".join([
            self.template_parser.get("rag", "document_prompt", {
                "doc_num": idx + 1,
                "chunk_text": getattr(doc, "text", ""),
            })
            for idx, doc in enumerate(retrieved_documents)
        ])

        system_prompt = self.template_parser.get("rag", "summarize_system_prompt")
        footer_prompt = self.template_parser.get("rag", "summarize_footer_prompt")

        chat_history = [
            self.generation_client.construct_prompt(prompt=system_prompt, role=self.generation_client.enums.SYSTEM.value)
        ]

        full_prompt = "\n\n".join([documents_prompts, footer_prompt])

        summary = self.generation_client.generate_text(
                prompt=full_prompt,
                chat_history=chat_history,
                max_output_tokens=4000
        )
        return summary

    async def generate_summary_stream(self, project: Project, limit: int = 15,
                                       chunk_model=None, asset_ids: list = None,
                                       language: str = "en"):
        """
        Streaming version of generate_summary.

        When *chunk_model* is supplied, uses the map-reduce SummaryGenerator
        over all ordered chunks and streams the final reduce step.
        Falls back to the legacy similarity-search single-prompt approach
        when *chunk_model* is None.
        """
        # ── Full-lecture map-reduce streaming path (preferred) ────────────
        if chunk_model is not None:
            from agents.response.SummaryGenerator import SummaryGenerator

            chunks = await chunk_model.get_all_chunks_ordered(
                project_id=project.project_id,
                asset_ids=asset_ids or [],
                max_chunks=2000,
            )

            if not chunks:
                yield "I could not find any indexed content for this project. Please process the lecture first."
                return

            generator = SummaryGenerator(
                generation_client=self.generation_client,
                template_parser=self.template_parser,
                language=language,
            )
            async for token in generator.generate_stream(chunks=chunks):
                yield token
            return

        # ── Legacy single-prompt streaming path ───────────────────────────
        query_text = "overview, main concepts, summary, introduction, conclusion, important details"
        fetch_limit = limit * 2
        retrieved_documents = await self.search_vector_db_collection(
            project=project,
            text=query_text,
            limit=fetch_limit,
        )

        retrieved_documents = self._rerank_documents(
            query=query_text,
            retrieved_documents=retrieved_documents,
            top_n=limit,
        )

        if not retrieved_documents:
            yield "I could not find enough information in the provided documents to generate a summary."
            return

        documents_prompts = "\n".join([
            self.template_parser.get("rag", "document_prompt", {
                "doc_num": idx + 1,
                "chunk_text": getattr(doc, "text", ""),
            })
            for idx, doc in enumerate(retrieved_documents)
        ])

        system_prompt = self.template_parser.get("rag", "summarize_system_prompt")
        footer_prompt = self.template_parser.get("rag", "summarize_footer_prompt")

        chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt,
                role=self.generation_client.enums.SYSTEM.value,
            )
        ]

        full_prompt = "\n\n".join([documents_prompts, footer_prompt])

        async for chunk in self.generation_client.generate_stream(
            prompt=full_prompt,
            chat_history=chat_history,
            max_output_tokens=4000,
        ):
            yield chunk