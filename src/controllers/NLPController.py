from .BaseController import BaseController
from models.db_schemes import Project, DataChunk
from stores.llm.LLMEnums import DocumentTypeEnum
from typing import List
import json

class NLPController(BaseController):

    def __init__(self, vectordb_client, generation_client,
                 embedding_client, template_parser):
        super().__init__()
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

    def _flatten_vector(self, raw_vec):
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

        collection_name = self.create_collection_name(project_id=project.project_id)
        texts = [ c.chunk_text.replace('\x00', '').strip() for c in chunks ]
        metadata = [ c.chunk_metadata for c in chunks]

        vectors = []
        for text in texts:
            raw_vec = self.embedding_client.embed_text(text=text, document_type=DocumentTypeEnum.DOCUMENT.value)
            flat_vec = self._flatten_vector(raw_vec)
            vectors.append(flat_vec)

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

        query_vector = self._flatten_vector(raw_vec)

        if not any(query_vector):
            return False

        results = await self.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=query_vector,
            limit=limit
        )
        return results

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

        for chunk in self.generation_client.generate_stream(prompt=full_prompt, chat_history=chat_history):
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

    async def generate_summary(self, project: Project, limit: int = 15):
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

    async def generate_summary_stream(self, project: Project, limit: int = 15):
        """
        Streaming version of generate_summary.
        Yields text chunks word-by-word so the client never hits a read timeout,
        even when the source document is entirely OCR-scanned and the LLM response
        is long (max_output_tokens=4000).
        """
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

        for chunk in self.generation_client.generate_stream(
            prompt=full_prompt,
            chat_history=chat_history,
        ):
            yield chunk
