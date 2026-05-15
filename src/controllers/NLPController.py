from .BaseController import BaseController
from models.db_schemes import Project
from typing import List
from stores.llm.LLMEnums import DocumentTypeEnum
from models.db_schemes.data_chunk import DataChunk
import json

class NLPController(BaseController):

    def __init__(self, generation_client, embedding_client, vectordb_client, template_parser):
        super().__init__()

        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.vectordb_client = vectordb_client
        self.template_parser = template_parser

    def create_collection_name(self, project_id: str):
        return f"Collection{project_id}".strip()

    def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return self.vectordb_client.delete_collection(collection_name=collection_name)

    def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = self.vectordb_client.get_collection_info(collection_name=collection_name)

        return json.loads(
            json.dumps(collection_info, default=lambda x:x.__dict__)
        )

    def index_into_vector_db(self, project: Project, chunks: List[DataChunk],
                                    chunks_ids: List[int],
                                    do_reset: bool = False):

        collection_name = self.create_collection_name(project_id=project.project_id)

        texts = [c.chunk_text for c in chunks]
        metadata = [c.chunk_metadata for c in chunks]
        vectors = [
            self.embedding_client.embed_text(text=text,
                                            document_type=DocumentTypeEnum.DOCUMENT.value)
            for text in texts
        ]

        _ = self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset
        )

        _ = self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            metadata=metadata,
            vectors=vectors,
            record_ids=chunks_ids
        )

        return True

    def search_vector_db_collection(self, project: Project, text: str, limit: int = 10):

        collection_name = self.create_collection_name(project_id=project.project_id)

        vector = self.embedding_client.embed_text(text=text,
                                                document_type=DocumentTypeEnum.QUERY.value)

        if not vector or len(vector) == 0:
            return False

        results = self.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=vector,
            limit=limit
        )

        if not results:
            return False

        return results

    def answer_rag_question(self, project: Project, query: str, limit: int=10):

        answer, full_prompt, chat_history = None, None, None

        # Fetch more documents initially for re-ranking
        retrieved_document = self.search_vector_db_collection(
            project=project,
            text=query,
            limit=limit * 3
        )

        if not retrieved_document or len(retrieved_document) == 0:
            return answer, full_prompt, chat_history

        # Rerank documents using Cohere if the embedding/generation client is Cohere
        # We can extract the underlying cohere client if it exists.
        reranked_documents = retrieved_document
        if hasattr(self.embedding_client, 'client') and self.embedding_client.__class__.__name__ == 'CoHereProvider':
            try:
                cohere_client = self.embedding_client.client
                docs_texts = [doc.text for doc in retrieved_document]

                # Using English rerank model by default for educational RAG
                rerank_results = cohere_client.rerank(
                    query=query,
                    documents=docs_texts,
                    top_n=limit,
                    model='rerank-english-v3.0'
                )

                # Re-order the retrieved_document list based on rerank results
                reranked_documents = []
                for result in rerank_results.results:
                    reranked_documents.append(retrieved_document[result.index])
            except Exception as e:
                print(f"Error during reranking: {e}")
                reranked_documents = retrieved_document[:limit]
        else:
            reranked_documents = retrieved_document[:limit]

        system_prompt = self.template_parser.get("rag", "system_prompt")

        documents_prompts = "\n".join([
            self.template_parser.get("rag", "document_prompt", {
                    "doc_num": idx + 1,
                    "chunk_text": doc.text
                })
            for idx, doc in enumerate(reranked_documents)
        ])

        footer_prompt = self.template_parser.get("rag", "footer_prompt", {
            "query": query
        })

        chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt,
                role=self.generation_client.enums.SYSTEM.value,
            )
        ]

        full_prompt = "\n\n".join([ documents_prompts,  footer_prompt])

        answer = self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=chat_history
        )

        return answer, full_prompt, chat_history

    def summarize_lecture(self, project: Project, chunks: List[DataChunk]):
        if not chunks:
            return None

        # Combine chunks into a single text block
        # We assume the chunks fit into the LLM context window for a summary
        lecture_text = "\n".join([c.chunk_text for c in chunks])

        system_prompt = "You are an expert tutor and summarizer. Please provide a comprehensive and structured summary of the following lecture material. Your response MUST be formatted in Markdown (use headings, bullet points, and bold text)."

        chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt,
                role=self.generation_client.enums.SYSTEM.value,
            )
        ]

        full_prompt = f"Please summarize the following lecture material:\n\n{lecture_text}\n\nSummary:"

        answer = self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=chat_history
        )

        return answer
