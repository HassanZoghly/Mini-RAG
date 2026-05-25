from ..LLMInterface import LLMInterface
from ..LLMEnums import CoHereEnums, DocumentTypeEnum
import cohere
import logging
import asyncio
from typing import List, Union

class CoHereProvider(LLMInterface):

    def __init__(self, api_key: str,
                       default_input_max_characters: int=400000,
                       default_generation_max_output_tokens: int=4000,
                       default_generation_temperature: float=0.1):

        self.api_key = api_key

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.rerank_model_id = None
        self.embedding_size = None

        self.client = cohere.Client(api_key=self.api_key)
        self.enums = CoHereEnums
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def set_rerank_model(self, model_id: str):
        self.rerank_model_id = model_id

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    def generate_text(self, prompt: str, chat_history: list=[], max_output_tokens: int=None,
                            temperature: float = None):

        if not self.client or not self.generation_model_id:
            self.logger.error("CoHere client or model was not set")
            return None

        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        response = self.client.chat(
            model = self.generation_model_id,
            chat_history = chat_history,
            message = self.process_text(prompt),
            temperature = temperature,
            max_tokens = max_output_tokens
        )

        return response.text if response else None

    async def generate_stream(self, prompt: str, chat_history: list=[], max_output_tokens: int=None,
                              temperature: float = None):
        if not self.client or not self.generation_model_id:
            yield "Error: CoHere client or model not set."
            return

        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        loop = asyncio.get_event_loop()
        queue = asyncio.Queue()

        def _run():
            try:
                response = self.client.chat_stream(
                    model=self.generation_model_id,
                    chat_history=chat_history,
                    message=self.process_text(prompt),
                    temperature=temperature,
                    max_tokens=max_output_tokens
                )
                for event in response:
                    if event.event_type == "text-generation":
                        loop.call_soon_threadsafe(queue.put_nowait, event.text)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, _run)

        while True:
            token = await queue.get()
            if token is None:
                break
            if isinstance(token, Exception):
                self.logger.error(f"Streaming error: {token}")
                yield f"Error: {token}"
                break
            yield token

    # الدالة الجديدة الخاصة بالـ Reranker
    def rerank(self, query: str, documents: List[str], top_n: int = 3):
        if not self.client or not self.rerank_model_id:
            self.logger.warning("CoHere client or rerank model not set. Skipping reranking.")
            return documents[:top_n]

        try:
            response = self.client.rerank(
                model=self.rerank_model_id,
                query=query,
                documents=documents,
                top_n=top_n
            )
            # استخراج النصوص بناءً على الترتيب الجديد
            ranked_texts = [documents[res.index] for res in response.results]
            return ranked_texts
        except Exception as e:
            self.logger.error(f"Reranking error: {e}")
            return documents[:top_n]

    def embed_text(self, text: Union[str, List[str]], document_type: str = None):
        if not self.client or not self.embedding_model_id:
            return None

        if isinstance(text, str): text = [text]
        input_type = CoHereEnums.QUERY if document_type == DocumentTypeEnum.QUERY else CoHereEnums.DOCUMENT

        response = self.client.embed(
            model = self.embedding_model_id,
            texts = [ self.process_text(t) for t in text ],
            input_type = input_type,
            embedding_types=['float'],
        )
        return [ f for f in response.embeddings.float ] if response else None

    def construct_prompt(self, prompt: str, role: str):
        return {"role": role, "text": prompt}
