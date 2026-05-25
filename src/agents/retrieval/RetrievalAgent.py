from agents.base import BaseAgent, AgentState
from stores.llm.LLMEnums import DocumentTypeEnum
from typing import List


class RetrievalAgent(BaseAgent):
    """
    Agent responsible for embedding the user query and retrieving the
    most semantically relevant text chunks from the vector database.

    The agent calls ``embedding_client`` and ``vectordb_client`` directly
    (rather than through ``NLPController``) because the controller's
    ``search_vector_db_collection`` method requires a full SQLAlchemy
    ``Project`` ORM object that is not available in the agent pipeline
    state.

    The collection name follows the same convention used by
    ``NLPController.create_collection_name``:
    ``collection_<vector_size>_<project_id>``

    Parameters
    ----------
    embedding_client : object
        LLM provider instance with ``embed_text`` and ``embedding_size``.
    vectordb_client : object
        Vector-DB provider instance with ``search_by_vector`` and
        ``default_vector_size``.
    nlp_controller : object
        NLPController instance; only its ``_flatten_vector`` helper is
        used for normalising raw embedding output.
    """

    def __init__(self, embedding_client, vectordb_client, nlp_controller, reranker_client=None) -> None:
        self._embedding_client = embedding_client
        self._vectordb_client = vectordb_client
        self._nlp_controller = nlp_controller
        self._reranker_client = reranker_client

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def agent_name(self) -> str:
        """Return the display name used in logs and agent trace entries."""
        return "RetrievalAgent"

    async def execute(self, state: AgentState) -> AgentState:
        """
        Embed the query and search the project's vector-DB collection.

        Steps
        -----
        1. Validate that ``query``, ``project_id``, and ``asset_ids``
           are present in *state*.
        2. Embed the query using ``embedding_client.embed_text`` with the
           QUERY document type and normalise the result via
           ``nlp_controller._flatten_vector``.
        3. Derive the collection name using the same convention as
           ``NLPController.create_collection_name``.
        4. Call ``vectordb_client.search_by_vector`` for top-10 results.
        5. Normalise every returned ``RetrievedDocument`` into a plain
           ``{text, score, metadata}`` dict; post-filter by ``asset_ids``
           when the list is non-empty.
        6. Write results to ``state["retrieved_chunks"]`` (empty list on
           failure or no results).
        7. Append a trace entry and return the updated state.

        Parameters
        ----------
        state : AgentState
            Current pipeline state.  Must contain ``query``,
            ``project_id``, and ``asset_ids``.

        Returns
        -------
        AgentState
            Updated state with ``retrieved_chunks`` populated.
        """
        self.validate_state(state, ["query", "project_id", "asset_ids"])

        query: str = state["query"]
        project_id: str = state["project_id"]
        asset_ids: List[str] = state["asset_ids"]

        self.log_step(
            f"searching for query='{query[:60]}' in project='{project_id}'"
        )

        # Derive collection name — mirrors NLPController.create_collection_name
        vector_size = self._vectordb_client.default_vector_size
        collection_name = f"collection_{vector_size}_{project_id}".strip()

        # Embed the query --------------------------------------------------
        try:
            raw_vec = self._embedding_client.embed_text(
                text=query,
                document_type=DocumentTypeEnum.QUERY.value,
            )
            query_vector = self._nlp_controller._flatten_vector(raw_vec)
        except Exception as exc:
            self.log_step(f"embedding failed: {exc}")
            state["retrieved_chunks"] = []
            state["agent_trace"].append(
                f"{self.agent_name}: embedding failed — 0 chunks retrieved"
            )
            return state

        if not any(query_vector):
            self.log_step("zero query vector — skipping search")
            state["retrieved_chunks"] = []
            state["agent_trace"].append(
                f"{self.agent_name}: empty query vector — 0 chunks retrieved"
            )
            return state

        # تحديد ما إذا كان الطلب شاملاً (تلخيص أو أسئلة) لزيادة سحب البيانات
        query_lower = query.lower()
        route = state.get("metadata", {}).get("route", "")
        is_broad_query = route in ["summary", "quiz"] or "summarize" in query_lower or "quiz" in query_lower or "ملخص" in query_lower or "امتحان" in query_lower

        # سحب 60 قطعة للملخصات لتغطية كل المحاضرات، وتقليل شرط التشابه
        fetch_limit = 60 if is_broad_query else 15
        MIN_RETRIEVAL_SCORE = 0.05 if is_broad_query else 0.30

        # Vector search ----------------------------------------------------
        try:
            raw_results = await self._vectordb_client.search_by_vector(
                collection_name=collection_name,
                vector=query_vector,
                limit=fetch_limit,
            )
        except Exception as exc:
            self.log_step(f"vector search raised an exception: {exc}")
            raw_results = None

        # Normalise results ------------------------------------------------
        chunks: List[dict] = []

        if raw_results:
            for doc in raw_results:
                if doc.score < MIN_RETRIEVAL_SCORE:
                    continue

                # Post-filter by asset_ids
                if asset_ids:
                    doc_asset_id = str((doc.metadata or {}).get("asset_id", ""))
                    if doc_asset_id not in [str(a) for a in asset_ids]:
                        continue

                chunks.append({
                    "text": doc.text,
                    "score": doc.score,
                    "metadata": doc.metadata if doc.metadata else {},
                })

        # --- NEW RERANKER STEP ---
        if self._reranker_client and len(chunks) > 0:
            # استخراج النصوص فقط لإرسالها للرينكر
            docs_texts = [c["text"] for c in chunks]

            self.log_step(f"Reranking {len(docs_texts)} chunks...")
            # اختيار أفضل 5 قطع بدقة شديدة
            ranked_texts = self._reranker_client.rerank(query=query, documents=docs_texts, top_n=5)

            # إعادة بناء الـ Chunks للحفاظ على الـ Metadata الأصلية
            reranked_chunks = []
            for r_text in ranked_texts:
                for c in chunks:
                    if c["text"] == r_text and c not in reranked_chunks:
                        reranked_chunks.append(c)
                        break
            chunks = reranked_chunks
            self.log_step(f"Kept top {len(chunks)} chunks after Reranking.")

        state["retrieved_chunks"] = chunks
        state["agent_trace"].append(
            f"{self.agent_name}: retrieved {len(chunks)} chunks (after Rerank)"
        )

        return state
