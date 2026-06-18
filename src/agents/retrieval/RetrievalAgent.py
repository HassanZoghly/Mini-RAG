"""
RetrievalAgent — Phase 3 improvements (items 4B/C, 8).

Changes vs original:
- Uses ``classify_task_type`` + ``RETRIEVAL_SIZES`` from ``intent_utils``
  for dynamic fetch_limit / top_n (item 4C).
- Summary queries bypass vector search entirely and pull ALL ordered
  chunks via ``ChunkModel.get_all_chunks_ordered`` (item 1 / 4C).
- Uses ``state["query_for_retrieval"]`` (the rewritten query from
  ``QueryRewriterAgent``) as the embedding input when available (item 4A).
- The asset_id post-filter now also checks the ``source`` field in
  metadata (which stores file_id after Phase 0 chunking changes).
- ``top_n`` is driven by the RETRIEVAL_SIZES table — no longer hardcoded
  to 5 (item 4B).
- ``db_client`` is an optional constructor argument; when provided,
  summary queries use it to instantiate ``ChunkModel``.
"""

from __future__ import annotations

from typing import List, Optional

from agents.base import BaseAgent, AgentState
from agents.base.intent_utils import (
    RETRIEVAL_SIZES,
    TASK_SUMMARY,
    classify_task_type,
)
from stores.llm.LLMEnums import DocumentTypeEnum


class RetrievalAgent(BaseAgent):
    """
    Embeds the (possibly rewritten) query and retrieves the most relevant
    chunks from the vector DB, with dynamic sizing based on task type.

    For summary queries, skips vector search and returns ALL ordered
    chunks for the project directly from the relational DB so that
    ``ReasoningAgent`` / ``ResponseFormatterAgent`` can hand them to
    ``SummaryGenerator`` for map-reduce processing.
    """

    def __init__(
        self,
        embedding_client,
        vectordb_client,
        nlp_controller,
        reranker_client=None,
        db_client=None,
    ) -> None:
        self._embedding_client = embedding_client
        self._vectordb_client = vectordb_client
        self._nlp_controller = nlp_controller
        self._reranker_client = reranker_client
        self._db_client = db_client  # Optional — needed for summary ordered retrieval

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def agent_name(self) -> str:
        return "RetrievalAgent"

    async def execute(self, state: AgentState) -> AgentState:
        self.validate_state(state, ["query", "project_id", "asset_ids"])

        query: str = state["query"]
        project_id: str = state["project_id"]
        asset_ids: List[str] = state.get("asset_ids") or []
        route: str = state.get("metadata", {}).get("route", "")
        project_db_id = await self._resolve_project_db_id(project_id)

        # Use the rewritten query for embedding if available (item 4A)
        retrieval_query: str = state.get("query_for_retrieval") or query

        task_type = classify_task_type(query, route)

        # ── SUMMARY: ordered full-lecture retrieval ──────────────────────
        if task_type == TASK_SUMMARY:
            chunks = await self._fetch_ordered_chunks(project_db_id, asset_ids)
            state["retrieved_chunks"] = chunks
            state["agent_trace"].append(
                f"{self.agent_name}: summary path — {len(chunks)} ordered chunks loaded"
            )
            self.log_step(f"summary: loaded {len(chunks)} ordered chunks")
            return state

        # ── NON-SUMMARY: vector search + dynamic rerank ──────────────────
        sizing = RETRIEVAL_SIZES.get(task_type, RETRIEVAL_SIZES["simple_qa"])
        fetch_limit: int = sizing["fetch_limit"]
        top_n: int = sizing["top_n"]

        self.log_step(
            f"task={task_type}, fetch_limit={fetch_limit}, top_n={top_n}, "
            f"query='{retrieval_query[:60]}'"
        )

        collection_name = self._nlp_controller.create_collection_name(
            project_id=project_db_id
        )

        # ── Embed query ──────────────────────────────────────────────────
        try:
            raw_vec = self._embedding_client.embed_text(
                text=retrieval_query,
                document_type=DocumentTypeEnum.QUERY.value,
            )
            query_vector = self._nlp_controller._flatten_vector(raw_vec)
        except Exception as exc:
            self.log_step(f"embedding failed: {exc}")
            state["retrieved_chunks"] = []
            state["agent_trace"].append(
                f"{self.agent_name}: embedding failed — 0 chunks"
            )
            return state

        if not any(query_vector):
            state["retrieved_chunks"] = []
            state["agent_trace"].append(
                f"{self.agent_name}: empty query vector — 0 chunks"
            )
            return state

        # ── Vector search ────────────────────────────────────────────────
        # Keep the threshold very low — cosine scores for valid but
        # technical queries can be well below 0.05.  top_n already caps
        # the number of chunks returned; a hard score cut only causes
        # silent empty-retrieval bugs.
        MIN_SCORE = 0.0
        try:
            raw_results = await self._vectordb_client.search_by_vector(
                collection_name=collection_name,
                vector=query_vector,
                limit=fetch_limit,
            )
        except Exception as exc:
            self.log_step(f"vector search failed: {exc}")
            raw_results = []

        # search_by_vector returns None when the collection is empty
        if raw_results is None:
            raw_results = []

        # ── Normalise + asset filter ─────────────────────────────────────
        chunks: List[dict] = []
        asset_ids_str = [str(a) for a in asset_ids] if asset_ids else []

        for doc in raw_results:
            if doc.score < MIN_SCORE:
                continue

            meta = doc.metadata or {}

            if asset_ids_str:
                # Match on either asset_id (int PK) or source (file_id string)
                doc_asset_id = str(meta.get("asset_id", ""))
                doc_source = str(meta.get("source", ""))
                if doc_asset_id not in asset_ids_str and doc_source not in asset_ids_str:
                    continue

            chunks.append({
                "text": doc.text,
                "score": doc.score,
                "metadata": meta,
            })

        # ── Rerank ──────────────────────────────────────────────────────
        if self._reranker_client and chunks:
            docs_texts = [c["text"] for c in chunks]
            self.log_step(f"reranking {len(docs_texts)} → top {top_n}")
            try:
                ranked_texts = self._reranker_client.rerank(
                    query=retrieval_query,
                    documents=docs_texts,
                    top_n=top_n,
                )
                reranked: List[dict] = []
                for r_text in ranked_texts:
                    for c in chunks:
                        if c["text"] == r_text and c not in reranked:
                            reranked.append(c)
                            break
                chunks = reranked
            except Exception as exc:
                self.log_step(f"reranking failed ({exc}), using top-{top_n} by score")
                chunks = sorted(chunks, key=lambda c: c["score"], reverse=True)[:top_n]
        else:
            # No reranker — trim by score
            chunks = sorted(chunks, key=lambda c: c["score"], reverse=True)[:top_n]

        state["retrieved_chunks"] = chunks
        state["agent_trace"].append(
            f"{self.agent_name}: {task_type} — {len(chunks)} chunks after rerank"
        )
        self.log_step(f"returning {len(chunks)} chunks")
        return state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _resolve_project_db_id(self, project_id: str) -> int:
        """
        Resolve the string project_id (UUID or slug) to the integer
        ``project_id`` (primary key) stored in the ``projects`` table.

        When ``db_client`` is available the lookup is performed via
        ``ProjectModel``.  If no ``db_client`` is configured (e.g. in
        tests) the raw value is returned unchanged so vector-search
        collection naming still works (it only needs a stable string).
        """
        if not self._db_client:
            return project_id  # type: ignore[return-value]

        try:
            from models.ProjectModel import ProjectModel

            project_model = await ProjectModel.create_instance(
                db_client=self._db_client
            )
            project = await project_model.get_project_or_create_one(
                project_id=project_id
            )
            return project.project_id
        except Exception as exc:
            self.log_step(
                f"_resolve_project_db_id failed ({exc}) — using raw project_id"
            )
            return project_id  # type: ignore[return-value]

    async def _fetch_ordered_chunks(
        self, project_id, asset_ids: List[str]
    ) -> List[dict]:
        """
        Pull ALL ordered chunks for the project from the relational DB.

        Returns a list of dicts in the same ``{text, score, metadata}``
        format used by vector-search results so downstream agents work
        uniformly.
        """
        if not self._db_client:
            self.log_step(
                "no db_client available for ordered summary retrieval — "
                "falling back to empty list"
            )
            return []

        try:
            from models.ChunkModel import ChunkModel

            chunk_model = await ChunkModel.create_instance(db_client=self._db_client)
            orm_chunks = await chunk_model.get_all_chunks_ordered(
                project_id=project_id,
                asset_ids=asset_ids or [],
                max_chunks=2000,
            )
            result = [
                {
                    "text": c.chunk_text or "",
                    "score": 1.0,           # Ordered retrieval → no relevance score
                    "metadata": c.chunk_metadata or {},
                    "_orm": c,              # Kept for SummaryGenerator label building
                }
                for c in orm_chunks
                if (c.chunk_text or "").strip()
            ]
            self.log_step(f"ordered retrieval: {len(result)} chunks")
            return result
        except Exception as exc:
            self.log_step(f"ordered retrieval failed: {exc}")
            return []