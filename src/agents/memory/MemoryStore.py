from .MemorySchema import MemoryRecord
from controllers.NLPController import NLPController
from stores.llm.LLMEnums import DocumentTypeEnum
from models.db_schemes.minirag.schemes.memory import MemoryModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

_CONSOLIDATION_THRESHOLD: int = 10
_CONSOLIDATION_BATCH: int = 5

class MemoryStore:
    def __init__(
        self,
        nlp_controller: NLPController,
        async_session_maker: async_sessionmaker, # تمرير الـ session maker من إعدادات قاعدة بياناتك
    ) -> None:
        self._nlp = nlp_controller
        self._async_session_maker = async_session_maker

    def _embed(self, text: str) -> List[float]:
        raw = self._nlp.embedding_client.embed_text(
            text=text,
            document_type=DocumentTypeEnum.DOCUMENT.value,
        )
        return self._nlp._flatten_vector(raw)

    def _embed_query(self, text: str) -> List[float]:
        raw = self._nlp.embedding_client.embed_text(
            text=text,
            document_type=DocumentTypeEnum.QUERY.value,
        )
        return self._nlp._flatten_vector(raw)

    async def store_memory(self, memory: MemoryRecord) -> bool:
        try:
            vector = self._embed(memory["content"])

            async with self._async_session_maker() as session:
                db_memory = MemoryModel(
                    memory_id=memory["memory_id"],
                    session_id=memory["session_id"],
                    memory_type=memory["memory_type"],
                    content=memory["content"],
                    summary=memory["summary"],
                    timestamp=memory["timestamp"],
                    importance_score=memory["importance_score"],
                    embedding=vector
                )
                session.add(db_memory)
                await session.commit()

            logger.info(
                "MemoryStore: stored memory_id=%s type=%s session=%s",
                memory["memory_id"], memory["memory_type"], memory["session_id"]
            )
            return True
        except Exception as exc:
            logger.error("MemoryStore.store_memory failed: %s", exc)
            return False

    async def retrieve_memories(
        self,
        query: str,
        session_id: str,
        k: int = 5,
        memory_type: Optional[str] = None,
    ) -> List[MemoryRecord]:
        try:
            query_vector = self._embed_query(query)
            if not any(query_vector):
                return []

            async with self._async_session_maker() as session:
                # بناء استعلام البحث
                stmt = select(MemoryModel).where(MemoryModel.session_id == session_id)

                # إضافة فلتر النوع إذا تم تمريره
                if memory_type:
                    stmt = stmt.where(MemoryModel.memory_type == memory_type)

                # الترتيب باستخدام المسافة الجيبية (Cosine Distance) الخاصة بـ pgvector
                stmt = stmt.order_by(
                    MemoryModel.embedding.cosine_distance(query_vector)
                ).limit(k)

                result = await session.execute(stmt)
                records = result.scalars().all()

                memories: List[MemoryRecord] = []
                for r in records:
                    memories.append({
                        "memory_id": str(r.memory_id),
                        "session_id": r.session_id,
                        "memory_type": r.memory_type,
                        "content": r.content,
                        "summary": r.summary or "",
                        "timestamp": r.timestamp,
                        "importance_score": float(r.importance_score)
                    })
                return memories

        except Exception as exc:
            logger.error("MemoryStore.retrieve_memories search failed: %s", exc)
            return []

    async def consolidate_memories(self, session_id: str) -> bool:
        # الكود الخاص بالدمج (Consolidation) كما هو في ملفك الأصلي بالضبط
        # لا يحتاج لتعديل لأنه يعتمد على دالتي `retrieve_memories` و `store_memory`
        pass
