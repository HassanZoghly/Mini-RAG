import os

from .minirag_base import SQLAlchemyBase
from sqlalchemy import Column, String, Float
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
import uuid

# Read the embedding size from the environment so the DB column stays in
# sync with whichever embedding model is configured.  Falls back to 384
# (the default model in .env.example) when the variable is absent.
_EMBEDDING_SIZE = int(os.getenv("EMBEDDING_MODEL_SIZE", "384"))


class MemoryModel(SQLAlchemyBase):
    __tablename__ = "memories"

    # استخدام UUID الأصلي لـ PostgreSQL
    memory_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Session ID للفلترة والفهرسة
    session_id = Column(String, index=True, nullable=False)

    memory_type = Column(String, nullable=False, default="short_term")
    content = Column(String, nullable=False)
    summary = Column(String, nullable=True)

    # تخزين الوقت كنص (ISO 8601) ليتوافق مع MemoryRecord لديك
    timestamp = Column(String, nullable=False)

    importance_score = Column(Float, default=0.5)

    # متجه pgvector — الحجم مرن بناءً على متغير البيئة EMBEDDING_MODEL_SIZE
    embedding = Column(Vector(_EMBEDDING_SIZE))