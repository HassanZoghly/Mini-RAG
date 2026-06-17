from .BaseDataModel import BaseDataModel
from .db_schemes import DataChunk
from .enums.DataBaseEnum import DataBaseEnum
from sqlalchemy.future import select
from sqlalchemy import func, delete

class ChunkModel(BaseDataModel):

    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.db_client = db_client

    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client)
        return instance

    async def create_chunk(self, chunk: DataChunk):

        async with self.db_client() as session:
            async with session.begin():
                session.add(chunk)
            await session.commit()
            await session.refresh(chunk)
        return chunk

    async def get_chunk(self, chunk_id: str):

        async with self.db_client() as session:
            result = await session.execute(select(DataChunk).where(DataChunk.chunk_id == chunk_id))
            chunk = result.scalar_one_or_none()
        return chunk

    async def insert_many_chunks(self, chunks: list, batch_size: int=100):

        async with self.db_client() as session:
            async with session.begin():
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    session.add_all(batch)
            await session.commit()
        return len(chunks)

    async def delete_chunks_by_project_id(self, project_id: int):
        async with self.db_client() as session:
            stmt = delete(DataChunk).where(DataChunk.chunk_project_id == project_id)
            result = await session.execute(stmt)
            await session.commit()
        return result.rowcount

    async def get_poject_chunks(self, project_id: int, page_no: int=1, page_size: int=50):
        async with self.db_client() as session:
            stmt = select(DataChunk).where(DataChunk.chunk_project_id == project_id).offset((page_no - 1) * page_size).limit(page_size)
            result = await session.execute(stmt)
            records = result.scalars().all()
        return records

    async def get_total_chunks_count(self, project_id: int):
        total_count = 0
        async with self.db_client() as session:
            count_sql = select(func.count(DataChunk.chunk_id)).where(DataChunk.chunk_project_id == project_id)
            records_count = await session.execute(count_sql)
            total_count = records_count.scalar()

        return total_count

    async def get_all_chunks_ordered(self, project_id: int, asset_ids: list = None,
                                       max_chunks: int = 1000):
        """
        Return *every* chunk for a project (optionally restricted to
        ``asset_ids``), ordered by ``(chunk_asset_id, chunk_order)`` so the
        original lecture structure/ordering is preserved.

        Used by the full-lecture summary pipeline (item 1), which needs
        the complete, ordered document rather than a small set of
        similarity-ranked chunks.

        ``max_chunks`` is a safety cap to avoid pulling an unbounded
        number of rows for extremely large projects.
        """
        async with self.db_client() as session:
            stmt = select(DataChunk).where(DataChunk.chunk_project_id == project_id)

            if asset_ids:
                normalized_ids = []
                for a in asset_ids:
                    try:
                        normalized_ids.append(int(a))
                    except (TypeError, ValueError):
                        continue
                if normalized_ids:
                    stmt = stmt.where(DataChunk.chunk_asset_id.in_(normalized_ids))

            stmt = stmt.order_by(DataChunk.chunk_asset_id, DataChunk.chunk_order).limit(max_chunks)

            result = await session.execute(stmt)
            records = result.scalars().all()
        return records
