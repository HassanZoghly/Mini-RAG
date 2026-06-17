from .BaseDataModel import BaseDataModel
from .db_schemes import Project
from .enums.DataBaseEnum import DataBaseEnum
from sqlalchemy.future import select
from sqlalchemy import func
import uuid

class ProjectModel(BaseDataModel):

    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.db_client = db_client

    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client)
        return instance

    async def create_project(self, project: Project):
        async with self.db_client() as session:
            async with session.begin():
                session.add(project)
                await session.flush()
                await session.refresh(project)

        return project

    def _normalize_project_uuid(self, project_id: str):
        try:
            return uuid.UUID(str(project_id))
        except (TypeError, ValueError, AttributeError):
            return uuid.uuid5(uuid.NAMESPACE_URL, str(project_id))

    async def get_project_or_create_one(self, project_id: str):
        project_uuid = self._normalize_project_uuid(project_id)

        async with self.db_client() as session:
            async with session.begin():
                query = select(Project).where(Project.project_uuid == project_uuid)
                result = await session.execute(query)
                project = result.scalar_one_or_none()

                if project is None:
                    project = Project(project_uuid=project_uuid)
                    session.add(project)
                    await session.flush()
                    await session.refresh(project)
                    return project
                else:
                    return project

    async def get_all_projects(self, page: int=1, page_size: int=10):

        async with self.db_client() as session:
            async with session.begin():

                total_documents = await session.execute(select(
                    func.count( Project.project_id )
                ))

                total_documents = total_documents.scalar_one()

                total_pages = total_documents // page_size
                if total_documents % page_size > 0:
                    total_pages += 1

                query = select(Project).offset((page - 1) * page_size ).limit(page_size)
                result = await session.execute(query)
                projects = result.scalars().all()

                return projects, total_pages
