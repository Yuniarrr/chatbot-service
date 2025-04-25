import logging
from typing import Optional

from app.models.users import RegisterForm
from app.core.database import session_manager
from app.core.logger import SRC_LOG_LEVELS
from app.core.exceptions import DatabaseException
from app.models.files import (
    File,
    FileCreateModel,
    files,
    FileReadModel,
    FileUpdateModel,
)


log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["SERVICE"])


class FileService:
    async def insert_new_file(self, form_data: FileCreateModel) -> FileReadModel:
        try:
            async with session_manager.session() as db:
                new_file = await files.create(db=db, object=form_data, commit=True)
                await db.refresh(new_file)
                return FileReadModel.model_validate(new_file)
        except Exception as e:
            log.error(f"Error inserting new file: {e}")
            raise DatabaseException(str(e))

    async def get_file_by_id(self, id: str) -> Optional[FileReadModel]:
        try:
            async with session_manager.session() as db:
                file = await files.get(db=db, id=id)
                if not file:
                    return None
                return FileReadModel.model_validate(file)
        except Exception as e:
            log.error(f"Error get file by id: {e}")
            raise DatabaseException(str(e))

    async def get_files(self, skip: Optional[int] = None, limit: Optional[int] = None):
        try:
            async with session_manager.session() as db:
                files_list = await files.get_multi(db=db, skip=skip, limit=limit)
                return files_list
        except Exception as e:
            log.error(f"Error get files: {e}")
            raise DatabaseException(str(e))

    async def update_file_by_id(
        self, file_id: str, form_data: FileUpdateModel
    ) -> FileReadModel:
        try:
            async with session_manager.session() as db:
                updated_file = await files.update(
                    db=db,
                    object=form_data,
                    commit=True,
                    return_as_model=True,
                    schema_to_select=File,
                    id=file_id,
                )
                updated_file = await db.merge(updated_file)
                await db.refresh(updated_file)
                return FileReadModel.model_validate(updated_file)
        except Exception as e:
            log.error(f"Error updating file: {e}")
            raise DatabaseException(str(e))

    async def delete_file_by_id(self, id: str):
        try:
            async with session_manager.session() as db:
                await files.db_delete(db, id)
        except Exception as e:
            log.error(f"Error deleting file: {e}")
            raise DatabaseException(str(e))


file_service = FileService()
