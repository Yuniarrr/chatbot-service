import logging
from typing import Optional

from fastapi import UploadFile

from app.models.users import RegisterForm
from app.core.database import session_manager
from app.core.logger import SRC_LOG_LEVELS
from app.core.exceptions import DatabaseException
from app.models.files import (
    File,
    FileCreateModel,
    FileStatus,
    files,
    FileReadModel,
    FileUpdateModel,
)
from app.core.exceptions import InternalServerException
from app.retrieval.embed import embedding_service
from app.retrieval.vector_store import vector_store_service

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
                if not files_list:
                    return []
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
            print(f"Deleting file with ID: {id}")
            if not id:
                raise ValueError("File ID is required for deletion.")

            async with session_manager.session() as db:
                await files.db_delete(db=db, id=id, allow_multiple=False)
        except Exception as e:
            log.error(f"Error deleting file: {e}")
            raise DatabaseException(str(e))


file_service = FileService()


async def process_file(
    _new_file: FileCreateModel,
    file: Optional[UploadFile] = None,
    url: Optional[str] = None,
    collection_name: Optional[str] = None,
):
    try:
        if file:
            loader_document = embedding_service.loader(
                _new_file.file_name,
                _new_file.meta.get("content_type"),
                _new_file.file_path,
            )
        elif url:
            loader_document = embedding_service.loader_url(url)

        print("loader_document")
        print(loader_document)

        splitted_document = embedding_service.split_document(loader_document)

        splitted_document = embedding_service.add_addtional_data_to_docs(
            docs=splitted_document,
            file_id=str(_new_file.id),
            file_name=_new_file.file_name,
        )

        await vector_store_service.add_vectostore(splitted_document, collection_name)

        update_file = FileUpdateModel(**{"status": FileStatus.SUCCESS})
        await file_service.update_file_by_id(_new_file.id, update_file)
    except Exception as e:
        update_file = FileUpdateModel(**{"status": FileStatus.FAILED})
        await file_service.update_file_by_id(_new_file.id, update_file)

        log.error(f"Error processing file: {_new_file.id}")
        raise InternalServerException(f"Error in processing file {e}")
