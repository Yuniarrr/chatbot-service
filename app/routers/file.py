import logging
import os
import uuid

from typing import Annotated, Optional
from fastapi import APIRouter, Form, Query, Request, UploadFile, File, Depends
from pydantic import BaseModel
from uuid import UUID

from app.core.response import ResponseModel
from app.core.constants import ERROR_MESSAGES, SUCCESS_MESSAGE
from app.core.logger import SRC_LOG_LEVELS
from app.core.exceptions import (
    BadRequestException,
    InternalServerException,
    NotFoundException,
)
from app.retrieval.loaders import Loader
from app.task import process_uploaded_file
from app.utils.auth import (
    TokenData,
    get_not_user,
)
from app.services.file import file_service
from app.services.uploader import uploader_service
from app.models.files import (
    FileCreateModel,
    FileStatus,
    FileUpdateModel,
    ProcessFileForm,
    FileReadModel,
)
from app.services.retrieval import retrieval_service
from app.retrieval.embed import embedding_service
from app.retrieval.vector_store import vector_store_service
from app.task import queue, process_uploaded_file

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["ROUTER"])


router = APIRouter()


@router.post("/", response_model=ResponseModel)
async def add_new_file(
    current_user: Annotated[TokenData, Depends(get_not_user)],
    request: Request,
    file: UploadFile = File(...),
    collection_name: Optional[str] = Form("administration"),
):
    print(f"file.content_type: {file.content_type}")
    print(collection_name)
    try:
        unsanitized_filename = file.filename
        filename = os.path.basename(unsanitized_filename)

        id = str(uuid.uuid4())
        name = filename
        filename = f"{id}_{filename}"
        contents, file_path = uploader_service.upload_file(file.file, filename)

        _new_file = FileCreateModel(
            **{
                "id": id,
                "user_id": current_user.id,
                "file_name": filename,
                "file_path": file_path,
                "status": FileStatus.AWAITING,
                "meta": {
                    "name": name,
                    "content_type": file.content_type,
                    "size": len(contents),
                    "collection_name": collection_name,
                },
            }
        )

        await file_service.insert_new_file(_new_file)

        try:
            loader_document = embedding_service.loader(
                _new_file.file_name,
                _new_file.meta.get("content_type"),
                _new_file.file_path,
            )

            splitted_document = embedding_service.split_document(loader_document)

            splitted_document = embedding_service.add_addtional_data_to_docs(
                docs=splitted_document,
                file_id=str(_new_file.id),
                file_name=_new_file.file_name,
            )

            vector_store_service.add_vectostore(splitted_document, collection_name)
        except Exception as e:
            update_file = FileUpdateModel(**{"status": FileStatus.FAILED})
            await file_service.update_file_by_id(_new_file.id, update_file)

            log.error(f"Error processing file: {_new_file.id}")
            raise InternalServerException(f"Error in processing file {e}")

        update_file = FileUpdateModel(**{"status": FileStatus.SUCCESS})
        updated = await file_service.update_file_by_id(_new_file.id, update_file)

        return ResponseModel(
            status_code=201, message=SUCCESS_MESSAGE.CREATED, data=updated
        )

        # return ResponseModel(
        #     status_code=201, message=SUCCESS_MESSAGE.CREATED, data=_new_file
        # )
    except Exception as e:
        raise InternalServerException(str(e))


@router.get("/", response_model=ResponseModel)
async def get_all_file(
    current_user: Annotated[TokenData, Depends(get_not_user)],
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1),
):
    try:
        files = await file_service.get_files(skip=skip, limit=limit)

        return ResponseModel(
            status_code=200, message=SUCCESS_MESSAGE.RETRIEVED, data=files["data"]
        )
    except Exception as e:
        raise InternalServerException(str(e))


@router.get("/{file_id}", response_model=ResponseModel)
async def get_file_by_id(
    file_id: UUID,
    current_user: Annotated[TokenData, Depends(get_not_user)],
):
    try:
        file = await file_service.get_file_by_id(str(file_id))
        if not file:
            raise NotFoundException(ERROR_MESSAGES.NOT_FOUND("file"))

        return ResponseModel(
            status_code=200, message=SUCCESS_MESSAGE.RETRIEVED, data=file
        )
    except Exception as e:
        raise InternalServerException(str(e))


@router.delete("/{file_id}", response_model=ResponseModel)
async def delete_file_by_id(
    file_id: UUID,
    current_user: Annotated[TokenData, Depends(get_not_user)],
    delete_file: bool = Query(False, description="Delete file from database"),
):
    try:
        file = await file_service.get_file_by_id(str(file_id))
        if not file:
            raise NotFoundException(ERROR_MESSAGES.NOT_FOUND("file"))

        await file_service.update_file_by_id(
            file.id,
            FileUpdateModel(
                **{
                    "status": FileStatus.DETACHED,
                }
            ),
        )

        if delete_file:
            uploader_service.delete_from_local(file.file_name)
            await file_service.delete_file_by_id(file.id)

        collection_name = file.meta.get("collection_name")
        vector_ids = await vector_store_service.get_vector_ids(
            str(file_id),
        )

        vector_ids_list = [item["id"] for item in vector_ids]

        vector_store_service.delete_by_ids(vector_ids_list, collection_name)

        return ResponseModel(status_code=200, message=SUCCESS_MESSAGE.DELETED)
    except Exception as e:
        raise InternalServerException(str(e))
