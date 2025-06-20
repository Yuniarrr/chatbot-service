import logging
import os
import uuid

from typing import Annotated, Optional
from fastapi import (
    APIRouter,
    Form,
    Query,
    Request,
    UploadFile,
    File,
    Depends,
    BackgroundTasks,
)
from pydantic import BaseModel
from uuid import UUID
from urllib.parse import urlparse

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
from app.services.file import file_service, process_file
from app.services.uploader import uploader_service
from app.models.files import (
    FileCreateModel,
    FileStatus,
    FileUpdateModel,
    ProcessFileForm,
    FileReadModel,
    UpdateFileForm,
)
from app.retrieval.vector_store import vector_store_service
from app.services.collection import collection_service
from app.models.collections import Collection

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["ROUTER"])


router = APIRouter()


@router.post("/", response_model=ResponseModel)
async def add_new_file(
    current_user: Annotated[TokenData, Depends(get_not_user)],
    request: Request,
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    collection_name: Optional[str] = Form("administration"),
    url: Optional[str] = Form(None),
    content: Optional[str] = Form(None),
):
    try:
        provided_sources = [bool(file), bool(url), bool(content)]
        if provided_sources.count(True) == 0:
            raise BadRequestException("A file, URL, or content must be provided.")
        elif provided_sources.count(True) > 1:
            raise BadRequestException(
                "Only one of file, URL, or content should be provided."
            )

        id = str(uuid.uuid4())

        if file:
            print(f"file.content_type: {file.content_type}")
            unsanitized_filename = file.filename
            filename = os.path.basename(unsanitized_filename)

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
        elif content is not None:
            filename = f"{id}_content.txt"
            contents, file_path = uploader_service.upload_text_content(
                content, filename
            )

            _new_file = FileCreateModel(
                id=id,
                user_id=current_user.id,
                file_name=filename,
                file_path=file_path,
                status=FileStatus.AWAITING,
                meta={
                    "name": filename,
                    "content_type": "text/plain",
                    "size": len(content.encode("utf-8")),
                    "collection_name": collection_name,
                },
            )

            await file_service.insert_new_file(_new_file)
        elif url:
            _new_file = FileCreateModel(
                **{
                    "id": id,
                    "user_id": current_user.id,
                    "file_name": url,
                    "file_path": url,
                    "status": FileStatus.AWAITING,
                    "meta": {
                        "name": url,
                        "content_type": "application/octet-stream",
                        "size": 0,
                        "collection_name": collection_name,
                    },
                }
            )

            await file_service.insert_new_file(_new_file)

        background_tasks.add_task(
            process_file, _new_file, file, url, collection_name, _new_file.meta
        )

        return ResponseModel(
            status_code=201, message=SUCCESS_MESSAGE.CREATED, data=_new_file
        )
    except Exception as e:
        raise InternalServerException(str(e))


@router.get("/", response_model=ResponseModel)
async def get_all_file(
    current_user: Annotated[TokenData, Depends(get_not_user)],
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1),
    search: Optional[str] = Query(None),
):
    try:
        files = await file_service.get_files(skip=skip, limit=limit, search=search)

        return ResponseModel(
            status_code=200, message=SUCCESS_MESSAGE.RETRIEVED, data=files
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


@router.patch("/{file_id}", response_model=ResponseModel)
async def update_file_by_id(
    file_id: UUID,
    current_user: Annotated[TokenData, Depends(get_not_user)],
    form_data: UpdateFileForm,
):
    try:
        file = await file_service.get_file_by_id(str(file_id))
        if not file:
            raise NotFoundException(ERROR_MESSAGES.NOT_FOUND("file"))

        if file.status == FileStatus.AWAITING:
            raise BadRequestException("File sedang dalam proses upload")

        update_file = FileUpdateModel(**{"status": FileStatus.AWAITING})
        await file_service.update_file_by_id(file.id, update_file)

        if file.file_name != form_data.file_name:
            try:
                new_file_path = uploader_service.rename_file(
                    file.file_name, form_data.file_name
                )
                form_data.file_path = new_file_path
            except FileNotFoundError:
                raise NotFoundException("File fisik tidak ditemukan di storage.")
            except FileExistsError:
                raise BadRequestException("File dengan nama baru sudah ada.")

        updated_file = await file_service.update_file_by_id(str(file_id), form_data)

        if form_data.meta is not None:
            form_data.meta["source"] = updated_file.file_path
            await vector_store_service.update_metadata_by_file_id(
                str(file_id),
                form_data.meta,
            )

        if form_data.meta["collection_name"] != file.meta["collection_name"]:
            collection_id = await vector_store_service.get_collection_id_by_name(
                form_data.meta["collection_name"]
            )
            await vector_store_service.update_collection_by_file_id(
                str(file_id), collection_id
            )

            await vector_store_service.refetch_bm25(
                collection_name=form_data.meta["collection_name"]
            )
            await vector_store_service.refetch_vector_store(
                collection_name=form_data.meta["collection_name"]
            )

        update_file = FileUpdateModel(**{"status": FileStatus.SUCCESS})
        await file_service.update_file_by_id(file.id, update_file)

        return ResponseModel(
            status_code=200, message=SUCCESS_MESSAGE.UPDATED, data=form_data
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

        collection_name = file.meta.get("collection_name")
        if not collection_name:
            raise ValueError("Missing collection name in file metadata")

        vector_ids = await vector_store_service.get_vector_ids(str(file_id))
        vector_ids_list = [item["id"] for item in vector_ids]

        if vector_ids_list:
            await vector_store_service.delete_by_ids(vector_ids_list, collection_name)

        if delete_file:
            uploader_service.delete_from_local(file.file_name)
            await file_service.delete_file_by_id(str(file_id))
        else:
            await file_service.update_file_by_id(
                str(file_id),
                FileUpdateModel(status=FileStatus.DETACHED),
            )

        await vector_store_service.refetch_bm25(collection_name)
        await vector_store_service.refetch_vector_store(collection_name)

        return ResponseModel(status_code=200, message=SUCCESS_MESSAGE.DELETED)
    except Exception as e:
        raise InternalServerException(str(e))
