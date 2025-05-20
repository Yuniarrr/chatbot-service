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
from app.retrieval.embed import embedding_service
from app.retrieval.vector_store import vector_store_service
from app.task import queue, process_uploaded_file
from app.retrieval.loaders import CustomITSLoader

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
    document_type: Optional[str] = Form(None),
    topik: Optional[str] = Form(None),
):
    try:
        if not file and not url:
            raise BadRequestException("Either a file or URL must be provided.")

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
                        "document_type": document_type,
                        "topik": topik,
                    },
                }
            )

            await file_service.insert_new_file(_new_file)
        elif url:
            parsed = urlparse(url)
            _new_file = FileCreateModel(
                **{
                    "id": id,
                    "user_id": current_user.id,
                    "file_name": url,
                    "file_path": url,
                    "status": FileStatus.AWAITING,
                    "meta": {
                        "name": f"{parsed.scheme}://{parsed.netloc}",
                        "content_type": "application/octet-stream",
                        "size": 0,
                        "collection_name": collection_name,
                        "document_type": document_type,
                        "topik": topik,
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

        await file_service.update_file_by_id(str(file_id), form_data)

        print("form_data.meta:", form_data.meta)

        if form_data.meta is not None:
            await vector_store_service.update_metadata_by_file_id(
                str(file_id),
                form_data.meta,
            )

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

        return ResponseModel(status_code=200, message=SUCCESS_MESSAGE.DELETED)
    except Exception as e:
        raise InternalServerException(str(e))
