import logging
import os
import uuid

from typing import Annotated, Optional
from fastapi import APIRouter, Form, Query, Request, UploadFile, File, Depends
from pydantic import BaseModel

from app.core.response import ResponseModel
from app.core.constants import ERROR_MESSAGES, SUCCESS_MESSAGE
from app.core.logger import SRC_LOG_LEVELS
from app.core.exceptions import (
    BadRequestException,
    InternalServerException,
    NotFoundException,
)
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
from app.retrieval.vector import VECTOR_DB_CLIENT

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
                },
            }
        )

        await file_service.insert_new_file(_new_file)

        try:
            file_item = None
            await retrieval_service.process_file(
                ProcessFileForm(file_id=id, collection_name=collection_name)
            )
            file_item = await file_service.get_file_by_id(id=id)
        except Exception as e:
            log.error(f"Error processing file: {file_item.id}")
            file_item = FileReadModel(
                **{
                    **file_item.model_dump(),
                    "error": str(e.detail) if hasattr(e, "detail") else str(e),
                }
            )
            print("ERROR")
            await file_service.update_file_by_id(
                file.id,
                FileUpdateModel(**{"status": FileStatus.FAILED}),
            )

        if file_item:
            return ResponseModel(
                status_code=201, message=SUCCESS_MESSAGE.CREATED, data=file_item
            )
        else:
            raise BadRequestException(ERROR_MESSAGES.FAILED_UPLOAD)

    except Exception as e:
        raise InternalServerException(str(e))


@router.get("/{file_id}", response_model=ResponseModel)
async def get_file_by_id(
    file_id: str,
    current_user: Annotated[TokenData, Depends(get_not_user)],
):
    try:
        file = await file_service.get_file_by_id(file_id)
        if not file:
            raise NotFoundException(ERROR_MESSAGES.NOT_FOUND("file"))

        return ResponseModel(
            status_code=200, message=SUCCESS_MESSAGE.RETRIEVED, data=file
        )
    except Exception as e:
        raise InternalServerException(str(e))


@router.delete("/{file_id}", response_model=ResponseModel)
async def delete_file_by_id(
    file_id: str,
    current_user: Annotated[TokenData, Depends(get_not_user)],
    delete_file: bool = Query(False, description="Delete file from database"),
):
    try:
        file = await file_service.get_file_by_id(file_id)
        if not file:
            raise NotFoundException(ERROR_MESSAGES.NOT_FOUND("file"))

        collection_name = file.meta.get("collection_name")
        # Remove content from the vector database
        VECTOR_DB_CLIENT.delete(
            collection_name=collection_name, filter={"file_id": file_id}
        )

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

        return ResponseModel(status_code=200, message=SUCCESS_MESSAGE.DELETED)
    except Exception as e:
        raise InternalServerException(str(e))
