import logging
import os
import uuid

from typing import Annotated
from fastapi import APIRouter, Request, UploadFile, File, Depends

from app.core.response import ResponseModel
from app.core.constants import ERROR_MESSAGES, SUCCESS_MESSAGE
from app.core.logger import SRC_LOG_LEVELS
from app.core.exceptions import (
    BadRequestException,
    InternalServerException,
)
from app.utils.auth import (
    TokenData,
    get_not_user,
)
from app.services.file import file_service
from app.services.uploader import uploader_service
from app.models.files import FileCreateModel, FileStatus, ProcessFileForm, FileReadModel
from app.services.retrieval import retrieval_service

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["ROUTER"])


router = APIRouter()


@router.post("/", response_model=ResponseModel)
async def add_new_file(
    current_user: Annotated[TokenData, Depends(get_not_user)],
    request: Request,
    file: UploadFile = File(...),
):
    log.info(f"file.content_type: {file.content_type}")
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
                "status": FileStatus.SUCCESS,
                "meta": {
                    "name": name,
                    "content_type": file.content_type,
                    "size": len(contents),
                },
            }
        )

        await file_service.insert_new_file(_new_file)

        try:
            await retrieval_service.process_file(ProcessFileForm(file_id=id))
            file_item = await file_service.get_file_by_id(id=id)
        except Exception as e:
            log.exception(e)
            log.error(f"Error processing file: {file_item.id}")
            file_item = FileReadModel(
                **{
                    **file_item.model_dump(),
                    "error": str(e.detail) if hasattr(e, "detail") else str(e),
                }
            )

        if file_item:
            return ResponseModel(
                status_code=201, message=SUCCESS_MESSAGE.CREATED, data=file_item
            )
        else:
            raise BadRequestException(ERROR_MESSAGES.FAILED_UPLOAD)

    except Exception as e:
        raise InternalServerException(str(e))
