import logging
import os
import uuid

from typing import Annotated, Optional
from fastapi import APIRouter, Form, Query, Request, UploadFile, File, Depends

from app.core.constants import ERROR_MESSAGES, SUCCESS_MESSAGE
from app.core.exceptions import InternalServerException, NotFoundException
from app.core.response import ResponseModel
from app.core.logger import SRC_LOG_LEVELS
from app.utils.auth import (
    TokenData,
    get_verified_user,
)
from app.services.feedback import feedback_service

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["ROUTER"])


router = APIRouter()


@router.get("/", response_model=ResponseModel)
async def get_all_opportunity(
    current_user: Annotated[TokenData, Depends(get_verified_user)],
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1),
    search: Optional[str] = Query(None),
):
    try:
        feedbacks = await feedback_service.get_feedbacks(
            skip=skip, limit=limit, search=search
        )

        return ResponseModel(
            status_code=200,
            message=SUCCESS_MESSAGE.RETRIEVED,
            data=feedbacks,
        )
    except Exception as e:
        raise InternalServerException(str(e))


@router.delete("/{feedback_id}", response_model=ResponseModel)
async def delete_opportunity_by_id(
    feedback_id: uuid.UUID,
    current_user: Annotated[TokenData, Depends(get_verified_user)],
):
    try:
        opportunity = await feedback_service.get_feedback_by_id(str(feedback_id))
        if not opportunity:
            raise NotFoundException(ERROR_MESSAGES.NOT_FOUND("opportunity"))

        await feedback_service.delete_feedback_by_id(str(feedback_id))

        return ResponseModel(status_code=200, message=SUCCESS_MESSAGE.DELETED)
    except Exception as e:
        raise InternalServerException(str(e))
