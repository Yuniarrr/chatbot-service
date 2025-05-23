import logging
import os
import uuid

from typing import Annotated, Optional
from fastapi import APIRouter, Form, Query, Request, UploadFile, File, Depends

from app.core.constants import SUCCESS_MESSAGE
from app.core.exceptions import InternalServerException
from app.core.response import ResponseModel
from app.core.logger import SRC_LOG_LEVELS
from app.models.conversations import ConversationForm
from app.utils.auth import (
    TokenData,
    get_verified_user,
)
from app.services.collection import collection_service

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["ROUTER"])


router = APIRouter()


@router.get("/", response_model=Optional[ResponseModel])
async def get_collection(user=Depends(get_verified_user)):
    try:
        new_conversation = await collection_service.get_active_collections()
        return ResponseModel(
            status_code=201, message=SUCCESS_MESSAGE.RETRIEVED, data=new_conversation
        )
    except Exception as e:
        raise InternalServerException(str(e))
