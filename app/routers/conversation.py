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
from app.services.conversation import conversation_service

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["ROUTER"])


router = APIRouter()


@router.post("/new", response_model=Optional[ResponseModel])
async def create_new_chat(form_data: ConversationForm, user=Depends(get_verified_user)):
    try:
        print(f"form_data: {form_data}")
        new_conversation = await conversation_service.create_new_conversation(
            title=form_data.title,
            user_id=user.id,
        )
        return ResponseModel(
            status_code=201, message=SUCCESS_MESSAGE.CREATED, data=new_conversation
        )
    except Exception as e:
        raise InternalServerException(str(e))
