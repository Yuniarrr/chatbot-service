import logging
import os
import uuid

from typing import Annotated, Optional
from fastapi import APIRouter, Form, Query, Request, UploadFile, File, Depends

from app.core.constants import ERROR_MESSAGES, SUCCESS_MESSAGE
from app.core.exceptions import InternalServerException, NotFoundException
from app.core.response import ResponseModel
from app.core.logger import SRC_LOG_LEVELS
from app.models.conversations import ConversationForm
from app.utils.auth import (
    TokenData,
    get_verified_user,
)
from app.services.conversation import conversation_service
from app.services.message import message_service

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


@router.get("/", response_model=Optional[ResponseModel])
async def get_conversations(
    user=Depends(get_verified_user),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1),
):
    try:
        conversations = await conversation_service.get_unique_conversation_items(
            skip=skip, limit=limit
        )
        return ResponseModel(
            status_code=200, message=SUCCESS_MESSAGE.RETRIEVED, data=conversations
        )
    except Exception as e:
        raise InternalServerException(str(e))


@router.get("/{param}", response_model=Optional[ResponseModel])
async def get_conversation(
    param: str,
    user=Depends(get_verified_user),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1),
):
    try:
        if param.startswith("whatsapp") or param.startswith("08"):
            conversations = await conversation_service.get_conversation_by_sender(
                sender=param, skip=skip, limit=limit
            )
        else:
            conversations = await conversation_service.get_conversations_by_user_id(
                user_id=param, skip=skip, limit=limit
            )
        return ResponseModel(
            status_code=200, message=SUCCESS_MESSAGE.RETRIEVED, data=conversations
        )
    except Exception as e:
        raise InternalServerException(str(e))


@router.delete("/{conversation_id}", response_model=Optional[ResponseModel])
async def get_conversation(
    conversation_id: uuid.UUID,
    user=Depends(get_verified_user),
):
    try:
        is_conversation_exist = await conversation_service.get_conversation_by_id(
            str(conversation_id)
        )
        if is_conversation_exist is None:
            raise NotFoundException(ERROR_MESSAGES.NOT_FOUND("percakapan"))

        await conversation_service.delete_conversation_by_id(str(conversation_id))

        return ResponseModel(status_code=200, message=SUCCESS_MESSAGE.DELETED)
    except Exception as e:
        raise InternalServerException(str(e))


@router.delete("/", response_model=Optional[ResponseModel])
async def get_conversation(
    sender: str = Query(),
    user=Depends(get_verified_user),
):
    try:
        if sender.startswith("whatsapp"):
            await conversation_service.delete_conversation_by_sender(sender)
        else:
            await conversation_service.delete_conversation_by_user_id(sender)

        return ResponseModel(status_code=200, message=SUCCESS_MESSAGE.DELETED)
    except Exception as e:
        raise InternalServerException(str(e))


@router.get("/message/{param}", response_model=Optional[ResponseModel])
async def get_conversation(
    param: str,
    user=Depends(get_verified_user),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1),
):
    try:
        messages = await message_service.get_messages_by_conversation_id(
            param, skip, limit
        )
        return ResponseModel(
            status_code=200, message=SUCCESS_MESSAGE.RETRIEVED, data=messages
        )
    except Exception as e:
        raise InternalServerException(str(e))
