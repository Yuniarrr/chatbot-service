import logging
import uuid

from typing import Optional
from fastapi import APIRouter, Depends

from app.core.constants import ERROR_MESSAGES, SUCCESS_MESSAGE
from app.core.exceptions import (
    InternalServerException,
    NotFoundException,
    BadRequestException,
)
from app.core.response import ResponseModel
from app.core.logger import SRC_LOG_LEVELS
from app.models.collections import UpdateCollectionForm
from app.utils.auth import (
    get_verified_user,
)
from app.services.collection import collection_service
from app.retrieval.chain import chain_service

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["ROUTER"])


router = APIRouter()


@router.get("/", response_model=Optional[ResponseModel])
async def get_collection(user=Depends(get_verified_user)):
    try:
        new_conversation = await collection_service.get_collections()
        return ResponseModel(
            status_code=200, message=SUCCESS_MESSAGE.RETRIEVED, data=new_conversation
        )
    except Exception as e:
        raise InternalServerException(str(e))


@router.patch("/{collection_id}", response_model=Optional[ResponseModel])
async def get_collection(
    collection_id: uuid.UUID,
    form_data: UpdateCollectionForm,
    user=Depends(get_verified_user),
):
    try:
        is_col_exist = await collection_service.get_collection_by_id(str(collection_id))

        if is_col_exist is None:
            raise NotFoundException(ERROR_MESSAGES.NOT_FOUND("Collection"))

        update_status = chain_service.update_collection_status(
            collection_name=is_col_exist.name, is_active=form_data.status
        )

        if not update_status:
            raise BadRequestException("Gagal memperbarui status")

        update_col = await collection_service.update_status_by_collection_id(
            collection_id=collection_id, status=form_data.status
        )
        return ResponseModel(
            status_code=200, message=SUCCESS_MESSAGE.UPDATED, data=update_col
        )
    except Exception as e:
        raise InternalServerException(str(e))
